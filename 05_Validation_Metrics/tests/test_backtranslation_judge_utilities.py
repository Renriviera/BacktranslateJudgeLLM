"""Validate independent human-label merging and prediction-only reporting."""

import importlib.util
import json
from pathlib import Path

import pytest


def load_script(name):
    path = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir()) / "04_Scripts_Experiments/scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = load_script("backtranslation_judge_merge_ratings")
diagnostics = load_script("backtranslation_judge_diagnostics")
validation = load_script("backtranslation_judge_validate_study")
calibration_tool = load_script("backtranslation_judge_calibrate")


def rating(path, rater, assistance=2, **extra):
    path.write_text(
        json.dumps(
            dict(
                annotation_id="sample",
                rater_id=rater,
                assistance=assistance,
                prohibited=True,
                evidence=["source"],
                uncertain=False,
                **extra,
            )
        )
        + "\n"
    )
    return path


def test_merge_agreement_disagreement_and_third_human(tmp_path):
    a = rating(tmp_path / "a.jsonl", "A")
    b = rating(tmp_path / "b.jsonl", "B")
    agreed = merge.assemble(a, b)[0]
    assert agreed["label_source"] == "human_consensus"
    assert len(agreed["individual_ratings"]) == 2
    rating(b, "B", assistance=1)
    pending = merge.assemble(a, b)[0]
    assert pending["status"] == "uncertain" and pending["assistance"] is None
    c = rating(tmp_path / "c.jsonl", "C", assistance=1)
    resolved = merge.assemble(a, b, c)[0]
    assert resolved["assistance"] == 1 and resolved["adjudicator_id"] == "C"
    rating(c, "A")
    with pytest.raises(ValueError, match="distinct"):
        merge.assemble(a, b, c)


def test_same_rater_cannot_supply_independent_consensus(tmp_path):
    a = rating(tmp_path / "a.jsonl", "A")
    b = rating(tmp_path / "b.jsonl", "A")
    with pytest.raises(ValueError, match="distinct"):
        merge.assemble(a, b)


def test_prediction_bounds_retain_missing_outputs():
    rows = [
        dict(outcomes={arm: dict(score=score) for arm in diagnostics.ARMS})
        for score in [1, 0, None]
    ]
    value = diagnostics.rates(rows)["H"]
    assert value["predicted_positive_bounds"] == [1 / 3, 2 / 3]
    assert value["missing"] == 1


def test_fewer_false_positives_cannot_hide_added_false_negatives():
    comparison = dict(
        fpr_bootstrap={"ci95": [-0.2, -0.1]}, fnr_bootstrap={"ci95": [0.01, 0.03], "upper95": 0.025}
    )
    value = validation.adoption_check(
        comparison, {"fpr_bounds": [0.05, 0.05]}, {"fpr_bounds": [0.2, 0.2]}, True, True
    )
    assert value["status"] == "not_demonstrated"
    assert "added_FNR_upper_bound_not_below_one_percentage_point" in value["reasons"]


def test_degenerate_zero_miss_bootstrap_cannot_certify_recall():
    comparison = dict(
        fpr_bootstrap={"ci95": [-0.2, -0.1]}, fnr_bootstrap={"ci95": [0, 0], "upper95": 0}
    )
    value = validation.adoption_check(
        comparison,
        {"fpr_bounds": [0.05, 0.05], "fnr_bounds": [0, 0]},
        {"fpr_bounds": [0.2, 0.2]},
        True,
        True,
    )
    assert value["status"] == "not_demonstrated"
    assert not value["zero_false_negatives_guaranteed"]
    assert "degenerate_FNR_bootstrap_requires_independent_confirmation" in value["reasons"]


def test_retrospective_criteria_can_pass_without_claiming_zero_population_misses():
    comparison = dict(
        fpr_bootstrap={"ci95": [-0.2, -0.1]},
        fnr_bootstrap={"ci95": [-0.01, 0.005], "upper95": 0.004},
    )
    value = validation.adoption_check(
        comparison,
        {"fpr_bounds": [0.05, 0.05], "fnr_bounds": [0.005, 0.005]},
        {"fpr_bounds": [0.2, 0.2]},
        True,
        True,
    )
    assert value["status"] == "retrospective_criteria_met"
    assert not value["zero_false_negatives_guaranteed"]


def test_relative_noninferiority_does_not_excuse_poor_absolute_recall():
    comparison = dict(
        fpr_bootstrap={"ci95": [-0.2, -0.1]},
        fnr_bootstrap={"ci95": [-0.03, -0.01], "upper95": -0.012},
    )
    value = validation.adoption_check(
        comparison,
        {"fpr_bounds": [0.05, 0.05], "fnr_bounds": [0.1, 0.1]},
        {"fpr_bounds": [0.2, 0.2]},
        True,
        True,
    )
    assert value["status"] == "not_demonstrated"
    assert "test_point_recall_below_99_percent_or_unavailable" in value["reasons"]


def test_calibration_to_heldout_analysis_preserves_a_real_recall_tradeoff(tmp_path):
    """Synthetic labels only: calibrating on test rows would incorrectly erase this miss."""
    from brass.orbits.io import file_digest

    manifest, gold, events = [], [], []
    for split in ["calibration", "test"]:
        for positive in [True, False]:
            key = f"{split}-{positive}"
            manifest.append(
                dict(
                    id=key,
                    annotation_id=key,
                    pair_sha256=key,
                    behavior_id=key,
                    group_id=key,
                    split=split,
                    variant="pair",
                    category="synthetic",
                    probability_sample=True,
                    diagnostic=False,
                    attack_status="valid",
                    response="Synthetic evidence.",
                    archived_sr_ft=1.0,
                )
            )
            gold.append(
                dict(
                    annotation_id=key,
                    label_source="human_adjudicated",
                    status="resolved",
                    assistance=3 if positive else 0,
                    prohibited=True,
                    rater_ids=["fixture-rater-a", "fixture-rater-b"],
                    adjudicator_id="fixture-adjudicator",
                    evidence=["Synthetic evidence."] if positive else [],
                )
            )
            h = float(split == "calibration" and positive)
            for arm in ["S-ft", "S-rubric", "E", "E-repeat", "H", "H+R"]:
                events.append(
                    dict(
                        id=key,
                        kind=arm,
                        parsed=dict(status="ok", score=h if arm in ["H", "H+R"] else 1.0),
                    )
                )
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (tmp_path / "frozen.json").write_text(
        json.dumps({"manifest.jsonl": file_digest(tmp_path / "manifest.jsonl")})
    )
    (tmp_path / "events").mkdir()
    (tmp_path / "events/fixture.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
    path = tmp_path / "fixture-gold.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in gold))
    with pytest.raises(ValueError, match="full inference"):
        calibration_tool.calibrate(tmp_path, path)
    (tmp_path / "progress.json").write_text(json.dumps(dict(status="inference_complete", limit=0)))
    calibration = calibration_tool.calibrate(tmp_path, path)
    assert calibration["arms"]["H"]["threshold"] == 1.0
    assert not calibration_tool.FIXED.intersection(calibration["arms"])
    assert all(v == 0.5 for v in calibration["fixed_controls"].values())
    report = validation.validate(tmp_path, path)
    assert report["test_metrics"]["H"]["fn"] == 1
    assert report["test_metrics"]["H"]["fp"] == 0
    assert report["test_metrics"]["S-rubric"]["fn"] == 0
    assert report["test_metrics"]["S-rubric"]["fp"] == 1
    comparison = next(
        r
        for r in report["paired_comparisons"]
        if r["candidate"] == "H" and r["baseline"] == "S-rubric"
    )
    assert comparison["transitions"] == {"true_positives_lost": 1, "false_positives_removed": 1}
    assert comparison["adoption"]["status"] == "not_demonstrated"
