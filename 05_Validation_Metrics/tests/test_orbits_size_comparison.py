"""Meaningful checks for paired inference, task grouping and fixed-response controls."""

import importlib.util
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from brass.orbits.io import append_jsonl, digest, read_jsonl, stable_seed
from brass.orbits.runner import Generation, OrbitConfig, Request, inverse_messages
from brass.orbits.size_comparison import (
    holm,
    lexical_recall,
    paired_groups,
    paired_summary,
    references,
)


def load_script(name):
    path = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir()) / "04_Scripts_Experiments/scripts/orbits" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(item, group, path, model, value):
    return {
        "item_id": item,
        "group_id": group,
        "trajectory": path,
        "model": model,
        "cohort": "attack",
        "subgroup": "pap",
        "mode": "sampled",
        "step": 2,
        "metric": value,
    }


def test_pair_before_averaging_and_do_not_count_related_prompts_as_independent():
    rows = [
        row("a", "g1", "sample:0", "7b", 0),
        row("a", "g1", "sample:0", "32b", 2),
        row("a", "g1", "sample:1", "7b", 100),  # missing partner must not change means
        row("b", "g1", "sample:0", "7b", 2),
        row("b", "g1", "sample:0", "32b", 4),
        row("c", "g2", "sample:0", "7b", 4),
        row("c", "g2", "sample:0", "32b", 3),
    ]
    groups = paired_groups(rows, "metric", cohort="attack")
    assert groups == {"g1": [1, 3], "g2": [4, 3]}
    summary = paired_summary(groups, "test", draws=1000)
    assert summary["n_groups"] == 2
    assert summary["difference_32b_minus_7b"] == pytest.approx(0.5)
    assert summary["ci95"][0] <= 0.5 <= summary["ci95"][1]


def test_missing_state_not_zero_or_stability():
    rows = [row("a", "g1", "sample:0", "7b", 0.1), row("a", "g1", "sample:0", "32b", None)]
    assert paired_groups(rows, "metric") == {}
    assert paired_summary({}, "missing")["difference_32b_minus_7b"] is None


def test_holm_monotone_and_missing_p_ignored():
    result = holm([{"p": 0.01}, {"p": 0.03}, {"p": 0.8}, {"p": None}])
    assert [r.get("holm_p") for r in result] == [0.03, 0.06, 0.8, None]


def test_exact_framing_and_core_reference_separation():
    parent = {"id": "p", "prompt": "Count 7 red marbles."}
    framed = {
        "id": "f",
        "control_type": "benign_framed",
        "parent_id": "p",
        "prompt": "Speak as a fictional librarian.\n\nCount 7 red marbles.",
    }
    content, frame = references(framed, {"p": parent})
    assert content == parent["prompt"]
    assert frame == "Speak as a fictional librarian."
    assert lexical_recall(frame, content, exclude=content) == 0
    assert lexical_recall(frame, frame + " " + content, exclude=content) == 1
    assert lexical_recall("7 marbles", "7") == 0.5
    assert references({"prompt": "wrapped request", "behavior": "underlying task"}, {}) == (
        "underlying task",
        None,
    )


def test_ambiguous_frame_fails_instead_of_inventing_spans():
    with pytest.raises(ValueError, match="exact"):
        references(
            {"control_type": "benign_framed", "parent_id": "p", "prompt": "No matching content"},
            {"p": {"prompt": "TASK"}},
        )


def test_fixed_response_uses_original_requests_and_resumes_without_calls(tmp_path):
    script = load_script("run_size_comparison.py")
    config = OrbitConfig(
        round_trips=2, sampled_trajectories=1, greedy=False, inverse_template="explicit_json"
    )
    messages = inverse_messages("Observed response only.", config.inverse_template)
    request = Request(
        "a|sample:0|1",
        messages,
        stable_seed(config.seed, "a", "sample:0", 1),
        config.inverse_temperature,
        config.inverse_max_tokens,
        config.top_p,
    )
    source = {
        "id": request.id,
        "item_id": "a",
        "group_id": "g",
        "trajectory": "sample:0",
        "stage": 1,
        "step": 1,
        "direction": "inverse",
        "cohort": "benign",
        "parent_id": "a|sample:0|0",
        "request_sha256": digest(asdict(request)),
        "seed": request.seed,
        "input_messages": messages,
        "text": '{"prompt":"old"}',
        "status": "ok",
        "reconstructed_prompt": "old",
    }
    append_jsonl(tmp_path / "baseline_7b/events.jsonl", [source])

    class Backend:
        fingerprint = {"test_only": True}
        calls = 0

        def generate(self, requests):
            self.calls += 1
            assert requests == [request]
            return [Generation('{"prompt":"new"}', 5, 10, "stop")]

    backend = Backend()
    script.fixed_inverse(tmp_path, config, backend)
    out = read_jsonl(tmp_path / "fixed_7b_response_inverse_32b/events.jsonl")
    assert out[0]["reconstructed_prompt"] == "new"
    assert out[0]["source_inverse_sha256"] == digest(source)
    script.fixed_inverse(tmp_path, config, backend)
    assert backend.calls == 1


def test_valid_inverse_survives_failed_forward_in_retention_metrics():
    script = load_script("analyze_size_comparison.py")
    item = {"id": "a", "prompt": "task", "behavior": "core"}
    vectors = {
        digest("task"): np.array([1.0, 0.0]),
        digest("core"): np.array([0.0, 1.0]),
        digest("reconstructed"): np.array([0.0, 1.0]),
    }
    got = script.features(item, {}, "reconstructed", None, "task", None, vectors)
    assert got["content_similarity"] == 1
    assert got["response_drift"] is None
    assert got["prompt_drift"] == 1


def test_complete_analysis_writes_report_with_censoring(tmp_path, monkeypatch):
    import json
    import sys

    from brass.orbits.io import write_json

    script = load_script("analyze_size_comparison.py")
    manifest = [
        {
            "id": "a",
            "group_id": "a",
            "cohort": "attack",
            "family": "pap",
            "prompt": "frame task",
            "behavior": "task",
        },
        {"id": "b", "group_id": "b", "cohort": "benign", "stratum": "gsm8k", "prompt": "task"},
        {
            "id": "f",
            "group_id": "b",
            "cohort": "control",
            "control_type": "benign_framed",
            "parent_id": "b",
            "prompt": "fictional frame task",
        },
    ]
    tests = [
        {"cohort": c, "metric": m}
        for c in ["attack", "benign"]
        for m in ["response_drift", "prompt_drift"]
    ] + [
        {"cohort": "attack", "metric": "content_similarity"},
        {"cohort": "control", "subgroup": "benign_framed", "metric": "frame_similarity"},
        {"cohort": "control", "subgroup": "benign_framed", "metric": "frame_lexical_recall"},
    ]
    write_json(
        tmp_path / "frozen/protocol.json", {"primary_tests": tests, "limits": ["fixture only"]}
    )
    write_json(tmp_path / "frozen/manifest.json", manifest)
    write_json(tmp_path / "preflight.json", {"models": {"embedding": {}}})
    texts = [
        "frame task",
        "task",
        "fictional frame task",
        "fictional frame",
        "response",
        "reconstruction",
    ]
    keys = [digest(t) for t in texts]
    vector_values = np.array([[1.0, 0.0] for t in texts])
    for model, folder in [("7b", "baseline_7b"), ("32b", "main_32b")]:
        records = []
        for item in manifest:
            for trajectory in [f"sample:{i}" for i in range(8)] + ["greedy"]:
                for stage in range(5):
                    inverse = stage % 2 == 1
                    records.append(
                        {
                            "id": f"{item['id']}|{trajectory}|{stage}",
                            "item_id": item["id"],
                            "trajectory": trajectory,
                            "stage": stage,
                            "step": (stage + 1) // 2,
                            "direction": "inverse" if inverse else "forward",
                            "status": "generation_truncated"
                            if model == "32b" and trajectory == "sample:0" and stage == 4
                            else "ok",
                            "text": "reconstruction" if inverse else "response",
                            "reconstructed_prompt": "reconstruction" if inverse else None,
                            "output_tokens": 2,
                        }
                    )
        append_jsonl(tmp_path / folder / "events.jsonl", records)
        np.savez_compressed(
            tmp_path / folder / "text_embeddings.npz", keys=np.array(keys), vectors=vector_values
        )
        if model == "7b":
            append_jsonl(
                tmp_path / "fixed_7b_response_inverse_32b/events.jsonl",
                [r for r in records if r["direction"] == "inverse"],
            )
    monkeypatch.setattr(script, "extend_vectors", lambda *args: {"fixture_only": True})
    monkeypatch.setattr(sys, "argv", ["analysis", "--run-dir", str(tmp_path), "--device", "cpu"])
    script.main()
    result = json.loads((tmp_path / "comparison_summary.json").read_text())
    assert len(result["primary_tests"]) == 7
    assert (tmp_path / "comparison_report.md").exists()
    assert (tmp_path / "response_drift.png").stat().st_size > 1000
    rows = json.loads((tmp_path / "paired_features.json").read_text())
    failed = next(
        r
        for r in rows
        if r["model"] == "32b"
        and r["item_id"] == "a"
        and r["trajectory"] == "sample:0"
        and r["step"] == 2
    )
    assert failed["response_drift"] is None
    assert failed["content_similarity"] == 1.0
