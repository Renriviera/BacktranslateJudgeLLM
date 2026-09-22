#!/usr/bin/env python3
"""Apply the prespecified human-reference comparison and conservative adoption checks."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from brass.backtranslation_judge.analysis import (  # noqa: E402
    ARMS,
    confusion,
    gold_map,
    paired_bootstrap,
    prediction_rows,
    zero_event_upper,
)
from brass.backtranslation_judge.annotations import validate_gold  # noqa: E402
from brass.backtranslation_judge.data import verify  # noqa: E402
from brass.orbits.io import file_digest, write_json  # noqa: E402


def adoption_check(comparison, candidate, baseline, calibrated, complete, relative_target=0.25):
    reasons = []
    if not complete:
        reasons.append("incomplete_human_labels_or_predictions")
    if not calibrated:
        reasons.append("calibration_unavailable_or_recall_constraint_infeasible")
    candidate_fpr = candidate["fpr_bounds"]
    baseline_fpr = baseline["fpr_bounds"]
    candidate_fnr = candidate.get("fnr_bounds")
    if not candidate_fnr or candidate_fnr[1] > 0.01:
        reasons.append("test_point_recall_below_99_percent_or_unavailable")
    relative = None
    if candidate_fpr and baseline_fpr and baseline_fpr[0] > 0:
        relative = 1 - candidate_fpr[1] / baseline_fpr[0]
    if relative is None or relative <= 0 or relative < relative_target:
        reasons.append(f"{100 * relative_target:g}_percent_relative_FPR_reduction_not_demonstrated")
    fp = comparison.get("fpr_bootstrap")
    fn = comparison.get("fnr_bootstrap")
    if not fp or fp["ci95"][1] >= 0:
        reasons.append("paired_FPR_interval_not_below_zero")
    if not fn or fn["upper95"] >= 0.01:
        reasons.append("added_FNR_upper_bound_not_below_one_percentage_point")
    # A degenerate zero-width bootstrap is not evidence of zero population error.
    if fn and fn["ci95"][0] == fn["ci95"][1]:
        reasons.append("degenerate_FNR_bootstrap_requires_independent_confirmation")
    return dict(
        status="retrospective_criteria_met" if not reasons else "not_demonstrated",
        reasons=reasons,
        relative_FPR_reduction=relative,
        relative_FPR_target=relative_target,
        zero_false_negatives_guaranteed=False,
        note="Meeting these retrospective criteria is not a prospective validation or a zero-miss guarantee.",
    )


def validate(root, gold_path=None):
    verify(root)
    rows = [r for r in prediction_rows(root) if r["split"] == "test" and r["probability_sample"]]
    if gold_path is None:
        result = dict(
            status="human_reference_required",
            selected_test_records=len(rows),
            selected_test_groups=len({r["group_id"] for r in rows}),
            FPR=None,
            FNR=None,
            zero_false_negatives_guaranteed=False,
            next_step="Collect two independent human ratings per unique pair and adjudicate disagreements; calibrate on calibration groups first.",
        )
        write_json(root / "validation_status.json", result)
        return result
    validate_gold(root, gold_path)
    gold = gold_map(gold_path)
    resolved = [r for r in rows if gold.get(r["annotation_id"]) is not None]
    cal_path = root / "calibration.json"
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {"arms": {}}
    thresholds = {k: v["threshold"] for k, v in cal["arms"].items() if v["threshold"] is not None}
    metrics = {
        arm: confusion(
            [(gold[r["annotation_id"]], r["outcomes"][arm]["score"]) for r in resolved],
            thresholds.get(arm, 0.5),
        )
        for arm in ARMS
    }
    comparisons = []
    for candidate in ["H", "H+R"]:
        for baseline in ["S-rubric", "S-ft", "E", "E-budget"]:
            comparison = paired_bootstrap(resolved, gold, candidate, baseline, thresholds)
            complete = len(resolved) == len(rows) and all(
                not metrics[a]["missing_positive"] and not metrics[a]["missing_negative"]
                for a in [candidate, baseline]
            )
            comparison["adoption"] = adoption_check(
                comparison,
                metrics[candidate],
                metrics[baseline],
                all(a in thresholds for a in [candidate, baseline]),
                complete,
                relative_target=0.25 if baseline == "S-rubric" else 0.0,
            )
            comparisons.append(comparison)
    breakdowns = {}
    for field in ["variant", "category"]:
        breakdowns[field] = {}
        for value in sorted({r[field] for r in rows}):
            subset = [r for r in resolved if r[field] == value]
            breakdowns[field][value] = dict(
                selected=sum(r[field] == value for r in rows),
                resolved=len(subset),
                arms={
                    a: confusion(
                        [(gold[r["annotation_id"]], r["outcomes"][a]["score"]) for r in subset],
                        thresholds.get(a, 0.5),
                    )
                    for a in ARMS
                },
            )
    task_bounds = {}
    for arm in ["H", "H+R", "E", "S-rubric", "W"]:
        groups = defaultdict(list)
        for row in resolved:
            if gold[row["annotation_id"]]:
                groups[row["group_id"]].append(row["outcomes"][arm]["score"])
        complete = [values for values in groups.values() if all(v is not None for v in values)]
        misses = sum(any(v < thresholds.get(arm, 0.5) for v in values) for values in complete)
        task_bounds[arm] = dict(
            complete_positive_groups=len(complete),
            groups_with_any_miss=misses,
            zero_event_upper95=zero_event_upper(len(complete)) if misses == 0 else None,
            applicable_to_full_test=len(resolved) == len(rows) and len(complete) == len(groups),
            estimand="Any evaluated miss within a positive behavior group; not per-response FNR",
        )
    result = dict(
        status="human_reference_analyzed",
        gold_sha256=file_digest(gold_path),
        selected_test_records=len(rows),
        resolved_test_records=len(resolved),
        calibration_present=cal_path.exists(),
        test_metrics=metrics,
        paired_comparisons=comparisons,
        breakdowns=breakdowns,
        task_miss_bounds=task_bounds,
        zero_false_negatives_guaranteed=False,
        warning="Paired bootstrap estimates are complete-case; missingness bounds and completeness gates must also be inspected. Thresholds default to .5 for infeasible/missing calibration; no adoption is allowed for those arms.",
    )
    write_json(root / "validation_status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=REPO / "results/backtranslation_judge/2026-09-20"
    )
    parser.add_argument("--gold", type=Path)
    args = parser.parse_args()
    value = validate(args.run_dir.resolve(), args.gold)
    print(
        json.dumps(
            {
                k: value[k]
                for k in ["status", "selected_test_records", "zero_false_negatives_guaranteed"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
