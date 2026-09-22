#!/usr/bin/env python3
"""Calibrate scalar judges while retaining fixed Wang and AND/OR diagnostic rules.

This postprocessing entrypoint leaves the running inference source snapshot intact.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))

from brass.backtranslation_judge.analysis import (  # noqa: E402
    ARMS,
    choose_threshold,
    gold_map,
    prediction_rows,
)
from brass.backtranslation_judge.annotations import validate_gold  # noqa: E402
from brass.backtranslation_judge.data import verify  # noqa: E402
from brass.orbits.io import digest, file_digest, write_json  # noqa: E402

FIXED = {"W", "S-rubric AND W", "S-rubric OR W"}


def calibrate(root, gold_path):
    verify(root)
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if progress.get("status") != "inference_complete" or progress.get("limit") != 0:
        raise ValueError("Complete the full inference run before freezing calibration")
    validate_gold(root, gold_path)
    gold = gold_map(gold_path)
    rows = [
        r for r in prediction_rows(root) if r["split"] == "calibration" and r["probability_sample"]
    ]
    missing = {r["annotation_id"] for r in rows if r["annotation_id"] not in gold}
    if missing:
        raise ValueError(f"Calibration incomplete: {len(missing)} annotation ids are missing")
    resolved = [r for r in rows if gold[r["annotation_id"]] is not None]
    result = dict(
        gold_sha256=file_digest(gold_path),
        predictions_sha256=digest(rows),
        calibration_script_sha256=file_digest(Path(__file__)),
        unresolved=len(rows) - len(resolved),
        arms={
            arm: choose_threshold(
                [(gold[r["annotation_id"]], r["outcomes"][arm]["score"]) for r in resolved]
            )
            for arm in ARMS
            if arm not in FIXED
        },
        fixed_controls={arm: 0.5 for arm in sorted(FIXED)},
        fixed_control_definition="W uses its native gates. AND/OR combine the original S-rubric >=0.5 decision with W; neither combination is threshold-calibrated.",
        selective_thresholds=dict(
            negative=0.1, positive=0.9, status="fixed exploratory thresholds; not recall-certified"
        ),
    )
    dest = root / "calibration.json"
    if dest.exists() and json.loads(dest.read_text()) != result:
        raise ValueError("Calibration is immutable; use a separately versioned study")
    write_json(dest, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=REPO / "06_Results_Artifacts/results/backtranslation_judge/2026-09-20"
    )
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.run_dir.resolve(), args.gold), indent=2))


if __name__ == "__main__":
    main()
