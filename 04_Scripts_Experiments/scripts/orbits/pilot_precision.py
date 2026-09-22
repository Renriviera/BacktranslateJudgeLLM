#!/usr/bin/env python
"""Illustrative main-study precision from pilot task-level geometric variability."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.orbits.io import file_digest, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    calibration = json.loads((root / "calibrated_pilot.json").read_text())
    path = root / calibration["directory"] / "geometry.json"
    geometry = json.loads(path.read_text())
    manifest = json.loads((root / "main_manifest.json").read_text())
    grouped, paths = defaultdict(list), defaultdict(dict)
    for r in geometry:
        if r["trajectory"] != "greedy" and r["generation_status"] == "ok":
            paths[(r["item_id"], r["trajectory"])][r["step"]] = r
    for states in paths.values():
        if 1 in states and 4 in states:
            a, b = states[1], states[4]
            grouped[(a["cohort"], a["group_id"])].append(
                b["response_distance_from_initial"] - a["response_distance_from_initial"]
            )
    variability, centered, sizes = {}, {}, {}
    for cohort in ["attack", "benign"]:
        values = np.array([np.mean(v) for (c, _), v in sorted(grouped.items()) if c == cohort])
        centered[cohort] = values - values.mean()
        sizes[cohort] = len(
            {r.get("analysis_group_id", r["group_id"]) for r in manifest if r["cohort"] == cohort}
        )
        variability[cohort] = {
            "pilot_groups": len(values),
            "pilot_group_sd": float(values.std(ddof=1)),
            "main_groups": sizes[cohort],
        }
    rng = np.random.default_rng(235711)
    noise = rng.choice(centered["attack"], (5000, sizes["attack"]), replace=True).mean(
        axis=1
    ) - rng.choice(centered["benign"], (5000, sizes["benign"]), replace=True).mean(axis=1)
    critical = float(np.quantile(np.abs(noise), 1 - 0.05 / 3))
    write_json(
        root / "pilot_precision_assessment.json",
        {
            "metric": "sampled response cosine distance from initial: step4 minus step1, averaged within task",
            "pilot_geometry_sha256": file_digest(path),
            "variability": variability,
            "centered_null_difference_interval95": np.quantile(noise, [0.025, 0.975]).tolist(),
            "simulations": 5000,
            "illustrative_three_comparison_critical_abs_difference": critical,
            "illustrative_shift_power": {
                str(shift): float(np.mean(np.abs(noise + shift) > critical))
                for shift in [0.0, 0.05, 0.10]
            },
            "limitations": [
                "Illustrative geometric proxy only; cosine units are not AUROC or task-retention probability.",
                "Assumes pilot variability transfers to the main tasks and ignores its estimation uncertainty.",
                "Does not establish power for a 0.05 paired AUROC improvement. Human labels and actual group sizes govern that analysis.",
                "PAP has only 21 eligible held-out prompts; family-specific and low-FPR conclusions will have limited precision.",
            ],
        },
    )
    print(json.dumps(variability))


if __name__ == "__main__":
    main()
