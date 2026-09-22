#!/usr/bin/env python
"""Freeze the core and prespecified diagnostic panels before viewing main outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.io import digest, file_digest, stable_seed, write_json  # noqa: E402
from brass.orbits.runner import OrbitConfig  # noqa: E402


def balanced(rows, n, label):
    bins = defaultdict(list)
    for row in rows:
        bins[row.get("control_type", row.get("family", row.get("stratum")))].append(row)
    for key in bins:
        bins[key].sort(key=lambda r: stable_seed(235711, label, r["id"]))
    out = []
    while len(out) < n:
        before = len(out)
        for key in sorted(bins):
            if bins[key] and len(out) < n:
                out.append(bins[key].pop(0))
        if len(out) == before:
            raise ValueError("Panel cannot meet frozen quota")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--code-image", required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if (root / "execution_queue.json").exists() or (root / "main_core/events.jsonl").exists():
        raise ValueError("Refuse to change a frozen or started main study")
    calibration = json.loads((root / "calibrated_pilot.json").read_text())
    gate = json.loads((root / calibration["directory"] / "validity_report.json").read_text())
    if not gate["run_complete"] or not gate["all_numeric_gates_pass"]:
        raise ValueError("Calibrated pilot must pass first")
    rows = json.loads((root / "main_manifest.json").read_text())
    config = OrbitConfig(**json.loads((root / "main_config.json").read_text()))
    panel = sum(
        [
            balanced([r for r in rows if r["cohort"] == c], 32, "diagnostic96")
            for c in ["attack", "benign", "control"]
        ],
        [],
    )
    deep = sum(
        [
            balanced([r for r in panel if r["cohort"] == c], 16, "deep48")
            for c in ["attack", "benign", "control"]
        ],
        [],
    )
    frozen = root / "frozen"
    jobs = []

    def add(name, manifest, cfg, reuse=None, max_stage=0, seed_archives=False):
        mp, cp = frozen / f"{name}_manifest.json", frozen / f"{name}_config.json"
        write_json(mp, manifest)
        write_json(cp, asdict(cfg))
        jobs.append(
            {
                "name": name,
                "manifest": str(mp),
                "config": str(cp),
                "output": str(root / name),
                "reuse": reuse,
                "reuse_max_stage": max_stage,
                "seed_archives": seed_archives,
            }
        )

    add("main_core", rows, config)
    add(
        "fixed_resampling",
        rows,
        replace(config, round_trips=0, sampled_trajectories=16, greedy=False),
        "main_core",
    )
    add(
        "sampled_forward_greedy_inverse",
        panel,
        replace(config, sampled_trajectories=4, greedy=False, inverse_temperature=0.0),
        "main_core",
    )
    add(
        "greedy_forward_sampled_inverse",
        panel,
        replace(config, sampled_trajectories=4, greedy=False, forward_temperature=0.0),
    )
    add(
        "alternate_inverse",
        panel,
        replace(
            config, sampled_trajectories=4, greedy=False, inverse_template="explicit_json_minimal"
        ),
        "main_core",
    )
    add(
        "deep_extension",
        deep,
        replace(config, round_trips=20, sampled_trajectories=4, greedy=False),
        "main_core",
        12,
    )
    add(
        "equal_call_resampling",
        panel,
        replace(config, round_trips=0, sampled_trajectories=52, greedy=False),
        "fixed_resampling",
    )
    archives, loaded = [], {}
    for row in rows:
        if row["cohort"] != "attack":
            continue
        path = REPO / row["details_path"]
        if path not in loaded:
            loaded[path] = json.loads(path.read_text())
        responses = loaded[path]["completions"]["attacked_instruct"][row["group_id"]]
        if digest(responses) != row["archived_completions_sha256"]:
            raise ValueError("Archived source responses changed")
        for index in row["archived_seed_indices"]:
            item = dict(row)
            item.update(
                id=f"{row['id']}:archive:{index}",
                parent_id=row["id"],
                starting_response=responses[index],
                starting_response_sha256=digest(responses[index]),
                archived_index=index,
            )
            archives.append(item)
    add(
        "archived_success_starts",
        archives,
        replace(config, sampled_trajectories=1, greedy=False),
        seed_archives=True,
    )
    add(
        calibration["directory"],
        json.loads((root / "pilot_manifest.json").read_text()),
        OrbitConfig(**json.loads((root / calibration["config_file"]).read_text())),
    )
    jobs[-1]["skip_generation"] = True
    freeze_paths = [Path(j[k]) for j in jobs for k in ["manifest", "config"]]
    source_paths = [
        *(REPO / "src/brass/orbits").glob("*.py"),
        *(REPO / "scripts/orbits").glob("*.py"),
    ]
    write_json(
        root / "execution_queue.json",
        {
            "schema_version": 1,
            "run_dir": str(root),
            "code_image": args.code_image,
            "jobs": jobs,
            "frozen_file_hashes": {str(p): file_digest(p) for p in freeze_paths},
            "source_hashes": {str(p): file_digest(p) for p in source_paths},
            "generated_token_limit": 80000000,
            "judge_call_limit": 200000,
            "diagnostic_panel_sha256": digest([r["id"] for r in panel]),
            "pending_optional_diagnostics": [
                "reference responses",
                "task-restating and context-restored processes",
                "local perturbations",
                "multiple inversions of one fixed response",
            ],
            "notes": [
                "Diagnostic panels are frozen without main outcomes.",
                "52 fixed-prompt samples match four six-round orbits in generation-call count, not FLOPs.",
                "Core and diagnostics stop at the shared hard budget; higher calibrated response caps make worst-case completion unaffordable.",
                "Optional diagnostics require an allocation record before their results are examined; no unbounded expansion is authorized.",
            ],
        },
    )
    print(
        f"Frozen {len(jobs)} computational arms (pilot scoring only); {len(panel)} diagnostic and {len(deep)} deep-extension items."
    )


if __name__ == "__main__":
    main()
