#!/usr/bin/env python
"""Summarize validity and export blinded review packets from real orbit events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.analysis import basic_trajectories, validity_report  # noqa: E402
from brass.orbits.io import digest, read_jsonl, stable_seed, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    events = read_jsonl(args.events_dir / "events.jsonl")
    report = validity_report(manifest, events)
    report["run_complete"] = (args.events_dir / "completion.json").exists()
    write_json(args.events_dir / "validity_report.json", report)
    write_json(args.events_dir / "trajectory_summary.json", basic_trajectories(manifest, events))
    review, key = [], {}
    lookup = {r["id"]: r for r in manifest}
    for cohort in ["attack", "benign", "control"]:
        candidates = [e for e in events if e["cohort"] == cohort and e["direction"] == "inverse"]
        candidates.sort(key=lambda e: stable_seed(235711, e["id"], "inverse_review"))
        for e in candidates[:20]:
            rid = digest(["blinded-review-v1", e["id"]])[:16]
            original = lookup[e["item_id"]]
            supplied = json.loads(e["input_messages"][0]["content"].split("\n\n", 1)[1])[
                "assistant_response"
            ]
            review.append(
                {
                    "review_id": rid,
                    "response_to_reconstruct": supplied,
                    "reconstruction_raw": e["text"],
                    "original_task_for_retention_judgment": original["prompt"],
                    "faithful_to_response": None,
                    "preserves_original_task": None,
                    "inverse_refusal": None,
                    "notes": None,
                }
            )
            key[rid] = {"event_id": e["id"], "cohort": cohort, "step": e["step"]}
    review.sort(key=lambda r: r["review_id"])
    write_json(args.events_dir / "blinded_inverse_review.json", review)
    write_json(args.events_dir / "inverse_review_key.json", key)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
