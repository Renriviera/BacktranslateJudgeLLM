#!/usr/bin/env python
"""Import observed successful responses as separate, explicitly historical starts."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.io import append_jsonl, digest, stable_seed, write_json  # noqa: E402
from brass.orbits.runner import OrbitConfig, Request  # noqa: E402


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["run-dir", "manifest", "config", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "archive_start_provenance.json").exists():
        return
    if (args.output / "events.jsonl").exists():
        raise ValueError("Destination already has events without archive provenance")
    cfg = OrbitConfig(**json.loads(args.config.read_text()))
    if cfg.sampled_trajectories != 1 or cfg.greedy:
        raise ValueError("Each archived response gets one explicitly conditioned trajectory")
    runtime = json.loads((args.run_dir / "preflight.json").read_text())
    tok = AutoTokenizer.from_pretrained(
        runtime["models"]["target"]["snapshot"], local_files_only=True
    )
    events = []
    for row in json.loads(args.manifest.read_text()):
        if digest(row["starting_response"]) != row["starting_response_sha256"]:
            raise ValueError("Archive seed hash mismatch")
        key = row["id"] + "|sample:0|0"
        seed = stable_seed(cfg.seed, row["id"], "sample:0", 0)
        request = Request(
            key,
            [{"role": "user", "content": row["prompt"]}],
            seed,
            cfg.forward_temperature,
            cfg.forward_max_tokens,
            cfg.top_p,
        )
        events.append(
            {
                "id": key,
                "item_id": row["id"],
                "group_id": row["group_id"],
                "trajectory": "sample:0",
                "stage": 0,
                "step": 0,
                "direction": "forward",
                "cohort": row["cohort"],
                "parent_id": None,
                "request_sha256": digest(asdict(request)),
                "seed": seed,
                "input_messages": request.messages,
                "text": row["starting_response"],
                "output_tokens": len(
                    tok.encode(row["starting_response"], add_special_tokens=False)
                ),
                "input_tokens": 0,
                "finish_reason": "historical_unrecorded",
                "stop_reason": None,
                "status": "ok",
                "reconstructed_prompt": None,
                "reservation_id": "imported_archive_no_new_generation",
                "batch_seconds": 0,
                "initial_source": "historical_successful_response",
                "historical_revision_verified": False,
                "historical_truncation_status": "unknown",
                "additional_generated_tokens": 0,
            }
        )
    append_jsonl(args.output / "events.jsonl", events)
    write_json(
        args.output / "archive_start_provenance.json",
        {
            "n_imported": len(events),
            "additional_generated_tokens": 0,
            "interpretation": "Success-conditioned historical starts; not fresh target replay. Historical stop metadata and exact revision are unknown.",
        },
    )
    print(f"Imported {len(events)} archived starts, no target generation.")


if __name__ == "__main__":
    main()
