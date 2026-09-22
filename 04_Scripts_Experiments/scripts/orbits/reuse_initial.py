#!/usr/bin/env python
"""Reuse identical initial generations across inverse-template calibration arms."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.orbits.io import append_jsonl, digest, read_jsonl, stable_seed, write_json  # noqa: E402
from brass.orbits.runner import OrbitConfig, Request  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    assert (args.source / "completion.json").exists(), "Source must have completed generation"
    assert not (args.output / "events.jsonl").exists(), "Destination already has events"
    spec = json.loads((args.source / "run_spec.json").read_text())
    config = OrbitConfig(**json.loads(args.config.read_text()))
    old = dict(spec["config"])
    new = asdict(config)
    old.pop("inverse_template")
    new.pop("inverse_template")
    assert old == new, "Only the inverse template may differ when initial generations are reused"
    manifest = {r["id"]: r for r in json.loads(args.manifest.read_text())}
    events = [e for e in read_jsonl(args.source / "events.jsonl") if e["stage"] == 0]
    assert len(events) == len(manifest) * (config.sampled_trajectories + int(config.greedy))
    for e in events:
        item = manifest[e["item_id"]]
        request = Request(
            e["id"],
            [{"role": "user", "content": item["prompt"]}],
            stable_seed(config.seed, item["id"], e["trajectory"], 0),
            config.forward_temperature if e["trajectory"] != "greedy" else 0.0,
            config.forward_max_tokens,
            config.top_p,
        )
        assert digest(asdict(request)) == e["request_sha256"], "Initial request mismatch"
    append_jsonl(args.output / "events.jsonl", events)
    write_json(
        args.output / "reused_initial_generations.json",
        {
            "source_directory": str(args.source.resolve()),
            "n_reused": len(events),
            "source_spec_sha256": digest(spec),
            "additional_generated_tokens": 0,
            "note": "Same checkpoint, prompts, seeds and forward settings; original ledger reservations retained.",
        },
    )
    print(f"Reused {len(events)} identical initial generations.")


if __name__ == "__main__":
    main()
