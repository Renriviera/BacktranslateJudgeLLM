#!/usr/bin/env python
"""Copy a compatible observed prefix for fixed-prompt resampling or deep extension."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.io import append_jsonl, digest, read_jsonl, stable_seed, write_json  # noqa: E402
from brass.orbits.runner import OrbitConfig, Request, inverse_messages  # noqa: E402


def copy_prefix(source, output, manifest_path, config_path, max_stage):
    if (output / "reused_prefix.json").exists():
        return
    if not (source / "completion.json").exists() or (output / "events.jsonl").exists():
        raise ValueError("Require a completed source and an empty destination")
    config = OrbitConfig(**json.loads(config_path.read_text()))
    source_spec = json.loads((source / "run_spec.json").read_text())
    for key in ["forward_max_tokens", "inverse_max_tokens", "max_model_len"]:
        if source_spec["config"][key] != asdict(config)[key]:
            raise ValueError(f"Incompatible context/generation cap: {key}")
    manifest = {r["id"]: r for r in json.loads(manifest_path.read_text())}
    names = {f"sample:{i}" for i in range(config.sampled_trajectories)}
    if config.greedy:
        names.add("greedy")
    kept = {}
    for e in sorted(read_jsonl(source / "events.jsonl"), key=lambda e: e["stage"]):
        if (
            e["item_id"] not in manifest
            or e["trajectory"] not in names
            or e["stage"] > min(max_stage, 2 * config.round_trips)
        ):
            continue
        if e["stage"] == 0:
            messages = [{"role": "user", "content": manifest[e["item_id"]]["prompt"]}]
        else:
            parent = kept[e["parent_id"]]
            messages = (
                inverse_messages(parent["text"], config.inverse_template)
                if e["direction"] == "inverse"
                else [{"role": "user", "content": parent["reconstructed_prompt"]}]
            )
        temp = (
            config.forward_temperature
            if e["direction"] == "forward"
            else config.inverse_temperature
        )
        request = Request(
            e["id"],
            messages,
            stable_seed(config.seed, e["item_id"], e["trajectory"], e["stage"]),
            0.0 if e["trajectory"] == "greedy" else temp,
            config.forward_max_tokens if e["direction"] == "forward" else config.inverse_max_tokens,
            config.top_p,
        )
        if digest(asdict(request)) != e["request_sha256"]:
            raise ValueError(f"Reused request mismatch: {e['id']}")
        kept[e["id"]] = e
    append_jsonl(output / "events.jsonl", list(kept.values()))
    write_json(
        output / "reused_prefix.json",
        {
            "source": str(source.resolve()),
            "source_spec_sha256": digest(source_spec),
            "n_events": len(kept),
            "max_stage": max_stage,
            "additional_tokens": 0,
            "manifest_sha256": digest(list(manifest.values())),
            "config_sha256": digest(asdict(config)),
        },
    )
    print(f"Reused {len(kept)} compatible observed events.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-stage", type=int, required=True)
    args = parser.parse_args()
    copy_prefix(args.source, args.output, args.manifest, args.config, args.max_stage)
