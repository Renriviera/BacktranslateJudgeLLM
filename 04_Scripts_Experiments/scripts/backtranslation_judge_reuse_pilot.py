#!/usr/bin/env python3
"""Reuse pilot inference only after checking exact inputs, parsing, models and decoding.

This permits a main-run scheduling change (e.g. larger batching), not a prompt/model change.
Every imported event retains its pilot source, file hash and original request fingerprint.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))

from brass.backtranslation_judge.data import verify  # noqa: E402
from brass.backtranslation_judge.prompts import messages, parse  # noqa: E402
from brass.backtranslation_judge.runner import all_events, initialize  # noqa: E402
from brass.orbits.io import append_jsonl, digest, file_digest, read_jsonl, write_json  # noqa: E402


def reuse(source, destination):
    verify(source)
    verify(destination)
    old = json.loads((source / "run_spec.json").read_text())
    new = initialize(REPO, destination)
    flexible = {"batch_size", "source_sha256"}
    if {k: v for k, v in old.items() if k not in flexible} != {
        k: v for k, v in new.items() if k not in flexible
    }:
        raise ValueError("Pilot and main model/decoding specifications differ")
    if all_events(destination):
        raise ValueError("Destination already has inference; import only once before main run")
    rows = {r["id"]: r for r in read_jsonl(destination / "manifest.jsonl")}
    events = all_events(source)
    verified = []
    worker_changed = (
        old["source_sha256"]["04_Scripts_Experiments/src/brass/backtranslation_judge/runner.py"]
        != new["source_sha256"]["04_Scripts_Experiments/src/brass/backtranslation_judge/runner.py"]
    )
    for event in events.values():
        row = rows[event["id"]]
        if row["pair_sha256"] != event["pair_sha256"]:
            raise ValueError("Response changed")
        kind = event["kind"]
        if kind == "H+R" and worker_changed:
            # A fresh target round trip can change auxiliary evidence even with the same seed.
            continue
        if kind in ("supported_inverse", "wang_inverse", "S-rubric", "E", "E-repeat", "H", "H+R"):
            extra = None
            if kind in ("H", "H+R"):
                inv = events.get((row["id"], "supported_inverse"))
                extra = {"reconstruction": inv.get("parsed") if inv else {"status": "missing"}}
                if kind == "H+R":
                    target = events.get((row["id"], "target_supported"))
                    extra["target_check"] = (
                        target.get("parsed") if target else {"status": "missing"}
                    )
            if digest(messages(kind, row, extra)) != event["request_sha256"]:
                raise ValueError(f"Prompt changed: {kind}")
            if (
                event.get("finish_reason") == "stop"
                and parse(kind, event["raw"], row["response"]) != event["parsed"]
            ):
                raise ValueError(f"Parsing changed: {kind}")
        else:
            # Non-generative replay and teacher-forced target scoring must retain exact worker implementation.
            worker = "04_Scripts_Experiments/src/brass/backtranslation_judge/runner.py"
            if old["source_sha256"][worker] != new["source_sha256"][worker]:
                continue
        verified.append(
            {
                **event,
                "reused_from": str(source),
                "pilot_spec_sha256": file_digest(source / "run_spec.json"),
            }
        )
    path = destination / "events/imported_pilot.jsonl"
    path.parent.mkdir(exist_ok=True)
    append_jsonl(path, verified)
    result = dict(
        source=str(source),
        destination=str(destination),
        verified_events=len(verified),
        skipped_non_generative_events=len(events) - len(verified),
        note="Exact generative request/model/decoding/parser parity checked. Target/S-ft events re-run if worker source changed.",
    )
    write_json(destination / "pilot_reuse.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reuse(args.source.resolve(), args.destination.resolve()), indent=2))


if __name__ == "__main__":
    main()
