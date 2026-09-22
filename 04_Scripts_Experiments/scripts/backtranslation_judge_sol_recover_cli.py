#!/usr/bin/env python3
"""Recover successful CLI judgments after the transport-event parser amendment."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from brass.paths import recorded_path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.orbits.io import file_digest, read_jsonl, write_json  # noqa: E402
from brass.sol_reference.codex_runner import (  # noqa: E402
    DEFAULT_OUT,
    parse_cli,
    policy_blocked,
    prepare,
)
from brass.sol_reference.runner import DEFAULT_PARENT, completed, now, parse_reference  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, default=DEFAULT_PARENT / "sol_reference_codex_medium")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    old, out = a.previous.resolve(), a.out.resolve()
    if (out / "reuse_provenance.json").exists():
        raise ValueError("Recovery already performed")
    old_spec = json.loads((old / "run_spec.json").read_text())
    spec = prepare(out, recorded_path(old_spec["api_run"]))
    for field in [
        "model",
        "reasoning_effort",
        "runtime",
        "request_set_sha256",
        "parent_manifest_sha256",
        "cli_version",
        "cli_entrypoint_sha256",
    ]:
        if old_spec[field] != spec[field]:
            raise ValueError(f"Cannot reuse changed {field}")
    old_config, new_config = dict(old_spec["configuration"]), dict(spec["configuration"])
    old_config.pop("model_instructions_file")
    new_config.pop("model_instructions_file")
    if (
        old_config != new_config
        or (old / "rubric_system.txt").read_bytes() != (out / "rubric_system.txt").read_bytes()
    ):
        raise ValueError("Runtime or rubric changed")
    if (old / "requests.jsonl").read_bytes() != (out / "requests.jsonl").read_bytes():
        raise ValueError("Request identity changed")
    units = {r["pair_sha256"]: r for r in read_jsonl(out / "requests.jsonl")}
    prior = completed(old, units)
    initial = completed(out, units)
    if any(e["parsed"]["status"] != "api_policy_blocked" for e in initial.values()):
        raise ValueError("Destination already contains inference")
    events = dict(prior)
    recovered = []
    for pair, unit in units.items():
        source = old / "traces" / f"{pair}.jsonl"
        if pair in events:
            e = events[pair]
            if (
                e["parsed"]["status"] != "api_policy_blocked"
                and parse_reference(e["raw"], unit["response"]) != e["parsed"]
            ):
                raise ValueError("Stored rubric parse changed")
        elif source.exists():
            trace = source.read_text()
            if policy_blocked(trace) and any(
                json.loads(line).get("type") == "turn.failed" for line in trace.splitlines()
            ):
                result = dict(
                    raw="", parsed={"status": "api_policy_blocked", "score": None}, usage=None
                )
            else:
                result = parse_cli(trace, unit["response"], 0)
            # Old runner did not save exit codes. A single successful terminal
            # turn plus one assistant message proves a completed rubric output.
            events[pair] = dict(
                pair_sha256=pair,
                row_ids=unit["row_ids"],
                request_sha256=unit["request_sha256"],
                model_requested=spec["model"],
                model_returned=None,
                reasoning_effort="medium",
                runtime=spec["runtime"],
                created_at_utc=now(),
                latency_seconds=None,
                attempts=1,
                finish_reason="stop",
                judge_refusal=None,
                trace_sha256=file_digest(source),
                recovery_note="Recovered a terminal assessment or explicit policy rejection from its saved trace; original CLI exit code was not recorded. No re-inference.",
                **result,
            )
            recovered.append(pair)
        if source.exists():
            shutil.copyfile(source, out / "traces" / source.name)
            stderr = source.with_suffix(".stderr.txt")
            if stderr.exists():
                shutil.copyfile(stderr, out / "traces" / stderr.name)
    (out / "events.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events.values())
    )
    write_json(
        out / "reuse_provenance.json",
        dict(
            previous_run=str(old),
            previous_run_spec_sha256=file_digest(old / "run_spec.json"),
            previous_events_sha256=file_digest(old / "events.jsonl"),
            previous_progress_sha256=file_digest(old / "progress.json"),
            retained_events=len(prior),
            recovered_successful_pairs=recovered,
            amendment="Permit intermediate CLI reconnect events only when a single completed turn and one rubric response are recorded; preserve explicit policy failures as unscored and never retry them. All prompts, model and reasoning settings unchanged. No re-inference.",
            recovery_script_sha256=file_digest(Path(__file__)),
            created_at_utc=now(),
        ),
    )
    print(json.dumps(dict(retained=len(prior), recovered=len(recovered), total=len(events))))


if __name__ == "__main__":
    main()
