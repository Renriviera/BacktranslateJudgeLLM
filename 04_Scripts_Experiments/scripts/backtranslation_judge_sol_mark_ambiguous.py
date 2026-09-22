#!/usr/bin/env python3
"""Preserve a inspected multiple-reply CLI session as unavailable without re-inference."""

import argparse
import fcntl
import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.orbits.io import file_digest, read_jsonl  # noqa: E402
from brass.sol_reference.codex_runner import DEFAULT_OUT  # noqa: E402
from brass.sol_reference.runner import completed, now, parse_reference  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pair", required=True)
    a = p.parse_args()
    root = a.run_dir.resolve()
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        units = {r["pair_sha256"]: r for r in read_jsonl(root / "requests.jsonl")}
        done = completed(root, units)
        if a.pair in done:
            raise ValueError("Already has a terminal event; do not overwrite")
        progress = json.loads((root / "progress.json").read_text())
        if a.pair not in {e["pair_sha256"] for e in progress["errors"]}:
            raise ValueError("Not a stopped runtime error")
        path = root / "traces" / f"{a.pair}.jsonl"
        trace = read_jsonl(path)
        items = [e["item"] for e in trace if e["type"] == "item.completed"]
        answers = [i["text"] for i in items if i["type"] == "agent_message"]
        terminals = [e for e in trace if e["type"] == "turn.completed"]
        assert len(answers) > 1 and len(terminals) == 1
        assert not [e for e in trace if e["type"] == "turn.failed"]
        assert not [i for i in items if i["type"] not in {"agent_message", "reasoning", "error"}]
        u = units[a.pair]
        usage = terminals[0].get("usage", {})
        raw = "\n\n".join(answers)
        event = dict(
            pair_sha256=a.pair,
            row_ids=u["row_ids"],
            request_sha256=u["request_sha256"],
            model_requested="gpt-5.6-sol",
            model_returned=None,
            reasoning_effort="medium",
            runtime="codex_exec_chatgpt_subscription",
            created_at_utc=now(),
            attempts=1,
            raw=raw,
            finish_reason="multiple_messages",
            judge_refusal=None,
            parsed=parse_reference(raw, u["response"], "multiple_messages"),
            cli_usage=usage,
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "prompt_tokens_details": {"cached_tokens": usage.get("cached_input_tokens", 0)},
                "completion_tokens_details": {
                    "reasoning_tokens": usage.get("reasoning_output_tokens", 0)
                },
            },
            trace_sha256=file_digest(path),
            resolution="Multiple rubric replies after internal CLI retry: preserved all text, excluded from scoring, never retried.",
            resolver_script_sha256=file_digest(Path(__file__)),
        )
        with (root / "events.jsonl").open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps({"pair": a.pair, "status": event["parsed"]["status"], "score": None}))


if __name__ == "__main__":
    main()
