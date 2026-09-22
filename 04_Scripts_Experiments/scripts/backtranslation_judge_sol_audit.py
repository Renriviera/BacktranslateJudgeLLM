#!/usr/bin/env python3
"""Audit completed Sol reference identities, raw traces, missingness and relative metrics."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.backtranslation_judge.prompts import messages  # noqa: E402
from brass.orbits.io import digest, file_digest, read_jsonl, write_json  # noqa: E402
from brass.sol_reference.analysis import load_rows  # noqa: E402
from brass.sol_reference.codex_runner import DEFAULT_OUT, parse_cli, policy_blocked  # noqa: E402
from brass.sol_reference.runner import now, parse_reference  # noqa: E402


def audit(root):
    spec, rows, units, events = load_rows(root)
    problems = []
    missing = []
    for source, expected in spec["source_sha256"].items():
        if file_digest(root / "source_snapshot" / source) != expected:
            problems.append(f"Source snapshot mismatch: {source}")
    for pair, unit in units.items():
        if (
            unit["messages"] != messages("S-rubric", unit)
            or digest(unit["messages"]) != unit["request_sha256"]
            or digest([unit["behavior"], unit["response"]]) != pair
        ):
            problems.append(f"Request mismatch: {pair}")
        event = events[pair]
        trace_path = root / "traces" / f"{pair}.jsonl"
        if trace_path.exists():
            trace = read_jsonl(trace_path)
            unexpected_tools = [
                e
                for e in trace
                if e["type"] in {"item.started", "item.completed"}
                and e.get("item", {}).get("type") not in {"agent_message", "reasoning", "error"}
            ]
            if unexpected_tools:
                problems.append(f"Unexpected tool in CLI trace: {pair}")
            if file_digest(trace_path) != event.get("trace_sha256"):
                problems.append(f"CLI trace hash mismatch: {pair}")
        if event["parsed"]["status"] == "api_policy_blocked":
            if trace_path.exists():
                if not policy_blocked(trace_path.read_text()):
                    problems.append(f"Policy event lacks an explicit denial trace: {pair}")
            elif pair not in spec["policy_blocked_pairs_not_retried"]:
                problems.append(f"Policy block lacks provenance: {pair}")
        elif event["parsed"]["status"] == "incomplete_multiple_messages":
            answers = [
                e["item"]["text"]
                for e in trace
                if e["type"] == "item.completed" and e["item"]["type"] == "agent_message"
            ]
            assert len(answers) > 1 and "\n\n".join(answers) == event["raw"]
            assert (
                parse_reference(event["raw"], unit["response"], "multiple_messages")
                == event["parsed"]
            )
        else:
            if file_digest(trace_path) != event["trace_sha256"]:
                problems.append(f"CLI trace hash mismatch: {pair}")
            reparsed = parse_cli(
                trace_path.read_text(), unit["response"], event.get("cli_return_code") or 0
            )
            if reparsed["raw"] != event["raw"] or reparsed["parsed"] != event["parsed"]:
                problems.append(f"CLI trace/assessment mismatch: {pair}")
    for row in rows:
        event = events[row["pair_sha256"]]
        if event["parsed"]["score"] is None:
            error_messages = []
            trace_path = root / "traces" / f"{row['pair_sha256']}.jsonl"
            if trace_path.exists():
                for e in read_jsonl(trace_path):
                    if e["type"] == "turn.failed":
                        error_messages.append(e.get("error", {}).get("message", ""))
            missing.append(
                dict(
                    id=row["id"],
                    pair_sha256=row["pair_sha256"],
                    variant=row["variant"],
                    category=row["category"],
                    status=event["parsed"]["status"],
                    sol_output=event["raw"],
                    terminal_errors=error_messages,
                    inherited_api_error=event.get("error"),
                )
            )
    analysis = json.loads((root / "analysis.json").read_text())
    for name, c in analysis["cohorts"].items():
        for arm, m in c["arms"].items():
            assert m["fp"] + m["tn"] + m["missing_negative"] == m["reference_negative"], (name, arm)
            assert m["fn"] + m["tp"] + m["missing_positive"] == m["reference_positive"], (name, arm)
            assert (
                m["reference_positive"] + m["reference_negative"] + m["reference_missing"]
                == m["selected"]
            ), (name, arm)
    for comparisons in analysis["paired_comparisons"].values():
        for c in comparisons:
            t = c["transitions"]
            assert sum(t.values()) == c["selected"]
            if c["paired_reference_negative"]:
                assert (
                    abs(
                        c["delta_fpr"]
                        - (t["new_false_positives"] - t["false_positives_removed"])
                        / c["paired_reference_negative"]
                    )
                    < 1e-12
                )
            if c["paired_reference_positive"]:
                assert (
                    abs(
                        c["delta_fnr"]
                        - (t["true_positives_lost"] - t["false_negatives_rescued"])
                        / c["paired_reference_positive"]
                    )
                    < 1e-12
                )
    assessment_rows = read_jsonl(root / "assessments.jsonl")
    assert {r["id"] for r in assessment_rows} == {r["id"] for r in rows}
    for r in assessment_rows:
        if r["sol_parsed"]["score"] is None:
            assert set(r["labels_relative_to_sol"].values()) == {"unresolved"}
    report = dict(
        status="passed" if not problems else "failed",
        checked_at_utc=now(),
        selected_records=len(rows),
        unique_pairs=len(units),
        valid_unique_references=sum(e["parsed"]["score"] is not None for e in events.values()),
        missing_selected_records=len(missing),
        missing_statuses=dict(Counter(m["status"] for m in missing)),
        problems=problems,
        checks=[
            "All exact rubric inputs and pair hashes",
            "Frozen source snapshot hashes",
            "Raw CLI trace hashes, terminal completion, no tools, one answer",
            "Stored score equals raw-response parse",
            "Unavailable labels never become negative",
            "All selected records exported exactly once",
            "Confusion denominators and paired-transition arithmetic",
        ],
        events_sha256=file_digest(root / "events.jsonl"),
        analysis_sha256=file_digest(root / "analysis.json"),
    )
    write_json(root / "integrity_audit.json", report)
    (root / "unavailable_references.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in missing)
    )
    if problems:
        raise ValueError(problems)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_OUT)
    audit(p.parse_args().run_dir.resolve())


if __name__ == "__main__":
    main()
