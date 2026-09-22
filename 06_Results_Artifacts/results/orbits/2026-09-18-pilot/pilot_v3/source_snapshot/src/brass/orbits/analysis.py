"""Descriptive pilot validity checks. No judge score is treated as human ground truth."""

from __future__ import annotations

from collections import Counter, defaultdict


def validity_report(manifest: list[dict], events: list[dict]) -> dict:
    lookup = {r["id"]: r for r in manifest}
    groups = defaultdict(list)
    for e in events:
        groups[lookup[e["item_id"]]["cohort"]].append(e)
    report = {}
    for cohort, rows in groups.items():
        inverse = [r for r in rows if r["direction"] == "inverse"]
        actual = [r for r in rows if r["finish_reason"] != "context_overflow"]
        truncated = sum(r["status"] == "generation_truncated" for r in actual)
        valid = sum(r["status"] == "ok" for r in inverse)
        refused = sum(r["status"] == "inverse_refusal_heuristic" for r in inverse)
        # Missing inverse outcomes do not receive artificial successes.
        parse_rate = valid / len(inverse) if inverse else None
        truncation_rate = truncated / len(actual) if actual else None
        refusal_rate = refused / len(inverse) if inverse else None
        report[cohort] = {
            "n_items": len({r["item_id"] for r in rows}),
            "n_events": len(rows),
            "n_inverse": len(inverse),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "inverse_usable_fraction": parse_rate,
            "truncation_fraction": truncation_rate,
            "inverse_refusal_heuristic_fraction": refusal_rate,
            "numeric_gates_pass": bool(
                parse_rate is not None
                and parse_rate >= 0.98
                and truncation_rate is not None
                and truncation_rate < 0.05
                and refusal_rate is not None
                and refusal_rate <= 0.10
                and not any(r["status"] == "context_overflow" for r in rows)
            ),
        }
    return {
        "cohorts": report,
        "all_numeric_gates_pass": len(report) == 3
        and all(r["numeric_gates_pass"] for r in report.values()),
        "manual_reconstruction_review": "pending",
        "human_success_labels": "pending",
        "total_generated_tokens": sum(r["output_tokens"] for r in events),
        "total_input_tokens": sum(r["input_tokens"] for r in events),
    }


def basic_trajectories(manifest, events):
    lookup = {r["id"]: r for r in manifest}
    groups = defaultdict(list)
    for event in events:
        groups[(event["item_id"], event["trajectory"])].append(event)
    rows = []
    for (key, trajectory), path in groups.items():
        path.sort(key=lambda e: e["stage"])
        forward = [e for e in path if e["direction"] == "forward"]
        seen, first_exact_recurrence = {}, None
        for e in forward:
            if e["text"] in seen and first_exact_recurrence is None and e["status"] == "ok":
                first_exact_recurrence = {"step": e["step"], "lag": e["step"] - seen[e["text"]]}
            seen[e["text"]] = e["step"]
        rows.append(
            {
                "item_id": key,
                "group_id": lookup[key]["group_id"],
                "cohort": lookup[key]["cohort"],
                "trajectory": trajectory,
                "observed_forward_states": len(forward),
                "last_stage": path[-1]["stage"],
                "terminal_status": path[-1]["status"],
                "first_exact_response_recurrence": first_exact_recurrence,
                "forward_token_lengths": [e["output_tokens"] for e in forward],
            }
        )
    return rows
