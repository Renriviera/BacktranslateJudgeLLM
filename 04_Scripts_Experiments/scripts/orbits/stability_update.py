#!/usr/bin/env python
"""Exploratory core-run stability comparisons; no detector or causality claim."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_ind

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.orbits.io import digest, file_digest, read_jsonl, stable_seed, write_json  # noqa: E402

METRICS = [
    "response_drift_6",
    "prompt_drift_6",
    "mean_response_step_distance",
    "late_response_step_distance",
    "response_drift_after_first_cycle",
    "dispersion_change_6_vs_0",
]


def pairwise_dispersion(vectors):
    """Mean cosine distance over unordered pairs of unit vectors."""
    v = np.asarray(vectors)
    n = len(v)
    if n < 2:
        return None
    return float(np.clip(1 - (np.dot(v.sum(axis=0), v.sum(axis=0)) - n) / (n * (n - 1)), 0, 2))


def interval(values, seed, draws=10000):
    values = np.array(values, dtype=float)
    if not len(values):
        return {"mean": None, "ci95": None, "n": 0}
    samples = (
        np.random.default_rng(seed).choice(values, (draws, len(values)), replace=True).mean(axis=1)
    )
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "n": len(values),
    }


def grouped(rows, metric):
    out = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            out[row["group_id"]].append(row[metric])
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


def contrast(a, b, metric):
    x, y = np.array(list(grouped(a, metric).values())), np.array(list(grouped(b, metric).values()))
    if len(x) < 2 or len(y) < 2:
        return {"metric": metric, "status": "insufficient_groups"}
    rng = np.random.default_rng(stable_seed(235711, metric, "contrast"))
    boot = rng.choice(x, (10000, len(x)), replace=True).mean(axis=1) - rng.choice(
        y, (10000, len(y)), replace=True
    ).mean(axis=1)
    return {
        "metric": metric,
        "attack_groups": len(x),
        "benign_groups": len(y),
        "attack_mean": float(x.mean()),
        "benign_mean": float(y.mean()),
        "attack_minus_benign": float(x.mean() - y.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "welch_p": float(ttest_ind(x, y, equal_var=False).pvalue),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    path = root / "main_core"
    manifest = json.loads((root / "frozen/main_core_manifest.json").read_text())
    events = read_jsonl(path / "events.jsonl")
    geometry = {r["id"]: r for r in json.loads((path / "geometry.json").read_text())}
    embeddings = np.load(path / "text_embeddings.npz")
    vectors = dict(zip(embeddings["keys"], embeddings["vectors"], strict=True))
    forward, trajectories = defaultdict(list), defaultdict(dict)
    for e in events:
        if e["direction"] == "forward":
            forward[e["item_id"]].append(e)
            trajectories[(e["item_id"], e["trajectory"])][e["step"]] = e
    rows, curves, dispersions = [], [], []
    refusal = re.compile(
        r"^(?:I'm sorry|I am sorry|I'm (?:really |very )?sorry|I (?:cannot|can't|am unable)|Sorry[,!])",
        re.I,
    )
    for item in manifest:
        item_id = item["id"]
        base = {
            "item_id": item_id,
            "group_id": item.get("analysis_group_id", item["group_id"]),
            "cohort": item["cohort"],
            "subgroup": item.get("control_type", item.get("family", item.get("stratum"))),
            "parent_id": item.get("parent_id"),
            "framing_kind": item.get("framing_kind"),
        }
        for mode in ["sampled", "greedy"]:
            source = [
                e for e in forward[item_id] if (e["trajectory"] == "greedy") == (mode == "greedy")
            ]
            starts = [e for e in source if e["step"] == 0]
            row = {
                **base,
                "mode": mode,
                "prompt_tokens": item.get("prompt_tokens"),
                "initial_output_tokens": float(np.mean([e["output_tokens"] for e in starts])),
                "initial_refusal_fraction": float(
                    np.mean([bool(refusal.match(e["text"].strip())) for e in starts])
                ),
                "initial_short_fraction": float(np.mean([e["output_tokens"] < 30 for e in starts])),
            }
            per_step, dispersion = {}, {}
            for step in range(7):
                es = [e for e in source if e["step"] == step and e["status"] == "ok"]
                values = {
                    k: float(
                        np.mean(
                            [geometry[e["id"]][k] for e in es if geometry[e["id"]][k] is not None]
                        )
                    )
                    for k in [
                        "response_distance_from_initial",
                        "prompt_distance_from_initial",
                        "response_step_distance",
                    ]
                    if es and (step or k != "response_step_distance")
                }
                per_step[step] = values
                curves.append(
                    {**base, "mode": mode, "step": step, "n_valid_trajectories": len(es), **values}
                )
                if len(es) >= 2:
                    v = np.array([vectors[digest(e["text"])] for e in es])
                    n = len(v)
                    dispersion[step] = pairwise_dispersion(v)
                    dispersions.append(
                        {**base, "step": step, "n_trajectories": n, "dispersion": dispersion[step]}
                    )
            row["response_drift_6"] = per_step[6].get("response_distance_from_initial")
            row["prompt_drift_6"] = per_step[6].get("prompt_distance_from_initial")
            row["mean_response_step_distance"] = (
                float(np.mean([per_step[t]["response_step_distance"] for t in range(1, 7)]))
                if all("response_step_distance" in per_step[t] for t in range(1, 7))
                else None
            )
            row["late_response_step_distance"] = (
                float(np.mean([per_step[t]["response_step_distance"] for t in range(4, 7)]))
                if all("response_step_distance" in per_step[t] for t in range(4, 7))
                else None
            )
            # Use exactly paired, observed steps within a trajectory for the drift increment.
            increments = []
            for e in starts:
                ts = trajectories[(item_id, e["trajectory"])]
                if all(t in ts and ts[t]["status"] == "ok" for t in [1, 6]):
                    increments.append(
                        geometry[ts[6]["id"]]["response_distance_from_initial"]
                        - geometry[ts[1]["id"]]["response_distance_from_initial"]
                    )
            row["response_drift_after_first_cycle"] = (
                float(np.mean(increments)) if increments else None
            )
            row["dispersion_change_6_vs_0"] = (
                dispersion[6] - dispersion[0] if 6 in dispersion and 0 in dispersion else None
            )
            row["observed_final_trajectories"] = sum(
                e["step"] == 6 and e["status"] == "ok" for e in source
            )
            row["initial_trajectories"] = len(starts)
            rows.append(row)
    sampled = [r for r in rows if r["mode"] == "sampled"]
    a, b = ([r for r in sampled if r["cohort"] == c] for c in ["attack", "benign"])
    comparisons = [contrast(a, b, metric) for metric in METRICS]
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["welch_p"])
    last = 0.0
    for rank, index in enumerate(order):
        last = max(last, min(1.0, (len(order) - rank) * comparisons[index]["welch_p"]))
        comparisons[index]["holm_adjusted_p_six_exploratory_metrics"] = last
    strata = []
    for mode in ["sampled", "greedy"]:
        for subgroup in sorted({r["subgroup"] for r in rows}):
            selected = [r for r in rows if r["mode"] == mode and r["subgroup"] == subgroup]
            strata.append(
                {
                    "mode": mode,
                    "subgroup": subgroup,
                    **{
                        m: interval(
                            list(grouped(selected, m).values()),
                            stable_seed(235711, mode, subgroup, m),
                        )
                        for m in METRICS
                    },
                }
            )

    # Match task-group means using initial prompt/output length only, never stability outcomes.
    def matching_rows(source):
        gs = defaultdict(list)
        for row in source:
            gs[row["group_id"]].append(row)
        return [
            {
                "group_id": g,
                **{
                    k: (
                        float(np.mean([r[k] for r in rs if r[k] is not None]))
                        if any(r[k] is not None for r in rs)
                        else None
                    )
                    for k in ["prompt_tokens", "initial_output_tokens", *METRICS]
                },
            }
            for g, rs in sorted(gs.items())
        ]

    ag, bg = matching_rows(a), matching_rows(b)
    xa = np.log1p([[r["prompt_tokens"], r["initial_output_tokens"]] for r in ag])
    xb = np.log1p([[r["prompt_tokens"], r["initial_output_tokens"]] for r in bg])
    delta = np.abs(xa[:, None, :] - xb[None, :, :])
    cost = np.where(delta.max(axis=2) <= np.log(2), np.linalg.norm(delta, axis=2), 1e6)
    rr, cc = linear_sum_assignment(
        np.concatenate([cost, np.full((len(ag), len(ag)), 10.0)], axis=1)
    )
    pairs = [(ag[i], bg[j]) for i, j in zip(rr, cc, strict=True) if j < len(bg) and cost[i, j] < 10]
    matched = {
        m: interval(
            [x[m] - y[m] for x, y in pairs if x[m] is not None and y[m] is not None],
            stable_seed(235711, m, "matched"),
        )
        for m in METRICS
    }
    # Within-task benign framing contrast: average the eight sampled paths first.
    lookup = {r["item_id"]: r for r in sampled}
    framed = [r for r in sampled if r["subgroup"] == "benign_framed"]
    framing = {
        m: interval(
            [
                r[m] - lookup[r["parent_id"]][m]
                for r in framed
                if r[m] is not None and lookup[r["parent_id"]][m] is not None
            ],
            stable_seed(235711, m, "framing"),
        )
        for m in METRICS
    }
    summary = {
        "status": "exploratory main-core geometric analysis; diagnostics and validated retention unavailable",
        "comparison_direction": "positive attack-minus-benign means more drift/variation for selected attack prompts",
        "estimand": "Sampled trajectories averaged within prompt, then equally within underlying task group. Greedy paths separate. Failures not carried forward.",
        "comparisons": comparisons,
        "subgroups": strata,
        "length_matched": {
            "method": "1:1 task-group optimal matching without replacement, caliper factor two in both initial prompt and response token lengths; outcomes not used",
            "n_pairs": len(pairs),
            "differences": matched,
            "pairs": [[x["group_id"], y["group_id"]] for x, y in pairs],
        },
        "benign_framed_minus_plain": framing,
        "input_hashes": {
            name: file_digest(path / name)
            for name in ["events.jsonl", "geometry.json", "text_embeddings.npz"]
        },
        "caveats": [
            "Geometric proximity does not establish original-task fidelity, harmful assistance, or detection accuracy.",
            "Attacks were selected by archived success, not guaranteed fresh initial success.",
            "Lengths, response format, refusals, source dataset and inverse-template behavior remain confounds.",
            "Intervals are 10,000 task-bootstrap percentile intervals; p values use Welch tests on independent task means with Holm adjustment for these six exploratory metrics.",
            "This analysis was specified after collection for a status update; it is not a prespecified confirmatory test or robustness validation.",
            "Only eight sampled initial outputs form the current dispersion baseline; the planned additional fixed-prompt and equal-call comparisons did not run.",
        ],
    }
    write_json(path / "stability_prompt_features.json", rows)
    write_json(path / "stability_step_curves.json", curves)
    write_json(path / "stability_dispersion_curves.json", dispersions)
    write_json(path / "stability_update.json", summary)
    print(
        json.dumps(
            {
                "comparisons": comparisons,
                "length_matched": {"n_pairs": len(pairs), "differences": matched},
                "framing": framing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
