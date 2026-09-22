#!/usr/bin/env python
"""Analyze the preregistered paired two-cycle checkpoint comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.io import digest, file_digest, read_jsonl, write_json  # noqa: E402
from brass.orbits.size_comparison import (  # noqa: E402
    holm,
    lexical_recall,
    paired_groups,
    paired_summary,
    references,
)


def extend_vectors(texts, vectors, spec, device):
    """Same pinned full-text chunk pooling as the 7B geometry; only embed new texts."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(8)
    model = SentenceTransformer(spec["snapshot"], device=device, local_files_only=True)
    chunk_size = min(192, model.max_seq_length - 16)
    missing = {digest(t): t for t in texts if t is not None and digest(t) not in vectors}
    chunks, owners, weights = [], [], []
    for key, text in missing.items():
        tokens = model.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        for start in range(0, max(len(tokens), 1), chunk_size):
            subset = tokens[start : start + chunk_size]
            chunks.append(model.tokenizer.decode(subset))
            owners.append(key)
            weights.append(max(len(subset), 1))
    if any(
        len(model.tokenizer.encode(c, add_special_tokens=True)) > model.max_seq_length
        for c in chunks
    ):
        raise RuntimeError("Refuse encoder truncation")
    if chunks:
        encoded = model.encode(
            chunks, batch_size=256, normalize_embeddings=False, show_progress_bar=True
        )
        pooled = {k: np.zeros(encoded.shape[1], dtype=np.float64) for k in missing}
        for owner, weight, vector in zip(owners, weights, encoded, strict=True):
            pooled[owner] += weight * vector
        vectors.update({k: v / max(np.linalg.norm(v), 1e-12) for k, v in pooled.items()})
    return {
        "new_texts": len(missing),
        "new_chunks": len(chunks),
        "chunk_tokens": chunk_size,
        "model": spec,
        "device": device,
    }


def similarity(vectors, a, b):
    if a is None or b is None:
        return None
    return float(np.clip(np.dot(vectors[digest(a)], vectors[digest(b)]), -1, 1))


def features(item, manifest, x, y, y0, yprev, vectors):
    content, frame = references(item, manifest)
    result = {
        "response_drift": None if y is None or y0 is None else 1 - similarity(vectors, y, y0),
        "prompt_drift": None if x is None else 1 - similarity(vectors, x, item["prompt"]),
        "response_step_distance": None
        if y is None or yprev is None
        else 1 - similarity(vectors, y, yprev),
        "content_similarity": similarity(vectors, x, content),
        "content_lexical_recall": None if x is None else lexical_recall(content, x),
        "frame_similarity": similarity(vectors, x, frame),
        "frame_lexical_recall": None
        if x is None or frame is None
        else lexical_recall(frame, x, exclude=content),
        "response_content_similarity": similarity(vectors, y, content),
        "response_frame_lexical_recall": None
        if y is None or frame is None
        else lexical_recall(frame, y, exclude=content),
    }
    return result


def base_row(item, model, trajectory, step):
    return {
        "model": model,
        "item_id": item["id"],
        "group_id": item.get("analysis_group_id", item["group_id"]),
        "cohort": item["cohort"],
        "subgroup": item.get("control_type", item.get("family", item.get("stratum"))),
        "trajectory": trajectory,
        "mode": "greedy" if trajectory == "greedy" else "sampled",
        "step": step,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    root = args.run_dir
    protocol = json.loads((root / "frozen/protocol.json").read_text())
    runtime = json.loads((root / "preflight.json").read_text())
    manifest = {r["id"]: r for r in json.loads((root / "frozen/manifest.json").read_text())}
    sources = {"7b": root / "baseline_7b", "32b": root / "main_32b"}
    events = {m: {e["id"]: e for e in read_jsonl(p / "events.jsonl")} for m, p in sources.items()}
    fixed = {e["id"]: e for e in read_jsonl(root / "fixed_7b_response_inverse_32b/events.jsonl")}
    vectors = {}
    for path in sources.values():
        with np.load(path / "text_embeddings.npz") as archive:
            vectors.update(dict(zip(archive["keys"], archive["vectors"], strict=True)))
    text_refs = [
        t for item in manifest.values() for t in references(item, manifest) if t is not None
    ]
    text_refs += [e["reconstructed_prompt"] for e in fixed.values() if e["status"] == "ok"]
    embedding = extend_vectors(text_refs, vectors, runtime["models"]["embedding"], args.device)
    write_json(root / "retention_embedding_provenance.json", embedding)
    scores = {}
    for model, path in sources.items():
        scores[model] = {
            name: {r["id"]: r for r in read_jsonl(path / f"{name}_scores.jsonl")}
            for name in ["benign", "strongreject", "harmbench"]
        }
    rows = []
    trajectories = [f"sample:{i}" for i in range(8)] + ["greedy"]
    for model, observed in events.items():
        for item in manifest.values():
            for trajectory in trajectories:
                prefix = f"{item['id']}|{trajectory}|"
                initial = observed.get(prefix + "0")
                y0 = initial["text"] if initial and initial["status"] == "ok" else None
                previous_y = None
                for step in range(3):
                    f = observed.get(prefix + str(2 * step))
                    g = observed.get(prefix + str(2 * step - 1)) if step else None
                    x = (
                        item["prompt"]
                        if step == 0
                        else g["reconstructed_prompt"]
                        if g and g["status"] == "ok"
                        else None
                    )
                    y = f["text"] if f and f["status"] == "ok" else None
                    row = {
                        **base_row(item, model, trajectory, step),
                        **features(item, manifest, x, y, y0, previous_y, vectors),
                        "forward_status": f["status"] if f else "missing_after_failure",
                        "inverse_status": g["status"]
                        if g
                        else "not_applicable"
                        if step == 0
                        else "missing_after_failure",
                        "output_tokens": f["output_tokens"] if f else None,
                    }
                    for name, lookup in scores[model].items():
                        score = lookup.get(prefix + str(2 * step), {})
                        row[name + "_score"] = score.get("correct" if name == "benign" else "score")
                        row[name + "_status"] = score.get("status", score.get("evaluator"))
                    rows.append(row)
                    previous_y = y
    fixed_rows = []
    for key, ref in events["7b"].items():
        if ref["direction"] != "inverse":
            continue
        item = manifest[ref["item_id"]]
        for model, e in [("7b", ref), ("32b", fixed.get(key))]:
            x = e["reconstructed_prompt"] if e and e["status"] == "ok" else None
            fixed_rows.append(
                {
                    **base_row(item, model, ref["trajectory"], ref["step"]),
                    **features(item, manifest, x, None, None, None, vectors),
                    "inverse_status": e["status"] if e else "missing",
                }
            )
    primary = []
    for test in protocol["primary_tests"]:
        label = ":".join([test.get("subgroup", test["cohort"]), test["metric"], "cycle2"])
        primary.append({**test, **paired_summary(paired_groups(rows, **test), label)})
    holm(primary)
    fixed_tests = [{"cohort": c, "metric": "content_similarity"} for c in ["attack", "benign"]] + [
        {"cohort": "control", "subgroup": "benign_framed", "metric": m}
        for m in ["frame_lexical_recall", "frame_similarity"]
    ]
    fixed_results = []
    for test in fixed_tests:
        label = "fixed_response:" + test.get("subgroup", test["cohort"]) + ":" + test["metric"]
        fixed_results.append(
            {**test, **paired_summary(paired_groups(fixed_rows, **test, step=1), label)}
        )
    holm(fixed_results)
    curves = []
    for mode in ["sampled", "greedy"]:
        for cohort in ["attack", "benign", "control"]:
            for step in range(3):
                for metric in [
                    "response_drift",
                    "prompt_drift",
                    "response_step_distance",
                    "content_similarity",
                ]:
                    label = f"{mode}:{cohort}:{step}:{metric}"
                    curves.append(
                        {
                            "mode": mode,
                            "cohort": cohort,
                            "step": step,
                            "metric": metric,
                            **paired_summary(
                                paired_groups(rows, metric, cohort=cohort, mode=mode, step=step),
                                label,
                            ),
                        }
                    )
    subgroups = []
    for subgroup in sorted({r["subgroup"] for r in rows}):
        for metric in [
            "response_drift",
            "prompt_drift",
            "content_similarity",
            "content_lexical_recall",
        ]:
            label = f"{subgroup}:{metric}"
            subgroups.append(
                {
                    "subgroup": subgroup,
                    "metric": metric,
                    **paired_summary(paired_groups(rows, metric, subgroup=subgroup), label),
                }
            )
    coverage = []
    for model in ["7b", "32b"]:
        for cohort in ["attack", "benign", "control"]:
            for step in range(3):
                selected = [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["cohort"] == cohort
                    and r["step"] == step
                    and r["mode"] == "sampled"
                ]
                coverage.append(
                    {
                        "model": model,
                        "cohort": cohort,
                        "step": step,
                        "planned_paths": len(selected),
                        "forward_status": dict(Counter(r["forward_status"] for r in selected)),
                        "inverse_status": dict(Counter(r["inverse_status"] for r in selected)),
                        "scorers": {
                            name: {
                                "observed": sum(r[name + "_score"] is not None for r in selected),
                                "positive": sum(
                                    r[name + "_score"] is not None and r[name + "_score"] >= 0.5
                                    for r in selected
                                ),
                            }
                            for name in ["benign", "strongreject", "harmbench"]
                        },
                    }
                )
    for name, value in [
        ("paired_features", rows),
        ("fixed_inverse_features", fixed_rows),
        ("paired_curves", curves),
        ("subgroup_comparisons", subgroups),
        ("coverage", coverage),
    ]:
        write_json(root / f"{name}.json", value)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_tests": primary,
        "fixed_response_tests": fixed_results,
        "protocol": protocol,
        "coverage": coverage,
        "input_hashes": {
            str(p): file_digest(p)
            for p in [
                root / "frozen/protocol.json",
                root / "baseline_7b/events.jsonl",
                root / "main_32b/events.jsonl",
                root / "fixed_7b_response_inverse_32b/events.jsonl",
            ]
        },
        "interpretation": "Negative differences favor 32B for drift; positive differences favor 32B for retention. Paired complete cases; consult coverage and benchmark effects. No pure parameter-count or entailment claim.",
    }
    write_json(root / "comparison_summary.json", summary)

    def line(r):
        if r["n_groups"] == 0:
            return f"- {r['label']}: no valid paired task groups."
        lo, hi = r["ci95"]
        p = r.get("holm_p")
        ptext = f"{p:.4g}" if p is not None else "unavailable"
        return f"- {r['label']}: 7B {r['mean_7b']:.4f}; 32B {r['mean_32b']:.4f}; difference {r['difference_32b_minus_7b']:+.4f}, 95% paired task-bootstrap interval [{lo:+.4f}, {hi:+.4f}], Holm p={ptext}; {r['n_groups']} groups."

    report = [
        "# Two-cycle OLMo checkpoint comparison",
        "",
        "Same 491 prompts, eight sampled trajectories and one greedy trajectory; 7B trajectories reused through cycle two. Both mappings use 32B in the new main arm. The inverse sees only the preceding response.",
        "",
        "The 32B checkpoint is Olmo-3.1-32B-Instruct; the reference is Olmo-3-7B-Instruct. This is a checkpoint comparison, not an isolated causal parameter-count intervention. The 7B chat template, stop tokens, decoding settings and per-call seeds are matched.",
        "",
        "## Prespecified primary paired comparisons (sampled, cycle two)",
        "",
        "Differences are 32B minus 7B. Lower drift and higher retention favor 32B. Confidence intervals are pointwise; seven primary p values have Holm correction. Related attack prompts are grouped by original task.",
        "",
        *[line(r) for r in primary],
        "",
        "## Same-response inverse diagnostic",
        "",
        "Both inverse models receive identical saved 7B responses and identical inverse instructions. These first-inversion comparisons have their own four-test Holm family; they isolate inverse checkpoint differences on this 7B-response distribution.",
        "",
        *[line(r) for r in fixed_results],
        "",
        "## Interpretation limits",
        "",
        *[f"- {s}" for s in protocol["limits"]],
        "- Framing estimates concern 40 known benign wrappers, not a gold decomposition of actual attack framing. Lexical recall penalizes paraphrases; embedding similarity can reward topic overlap without preserving constraints.",
        "- Invalid and truncated paths are excluded pairwise from each metric, with all planned path counts in coverage.json. Missing judge labels are not negative. HarmBench cannot label overlength full responses; StrongREJECT may see only a prefix.",
        "- Original-task benchmark scores and initial attack success should be inspected alongside geometry. A stable refusal is not retained task fulfillment.",
        "",
        "Machine-readable results: comparison_summary.json, subgroup_comparisons.json, paired_curves.json, coverage.json and paired_features.json.",
        "",
        "![Response drift by cycle](response_drift.png)",
        "",
    ]
    (root / "comparison_report.md").write_text("\n".join(report))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, cohort in zip(axes, ["attack", "benign"], strict=True):
        selected = sorted(
            [
                r
                for r in curves
                if r["mode"] == "sampled"
                and r["cohort"] == cohort
                and r["metric"] == "response_drift"
            ],
            key=lambda r: r["step"],
        )
        for model, label in [("7b", "OLMo 3 7B Instruct"), ("32b", "OLMo 3.1 32B Instruct")]:
            ax.plot(
                [r["step"] for r in selected],
                [r["mean_" + model] for r in selected],
                marker="o",
                label=label,
            )
        ax.set(title=cohort.capitalize(), xlabel="Completed round trips", xticks=[0, 1, 2])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean response cosine distance from cycle 0")
    axes[1].legend(fontsize=8)
    fig.suptitle("Two-cycle response drift: paired task-group means")
    fig.text(
        0.02,
        0.01,
        "Source: frozen 2026-09-19 size study; eight sampled paths/prompt; paired complete cases.",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(root / "response_drift.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "report": str(root / "comparison_report.md"),
                "primary_tests": len(primary),
                "fixed_response_tests": len(fixed_results),
            }
        )
    )


if __name__ == "__main__":
    main()
