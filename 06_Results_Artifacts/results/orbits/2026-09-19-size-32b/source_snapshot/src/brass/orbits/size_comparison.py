"""Paired two-cycle comparisons and explicitly limited retention proxies."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_1samp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from brass.orbits.io import stable_seed


def terms(text):
    return {
        w
        for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]*|\d+(?:\.\d+)?", text.lower())
        if w not in ENGLISH_STOP_WORDS and (len(w) > 1 or w.isdigit())
    }


def lexical_recall(reference, reconstructed, exclude=""):
    """Unique content-word recall; paraphrases can score low. No entailment claim."""
    wanted = terms(reference) - terms(exclude)
    return len(wanted & terms(reconstructed)) / len(wanted) if wanted else None


def references(item, manifest):
    if item.get("control_type") == "benign_framed":
        content = manifest[item["parent_id"]]["prompt"]
        if item["prompt"].count(content) != 1:
            raise ValueError("Cannot recover an exact benign framing span")
        frame = item["prompt"].replace(content, "", 1).strip()
        return content, frame
    return item.get("behavior", item["prompt"]), None


def paired_groups(rows, metric, cohort=None, subgroup=None, mode="sampled", step=2):
    """Pair paths before prompt averaging, then collapse related prompts into tasks."""
    paths = defaultdict(dict)
    for row in rows:
        if row["step"] != step or row["mode"] != mode:
            continue
        if cohort and row["cohort"] != cohort:
            continue
        if subgroup and row["subgroup"] != subgroup:
            continue
        if row.get(metric) is not None:
            paths[(row["item_id"], row["trajectory"])][row["model"]] = row
    items = defaultdict(list)
    for pair in paths.values():
        if set(pair) == {"7b", "32b"}:
            a, b = pair["7b"], pair["32b"]
            items[(a["group_id"], a["item_id"])].append((a[metric], b[metric]))
    tasks = defaultdict(list)
    for (group_id, _), values in items.items():
        tasks[group_id].append(np.mean(values, axis=0))
    return {
        group_id: np.mean(values, axis=0).tolist() for group_id, values in sorted(tasks.items())
    }


def paired_summary(groups, label, draws=10000):
    if not groups:
        return {
            "label": label,
            "n_groups": 0,
            "mean_7b": None,
            "mean_32b": None,
            "difference_32b_minus_7b": None,
            "ci95": None,
            "p": None,
        }
    a = np.asarray(list(groups.values()), dtype=float)
    delta = a[:, 1] - a[:, 0]
    rng = np.random.default_rng(stable_seed(235711, label, "paired_size"))
    boot = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
    if len(delta) < 2:
        p = None
    elif np.std(delta) < 1e-14:
        p = 1.0 if abs(float(delta.mean())) < 1e-14 else 0.0
    else:
        p = float(ttest_1samp(delta, 0).pvalue)
    return {
        "label": label,
        "n_groups": len(delta),
        "mean_7b": float(a[:, 0].mean()),
        "mean_32b": float(a[:, 1].mean()),
        "difference_32b_minus_7b": float(delta.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "p": p,
    }


def holm(rows):
    valid = sorted((r for r in rows if r["p"] is not None), key=lambda r: r["p"])
    previous = 0.0
    for i, row in enumerate(valid):
        previous = max(previous, min(1.0, row["p"] * (len(valid) - i)))
        row["holm_p"] = previous
    return rows
