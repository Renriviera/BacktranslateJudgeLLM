"""Outcome-blind lexical near-duplicate groups for evaluation splits."""

import re
import unicodedata
from collections import defaultdict

from brass.orbits.io import digest


def grouped_main(main, pilot):
    """Union native task groups at five-word-shingle Jaccard >= .85.

    This conservative lexical check cannot establish semantic independence. Native source
    groups (including shared passages and behavior variants) are always kept together.
    """
    groups, texts = {}, defaultdict(set)
    for row in [*pilot, *main]:
        group = row["group_id"]
        groups[group] = group
        text = row.get("behavior", row["prompt"])
        tokens = tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower()))
        texts[group].add(tokens)

    def root(g):
        while groups[g] != g:
            groups[g] = groups[groups[g]]
            g = groups[g]
        return g

    profiles = []
    for group, variants in sorted(texts.items()):
        for tokens in sorted(variants):
            profiles.append(
                (group, tokens, set(zip(*(tokens[i:] for i in range(5)), strict=False)))
            )
    pairs = []
    for i, (ga, ta, sa) in enumerate(profiles):
        for gb, tb, sb in profiles[i + 1 :]:
            if root(ga) == root(gb):
                continue
            similar = ta == tb
            if (
                not similar
                and min(len(ta), len(tb)) >= 12
                and min(len(ta), len(tb)) >= 0.8 * max(len(ta), len(tb))
            ):
                union = len(sa | sb)
                similar = bool(union and len(sa & sb) / union >= 0.85)
            if similar:
                a, b = sorted([root(ga), root(gb)])
                groups[b] = a
                pairs.append([ga, gb])
    components = defaultdict(list)
    for g in groups:
        components[root(g)].append(g)
    canonical = {
        g: members[0] if len(members) == 1 else "near:" + digest(sorted(members))[:20]
        for members in components.values()
        for g in members
    }
    excluded = {canonical[r["group_id"]] for r in pilot}
    kept, removed = [], []
    for row in main:
        item = {**row, "analysis_group_id": canonical[row["group_id"]]}
        (removed if item["analysis_group_id"] in excluded else kept).append(item)
    return kept, {
        "algorithm": "NFKC-lowercase word 5-shingle Jaccard >=0.85 with length ratio >=0.8; exact match for short texts; transitive native-group union",
        "near_duplicate_pairs": pairs,
        "removed_main_ids": [r["id"] for r in removed],
        "source_to_analysis_group": canonical,
        "n_main_analysis_groups": len({r["analysis_group_id"] for r in kept}),
        "limitation": "Lexical screening plus native task groups; semantic paraphrases may remain.",
    }
