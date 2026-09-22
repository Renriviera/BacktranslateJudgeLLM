"""Schema and matching checks for StrongREJECT FP minimal pairs."""

from __future__ import annotations

import json
from pathlib import Path

PAIRS_PATH = Path("06_Results_Artifacts/results/fp_robustness/minimal_pairs/instantiated_pairs.json")
TEMPLATES_PATH = Path("06_Results_Artifacts/results/fp_robustness/minimal_pairs/templates.json")
FAMILIES = {
    "defensive_inversion",
    "refusal_prevention",
    "consequences_recognition",
    "structural_substitution",
}


def test_templates_cover_four_families():
    templates = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
    ids = {fam["id"] for fam in templates["families"]}
    assert ids == FAMILIES


def test_instantiated_pairs_schema_and_matching():
    payload = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    pairs = payload["pairs"]
    assert len(pairs) >= 8
    counts = {fam: 0 for fam in FAMILIES}
    seen_ids: set[str] = set()
    for pair in pairs:
        fam = pair["family"]
        assert fam in FAMILIES
        counts[fam] += 1
        assert pair["pair_id"] not in seen_ids
        seen_ids.add(pair["pair_id"])
        assert pair["forbidden_prompt_id"].startswith("strongreject:")
        assert pair["forbidden_prompt"].strip()
        assert pair["benign"]["human_label"] == "non_assistance"
        assert pair["harmful"]["role"] == "harmful_structure_only"
        assert pair["harmful"]["human_label"] == "harmful_structure_only"
        text_h = pair["harmful"]["text"]
        assert "[REDACTED" in text_h or "harmful_structure_only" in text_h
        ratio = pair["match"]["word_count_ratio"]
        assert 0.75 <= ratio <= 1.25, (pair["pair_id"], ratio)
        assert abs(pair["match"]["list_item_delta"]) <= 2
    assert all(n >= 2 for n in counts.values()), counts


def test_seed_cases_keep_stored_completions():
    payload = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    seeds = [p for p in payload["pairs"] if p["seed_case"]]
    assert len(seeds) >= 8
    for pair in seeds:
        assert pair["benign"]["origin"] == "stored_attacked_instruct"
        assert pair["benign"].get("stored_strongreject_score") is not None
