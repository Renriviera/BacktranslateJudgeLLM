#!/usr/bin/env python3
"""Read-only archive audit for the proposed backtranslation judge study.

No model inference, API calls, or human labels. Writes only the requested report.
Run: python3 scripts/backtranslation_judge_inventory.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "pap_authority": "pap/olmo3_7b_instruct.authority_endorsement.json",
    "pap_logic": "pap/olmo3_7b_instruct.logical_appeal.json",
    "pap_misrep": "pap/olmo3_7b_instruct.json",
    "pair": "pair/olmo3_7b_instruct.json",
    "slotgcg": "slotgcg/olmo3_7b_instruct.json",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/backtranslation_judge/design_inventory.json")
    args = parser.parse_args()
    sources = {}

    def read(path):
        path = ROOT / path
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    recovery_path = "results/orbits/2026-09-18-pilot/archive_strongreject.jsonl"
    recovery = {}
    for line in read(recovery_path).split("\n"):
        if line.strip():
            row = json.loads(line)
            assert row["id"] not in recovery, "Duplicate recovered score key"
            recovery[row["id"]] = row
    selected = json.loads(read(".audit_selected.json"))
    pairs = json.loads(read("results/fp_robustness/minimal_pairs/instantiated_pairs.json"))
    known = {r["id"] for r in selected["rows"]}
    known.update(r["forbidden_prompt_id"] for r in pairs["pairs"])
    known.update(f"strongreject:{i}" for i in [7, 64, 174, 248, 278])
    report = []
    canonical = None
    all_pair_hashes = Counter()
    for variant, cache_relative in VARIANTS.items():
        detail_path = f"results/{variant}_strongreject_olmo3_7b/details.json"
        data = json.loads(read(detail_path))
        cache = json.loads(read(f"results/attacks/{cache_relative}"))
        prompts = {p["id"]: p for p in data["prompts"]}
        identities = {pid: p["prompt"] for pid, p in prompts.items()}
        if canonical is None:
            canonical = identities
        assert identities == canonical, "Behavior mapping differs across variants"
        counts = Counter()
        status = Counter()
        per_prompt = data["judges"]["strongreject"]["per_prompt"]
        for pid, prompt in prompts.items():
            assert pid in cache, f"Missing attack cache record: {variant}/{pid}"
            cached = cache[pid]
            status[cached.get("pair_status", cached.get("pap_status", "unrecorded"))] += 1
            counts["attacked_prompt_equals_original"] += cached["attacked_prompt"] == prompt["prompt"]
            responses = data["completions"]["attacked_instruct"][pid]
            assert len(responses) == 16
            vector = per_prompt[pid].get("scores")
            assert vector is None or len(vector) == len(responses)
            for i, response in enumerate(responses):
                pair_hash = digest([prompt["prompt"], response])
                all_pair_hashes[pair_hash] += 1
                counts["responses"] += 1
                if vector is not None:
                    score = vector[i]
                    counts["scores_from_details"] += 1
                else:
                    recovered = recovery[f"{variant}:{pid}:{i}"]
                    assert recovered["pair_sha256"] == pair_hash, "Recovered score does not match exact response"
                    score = recovered["score"]
                    counts["scores_recovered_hash_verified"] += 1
                    counts["recovered_scores_truncated"] += bool(recovered.get("judge_truncated"))
                assert score is not None and 0 <= score <= 1
                counts["sr_ft_ge_0_5"] += score >= 0.5
        report.append({
            "variant": variant,
            "details_path": detail_path,
            "attack_cache_path": f"results/attacks/{cache_relative}",
            "behaviors": len(prompts),
            "categories": dict(Counter(p["category"] for p in prompts.values())),
            "attack_status": dict(status),
            **dict(counts),
        })
    assert canonical is not None
    output = {
        "status": "design_inventory_only_no_new_judgments_or_gold_labels",
        "target": "allenai/Olmo-3-7B-Instruct",
        "variants": report,
        "total_attacked_responses": sum(r["responses"] for r in report),
        "behavior_ids": len(canonical),
        "exact_unique_behavior_texts": len(set(canonical.values())),
        "unique_behavior_response_pairs": len(all_pair_hashes),
        "repeated_behavior_response_records": sum(n - 1 for n in all_pair_hashes.values()),
        "known_development_behavior_ids": sorted(known, key=lambda s: int(s.split(":")[1])),
        "known_development_ids_note": "Minimum exclusion set, not a complete exposure history; include relatives and other reviewed cases before freezing splits.",
        "proposed_sample": {
            "responses_per_behavior_per_variant": 4,
            "total": len(canonical) * len(VARIANTS) * 4,
            "sampling": "uniform without replacement; independent of judge score; inclusion probability 1/4 within each cell",
            "provisional_behavior_split": {"development": 60, "calibration": 60, "test": 193},
            "split_status": "not_generated; near-duplicate and prior-exposure audit required before freeze",
        },
        "source_sha256": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "responses": output["total_attacked_responses"],
                      "behaviors": output["behavior_ids"], "unique_pairs": len(all_pair_hashes),
                      "known_development_ids": len(known),
                      "recovered_hash_verified": sum(r.get("scores_recovered_hash_verified", 0) for r in report)}, indent=2))


if __name__ == "__main__":
    main()
