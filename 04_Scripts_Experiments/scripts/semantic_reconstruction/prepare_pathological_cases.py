#!/usr/bin/env python3
"""Materialize response-level cases listed in sibling PATHOLOGICAL_CASES.md."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT.parent / "PATHOLOGICAL_CASES.md"
ATTACKS = ROOT.parent / "attacks"
DATA = ROOT / "04_Scripts_Experiments/src/brass/attacks/external/SlotGCG/data/behavior_datasets"


def catalog_rows() -> list[dict[str, str]]:
    section = None
    rows = []
    for line in CATALOG.read_text().splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        if section not in {"SlotGCG", "PAIR"} or not line.startswith("|"):
            continue
        cells = [part.strip().strip("`") for part in line.strip().strip("|").split("|")]
        if not cells or cells[0] == "Source" or set(cells[0]) <= {"-", ":"}:
            continue
        if len(cells) < 6:
            continue
        rows.append({
            "attack": section.lower(),
            "source": cells[0],
            "behavior_id": cells[1],
            "prefix_success": cells[2],
            "legacy_judge_success": cells[3],
            "pathology": cells[4],
            "excerpt": cells[5],
        })
    return rows


def main() -> None:
    advbench = {}
    with (DATA / "extra_behavior_datasets/advbench_behaviors.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            advbench[row["BehaviorID"]] = row["Behavior"]
    pair_behaviors = {}
    with (ATTACKS / "pair/src/data/harmful_behaviors_custom.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            pair_behaviors[f"pair_{row['Original index']}"] = row["goal"]

    cases = []
    for row in catalog_rows():
        source_path = ATTACKS / row["attack"] / "results/common-evaluation" / row["source"]
        data = json.loads(source_path.read_text())
        record = next((r for r in data["records"] if r["behavior_id"] == row["behavior_id"]), None)
        if record is None:
            raise KeyError(f"{row['behavior_id']} missing from {source_path}")
        behavior = (advbench if row["attack"] == "slotgcg" else pair_behaviors).get(row["behavior_id"])
        if not behavior:
            raise KeyError(f"No clean benchmark behavior found for {row['behavior_id']}")
        case_id = f"{row['attack']}__{Path(row['source']).stem}__{row['behavior_id']}"
        cases.append({
            "case_id": case_id,
            **row,
            "original_behavior": behavior,
            "attack_input": record["test_case"],
            "target_response": record["response"],
            "legacy_prefix_success": record["prefix_success"],
            "legacy_judge": record["qwen_judge"],
            "source_record": str(source_path.relative_to(ROOT.parent)),
        })

    output = ROOT / "06_Results_Artifacts/new_runs/semantic_reconstruction/pathological_cases.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases))
    print(f"Wrote {len(cases)} cases to {output}")


if __name__ == "__main__":
    main()
