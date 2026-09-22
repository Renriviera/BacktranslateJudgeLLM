#!/usr/bin/env python3
"""Merge two human ratings and optional third-rater decisions, preserving unresolved cases."""

import argparse
import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))

from brass.orbits.io import read_jsonl  # noqa: E402


def indexed(path):
    result = {}
    for row in read_jsonl(path):
        aid = row["annotation_id"]
        if aid in result:
            raise ValueError("Duplicate annotation in one rater file")
        if not row.get("rater_id"):
            raise ValueError("Rater ID is required")
        if type(row.get("assistance")) is not int or row["assistance"] not in range(4):
            raise ValueError("Assistance must be 0..3")
        if row.get("prohibited") is not None and type(row["prohibited"]) is not bool:
            raise ValueError("Prohibitedness must be boolean or null")
        result[aid] = row
    return result


def assemble(rater_a, rater_b, decisions=None):
    a, b = indexed(rater_a), indexed(rater_b)
    third = indexed(decisions) if decisions else {}
    gold = []
    for aid in sorted(set(a) & set(b)):
        first, second = a[aid], b[aid]
        if first["rater_id"] == second["rater_id"]:
            raise ValueError("Two distinct independent human raters are required")
        row = dict(
            annotation_id=aid,
            rater_ids=[first["rater_id"], second["rater_id"]],
            individual_ratings=[first, second],
        )
        if aid in third:
            decision = third[aid]
            if decision["rater_id"] in row["rater_ids"]:
                raise ValueError("Adjudicator must be distinct from the two initial raters")
            row.update(label_source="human_adjudicated", adjudicator_id=decision["rater_id"])
        elif (first["assistance"], first["prohibited"]) == (
            second["assistance"],
            second["prohibited"],
        ) and not any(r.get("uncertain") for r in [first, second]):
            decision = first
            row.update(label_source="human_consensus")
        else:
            gold.append(
                {
                    **row,
                    "label_source": "human_review_pending",
                    "status": "uncertain",
                    "assistance": None,
                    "prohibited": None,
                    "evidence": [],
                    "notes": "Independent raters disagree or are uncertain; third-rater adjudication pending.",
                }
            )
            continue
        uncertain = decision.get("uncertain", False) or decision["prohibited"] is None
        row.update(
            status="uncertain" if uncertain else "resolved",
            assistance=decision["assistance"],
            prohibited=decision["prohibited"],
            evidence=decision.get("evidence", []),
            notes=decision.get("notes", ""),
        )
        gold.append(row)
    if not gold:
        raise ValueError("No response has two human ratings")
    return gold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater-a", type=Path, required=True)
    parser.add_argument("--rater-b", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gold = assemble(args.rater_a, args.rater_b, args.adjudications)
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing gold file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gold))
    print(
        json.dumps(
            dict(
                rows=len(gold),
                resolved=sum(r["status"] == "resolved" for r in gold),
                output=str(args.output),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
