#!/usr/bin/env python3
"""Prepare/run the independent Sol StrongREJECT reference for the frozen study."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from brass.sol_reference.runner import DEFAULT_OUT, DEFAULT_PARENT, prepare, run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--effort", choices=["low", "medium"], default="medium")
    parser.add_argument(
        "--limit", type=int, default=0, help="Development-only pilot size; zero runs every pair"
    )
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.limit < 0 or not 1 <= args.concurrency <= 16:
        parser.error("Invalid pilot limit or concurrency")
    if args.command == "prepare":
        spec = prepare(args.parent_run, args.out, args.effort)
        print(
            json.dumps(
                {
                    k: spec[k]
                    for k in ["model", "reasoning_effort", "selected_records", "unique_pairs"]
                },
                indent=2,
            )
        )
    else:
        run(args.parent_run, args.out, args.effort, args.limit, args.concurrency)


if __name__ == "__main__":
    main()
