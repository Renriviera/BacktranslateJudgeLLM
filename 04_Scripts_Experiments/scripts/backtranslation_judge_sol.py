#!/usr/bin/env python3
"""Prepare/run the independent Sol StrongREJECT reference for the frozen study."""

import argparse
import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))

from brass.sol_reference.runner import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_PARENT,
    prepare,
    reuse_previous,
    run,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--effort", choices=["low", "medium"], default="medium")
    parser.add_argument(
        "--reuse-from",
        type=Path,
        help="Prepare only: import identical completed judgments into a fresh run",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Development-only pilot size; zero runs every pair"
    )
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.limit < 0 or not 1 <= args.concurrency <= 16:
        parser.error("Invalid pilot limit or concurrency")
    if args.reuse_from and args.command != "prepare":
        parser.error("--reuse-from requires prepare")
    if args.command == "prepare":
        spec = prepare(args.parent_run, args.out, args.effort)
        if args.reuse_from:
            reuse_previous(args.reuse_from, args.out)
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
