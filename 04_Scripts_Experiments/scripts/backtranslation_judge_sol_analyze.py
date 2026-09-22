#!/usr/bin/env python3
"""Analyze the completed Sol reference without modifying the original inference run."""

import argparse
import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.sol_reference.analysis import analyze  # noqa: E402
from brass.sol_reference.codex_runner import DEFAULT_OUT  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Export explicitly provisional results; do not make whole-study claims",
    )
    args = parser.parse_args()
    result = analyze(args.run_dir, allow_partial=args.allow_partial)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ["model", "selected_records", "reference_complete", "usage"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
