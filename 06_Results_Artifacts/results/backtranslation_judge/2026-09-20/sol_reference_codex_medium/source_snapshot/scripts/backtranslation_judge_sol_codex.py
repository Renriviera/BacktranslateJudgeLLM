#!/usr/bin/env python3
"""Run the independent Sol reference through ChatGPT-funded Codex sessions."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from brass.sol_reference.codex_runner import DEFAULT_OUT, DEFAULT_PARENT, prepare, run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "run"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--api-run", type=Path, default=DEFAULT_PARENT / "sol_reference_medium_v2")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    a = p.parse_args()
    if a.limit < 0 or not 1 <= a.concurrency <= 8:
        p.error("Limit must be nonnegative; concurrency must be 1 to 8")
    if a.command == "prepare":
        prepare(a.out, a.api_run)
    else:
        run(a.out, a.api_run, a.limit, a.concurrency)


if __name__ == "__main__":
    main()
