#!/usr/bin/env python3
"""Prepare, execute, annotate, calibrate, and analyze the fixed-response judge experiment."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "run",
            "stage",
            "analyze",
            "export-annotations",
            "validate-gold",
            "calibrate",
        ],
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True
    )
    parser.add_argument("--stage", choices=["qwen_pre", "target", "qwen_post", "sft"])
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Development-only pilot row count; 0 runs the entire manifest",
    )
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--gold", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if args.command == "prepare":
        from brass.backtranslation_judge.data import prepare

        print(json.dumps(prepare(REPO, root), indent=2))
    elif args.command in ("run", "stage"):
        from brass.backtranslation_judge.runner import run, stage

        if args.command == "stage":
            if not args.stage:
                parser.error("--stage is required")
            stage(REPO, root, args.stage, args.limit)
        else:
            run(REPO, root, args.limit, args.wait_pid)
    elif args.command == "analyze":
        from brass.backtranslation_judge.analysis import analyze

        if args.gold:
            from brass.backtranslation_judge.annotations import validate_gold

            validate_gold(root, args.gold)
        result = analyze(root, args.gold)
        print(
            json.dumps(
                dict(
                    inference=result["inference"], human_gold_present=result["human_gold_present"]
                ),
                indent=2,
            )
        )
    elif args.command == "export-annotations":
        from brass.backtranslation_judge.annotations import export_annotations

        print(json.dumps(export_annotations(root), indent=2))
    elif args.command in ("calibrate", "validate-gold"):
        if not args.gold:
            parser.error("--gold is required")
        from brass.backtranslation_judge.annotations import validate_gold

        result = validate_gold(root, args.gold)
        if args.command == "calibrate":
            from brass.backtranslation_judge.analysis import calibrate

            result = calibrate(root, args.gold)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
