#!/usr/bin/env python3
"""Portable entry point; importing this file never launches inference."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "04_Scripts_Experiments/src"
SCRIPTS = ROOT / "04_Scripts_Experiments/scripts"
COMMANDS = {
    "judge": SCRIPTS / "backtranslation_judge.py",
    "sol-api": SCRIPTS / "backtranslation_judge_sol.py",
    "sol-codex": SCRIPTS / "backtranslation_judge_sol_codex.py",
    "sol-analyze": SCRIPTS / "backtranslation_judge_sol_analyze.py",
    "orbits": SCRIPTS / "orbits/run.py",
    "orbit-summary": SCRIPTS / "orbits/summarize.py",
    "orbit-analysis": SCRIPTS / "orbits/analyze.py",
    "size-comparison": SCRIPTS / "orbits/analyze_size_comparison.py",
    "score-benign": SCRIPTS / "orbits/score_benign.py",
    "score-harmful": SCRIPTS / "orbits/score_harmful.py",
    "fp-recurrence": SCRIPTS / "fp_robustness_recurrence.py",
    "prepare": ROOT / "02_Preprocessing/prepare.py",
    "models": ROOT / "03_Models/download_models.py",
    "audit": ROOT / "05_Validation_Metrics/credential_scan.py",
    "verify-artifacts": ROOT / "05_Validation_Metrics/verify_artifacts.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*COMMANDS, "script", "test", "pipeline"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(SOURCE))
    os.environ["PYTHONPATH"] = str(SOURCE) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if args.command == "test":
        import pytest
        raise SystemExit(pytest.main(args.arguments or ["05_Validation_Metrics/tests"]))
    if args.command == "pipeline":
        sys.argv = ["brass.pipeline.run_experiment", *args.arguments]
        runpy.run_module("brass.pipeline.run_experiment", run_name="__main__")
        return
    if args.command == "script":
        if not args.arguments:
            parser.error("script requires a path relative to 04_Scripts_Experiments/scripts")
        script = (SCRIPTS / args.arguments.pop(0)).resolve()
        if not script.is_relative_to(SCRIPTS) or script.suffix != ".py":
            parser.error("script must name a Python entry point inside the scripts directory")
    else:
        script = COMMANDS[args.command]
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *args.arguments]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
