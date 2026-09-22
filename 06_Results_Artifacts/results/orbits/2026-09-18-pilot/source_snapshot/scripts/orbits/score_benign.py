#!/usr/bin/env python
"""Score original benign benchmark tasks against each observed forward state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.benchmarks import extract_python, score_basic  # noqa: E402
from brass.orbits.io import append_jsonl, digest, read_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument(
        "--code-image", help="Immutable local Docker image ID; omit to defer code scoring"
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.run_dir / "evaluators"))
    os.environ["NLTK_DATA"] = str((args.run_dir / "evaluators/nltk_data").resolve())
    import langdetect

    langdetect.DetectorFactory.seed = 235711
    from instruction_following_eval.evaluation_lib import (
        InputExample,
        test_instruction_following_strict,
    )

    manifest = {r["id"]: r for r in json.loads(args.manifest.read_text())}
    events = [
        e for e in read_jsonl(args.events_dir / "events.jsonl") if e["direction"] == "forward"
    ]
    output = args.events_dir / "benign_scores.jsonl"
    done = {r["id"] for r in read_jsonl(output)}
    for e in events:
        item = manifest[e["item_id"]]
        if e["id"] in done or "evaluation" not in item:
            continue
        evaluation = item["evaluation"]
        row = {
            "id": e["id"],
            "event_sha256": digest(e),
            "scored_against": "original_task",
            "generation_status": e["status"],
            **score_basic(evaluation, e["text"]),
        }
        if evaluation["kind"] == "ifeval":
            example = InputExample(
                0, evaluation["instruction_id_list"], item["prompt"], evaluation["kwargs"]
            )
            result = test_instruction_following_strict(example, {item["prompt"]: e["text"]})
            row.update(
                correct=result.follow_all_instructions,
                constraints=result.follow_instruction_list,
                evaluator="official_ifeval_strict",
            )
        elif evaluation["kind"] == "code":
            if not args.code_image:
                continue  # Explicitly pending, not silently marked incorrect or completed.
            payload = {"problem": evaluation["spec"], "solution": extract_python(e["text"])}
            command = [
                "docker",
                "run",
                "--rm",
                "-i",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=128",
                "--memory=2g",
                "--cpus=2",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=256m",
                args.code_image,
            ]
            try:
                completed = subprocess.run(
                    command, input=json.dumps(payload), text=True, capture_output=True, timeout=150
                )
                if completed.returncode != 0:
                    row.update(evaluation_error=completed.stderr[-2000:], correct=None)
                else:
                    result = json.loads(completed.stdout.strip().splitlines()[-1])
                    row.update(
                        code_tests=result,
                        correct=all(result[k]["status"] == "pass" for k in ["base", "plus"]),
                        evaluator="evalplus_0.3.1_isolated",
                        image_id=args.code_image,
                    )
            except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                row.update(evaluation_error=type(exc).__name__, correct=None)
        append_jsonl(output, [row])
    print(f"Scored benign events: {len(read_jsonl(output))}")


if __name__ == "__main__":
    main()
