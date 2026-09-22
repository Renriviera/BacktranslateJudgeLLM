#!/usr/bin/env python
"""Judge observed forward states against their original benchmark behavior."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.io import append_jsonl, digest, read_jsonl  # noqa: E402
from brass.orbits.runner import Budget  # noqa: E402


def main():
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--judge", choices=["strongreject", "harmbench"], required=True)
    args = parser.parse_args()
    from brass.metrics.judge_asr import (
        HARMBENCH_TEMPLATE,
        HarmBenchJudge,
        StrongRejectFinetunedJudge,
    )
    from brass.serving.vllm_engine import ModelSpec, VLLMEngine

    manifest = {r["id"]: r for r in json.loads(args.manifest.read_text())}
    runtime = json.loads((args.run_dir / "preflight.json").read_text())
    model = runtime["models"][args.judge]
    events = [
        e
        for e in read_jsonl(args.events_dir / "events.jsonl")
        if e["direction"] == "forward" and "behavior" in manifest[e["item_id"]]
    ]
    dest = args.events_dir / f"{args.judge}_scores.jsonl"
    done = {r["id"] for r in read_jsonl(dest)}
    pending = [e for e in events if e["id"] not in done]
    if not pending:
        return
    # Include prior archive judge calls in the same study-wide ceiling.
    budget = Budget(args.run_dir / "judge_budget.jsonl", 200_000)
    if not budget.path.exists():
        archived = sum(
            len(read_jsonl(args.run_dir / f"archive_{j}.jsonl"))
            for j in ["strongreject", "harmbench"]
        )
        budget.reserve("archived-judge-calls", archived)
        budget.settle("archived-judge-calls", archived)
    kwargs = dict(
        name=args.judge,
        hf_id=model["snapshot"],
        is_chat=False,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.70,
        extra={"enforce_eager": True},
    )
    if args.judge == "strongreject":
        kwargs.update(
            enable_lora=True,
            lora_path=runtime["models"]["strongreject_adapter"]["snapshot"],
            max_lora_rank=16,
            extra={"enforce_eager": True, "max_logprobs": 25},
        )
    with VLLMEngine(ModelSpec(**kwargs), seed=235711) as engine:
        judge = (
            StrongRejectFinetunedJudge(engine)
            if args.judge == "strongreject"
            else HarmBenchJudge(engine)
        )
        if args.judge == "harmbench":
            # Never extend a classifier past its native position limit or silently crop it.
            # Unscorable full responses remain missing labels, not negative classifications.
            kept, excluded = [], []
            for e in pending:
                rendered = HARMBENCH_TEMPLATE.format(
                    behavior=manifest[e["item_id"]]["behavior"], generation=e["text"]
                )
                input_tokens = len(engine.tokenizer.encode(rendered))
                if input_tokens + 1 > 2048:
                    excluded.append(
                        {
                            "id": e["id"],
                            "score": None,
                            "judge": args.judge,
                            "status": "judge_context_overflow",
                            "model_revision": model["revision"],
                            "generation_status": e["status"],
                            "judge_input_tokens": input_tokens,
                            "judge_context_limit": 2048,
                            "event_sha256": digest(e),
                            "scored_against": "original_unwrapped_behavior",
                            "label_status": "unscored_full_response",
                            "judge_truncated": False,
                        }
                    )
                else:
                    kept.append(e)
            append_jsonl(dest, excluded)
            pending = kept
            print(
                f"HarmBench full-response context exclusions: {len(excluded)}; eligible: {len(kept)}",
                flush=True,
            )
        for start in range(0, len(pending), 64):
            batch = pending[start : start + 64]
            # Retain an interrupted attempt's charge, but permit an explicit resume.
            reservation = digest([str(dest), [e["id"] for e in batch], time.time_ns()])
            budget.reserve(reservation, len(batch))
            triples = [(e["id"], manifest[e["item_id"]]["behavior"], e["text"]) for e in batch]
            results = judge.score(triples)
            rows = []
            for e, result in zip(batch, results, strict=True):
                assert result.prompt_id == e["id"]
                length = len(engine.tokenizer.encode(e["text"]))
                rows.append(
                    {
                        "id": e["id"],
                        "score": result.scores[0],
                        "status": "ok",
                        "judge": args.judge,
                        "model_revision": model["revision"],
                        "generation_status": e["status"],
                        "response_tokens": length,
                        "judge_truncated": args.judge == "strongreject" and length > 512,
                        "event_sha256": digest(e),
                        "scored_against": "original_unwrapped_behavior",
                        "label_status": "judge_provisional",
                    }
                )
            append_jsonl(dest, rows)
            budget.settle(reservation, len(batch))
            print(f"{args.judge}: {start + len(batch)}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
