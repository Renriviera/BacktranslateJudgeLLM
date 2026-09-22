#!/usr/bin/env python
"""Judge observed forward states against their original benchmark behavior."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    from brass.metrics.judge_asr import HarmBenchJudge, StrongRejectFinetunedJudge
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
        max_model_len=8192 if args.judge == "harmbench" else 2048,
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
        for start in range(0, len(pending), 64):
            batch = pending[start : start + 64]
            reservation = digest([str(dest), [e["id"] for e in batch]])
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
