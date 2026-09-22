#!/usr/bin/env python
"""Recover missing per-response judge scores into a new experiment directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from brass.paths import recorded_path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.orbits.io import append_jsonl, digest, read_jsonl  # noqa: E402


def main():
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge", choices=["strongreject", "harmbench"], default="strongreject")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    runtime = json.loads((args.run_dir / "preflight.json").read_text())
    inventory = json.loads((args.manifest or args.run_dir / "attack_inventory.json").read_text())
    rows = [
        r
        for r in inventory
        if r.get("details_path") and (args.judge == "harmbench" or r.get("sr_scores") is None)
    ]
    dest = args.run_dir / f"archive_{args.judge}.jsonl"
    done = {r["id"]: r for r in read_jsonl(dest)}
    pending = []
    loaded = {}
    for r in rows:
        path = recorded_path(r["details_path"])
        if path not in loaded:
            loaded[path] = json.loads(path.read_text())
        responses = loaded[path]["completions"]["attacked_instruct"][r["group_id"]]
        for i, response in enumerate(responses):
            key = f'{r["id"]}:{i}'
            pair_hash = digest([r["behavior"], response])
            if key in done:
                assert done[key]["pair_sha256"] == pair_hash
                continue
            pending.append((key, r["behavior"], response, pair_hash))
    if not pending:
        print("No pending archive scores.")
        return
    from brass.metrics.judge_asr import HarmBenchJudge, StrongRejectFinetunedJudge
    from brass.serving.vllm_engine import ModelSpec, VLLMEngine

    model = runtime["models"][args.judge]
    kwargs = dict(
        name=args.judge,
        hf_id=model["snapshot"],
        is_chat=False,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.60,
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
        for start in range(0, len(pending), 128):
            batch = pending[start : start + 128]
            results = judge.score(
                [(key, behavior, response) for key, behavior, response, _ in batch]
            )
            assert len(results) == len(batch)
            output = []
            for result, (key, _, response, pair_hash) in zip(results, batch, strict=True):
                assert result.prompt_id == key and len(result.scores) == 1
                length = len(engine.tokenizer.encode(response))
                output.append(
                    {
                        "id": key,
                        "score": result.scores[0],
                        "pair_sha256": pair_hash,
                        "judge": args.judge,
                        "revision": model["revision"],
                        "response_tokens": length,
                        "judge_truncated": args.judge == "strongreject" and length > 512,
                    }
                )
            append_jsonl(dest, output)
            print(f"Scored {min(start + 128, len(pending))}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
