#!/usr/bin/env python
"""Teacher-forced reconstruction surprisal; a model cross-entropy, not Shannon MI."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.io import append_jsonl, digest, read_jsonl  # noqa: E402
from brass.orbits.runner import Budget, inverse_messages  # noqa: E402


def main():
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    args = parser.parse_args()
    from vllm import SamplingParams

    from brass.serving.vllm_engine import ModelSpec, VLLMEngine

    manifest = {r["id"]: r for r in json.loads(args.manifest.read_text())}
    runtime = json.loads((args.run_dir / "preflight.json").read_text())
    model = runtime["models"]["target"]
    events = [
        e for e in read_jsonl(args.events_dir / "events.jsonl") if e["direction"] == "forward"
    ]
    events += [{"id": "prior:" + key, "item_id": key, "text": ""} for key in manifest]
    dest = args.events_dir / "inverse_surprisal.jsonl"
    done = {r["id"] for r in read_jsonl(dest)}
    events = [e for e in events if e["id"] not in done]
    if not events:
        return
    budget = Budget(args.run_dir / "token_budget.jsonl", 80_000_000)
    spec = ModelSpec(
        name="inverse_surprisal",
        hf_id=model["snapshot"],
        is_chat=True,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.8,
        extra={"enforce_eager": True, "max_num_seqs": 64},
    )
    with VLLMEngine(spec, seed=235711) as engine:
        tok = engine.tokenizer
        for start in range(0, len(events), 32):
            batch = events[start : start + 32]
            requests, spans, kept, excluded = [], [], [], []
            for e in batch:
                context = tok.apply_chat_template(
                    inverse_messages(e["text"]), tokenize=False, add_generation_prompt=True
                )
                prefix = context + '{"prompt": "'
                content = json.dumps(manifest[e["item_id"]]["prompt"], ensure_ascii=False)[1:-1]
                prefix_ids = tok.encode(prefix, add_special_tokens=False)
                full_ids = tok.encode(prefix + content, add_special_tokens=False)
                # Locate the first differing token to handle tokenization across the boundary.
                boundary = 0
                for a, b in zip(prefix_ids, full_ids, strict=False):
                    if a != b:
                        break
                    boundary += 1
                if len(full_ids) + 1 > 8192:
                    excluded.append(
                        {"id": e["id"], "status": "context_overflow", "bits_per_token": None}
                    )
                    continue
                requests.append({"prompt_token_ids": full_ids})
                spans.append((max(1, boundary), full_ids))
                kept.append(e)
            append_jsonl(dest, excluded)
            if not requests:
                continue
            reservation = digest([str(dest), start, time.time_ns()])
            budget.reserve(reservation, len(requests))
            outputs = engine.llm.generate(
                requests,
                SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
                use_tqdm=False,
            )
            rows = []
            for e, (boundary, ids), out in zip(kept, spans, outputs, strict=True):
                lps = out.prompt_logprobs
                if lps is None or len(lps) != len(ids):
                    raise RuntimeError("Missing teacher-forced token log probabilities")
                values = []
                for pos in range(boundary, len(ids)):
                    entry = lps[pos]
                    if not entry or ids[pos] not in entry:
                        raise RuntimeError("Realized token absent from prompt log probabilities")
                    values.append(entry[ids[pos]].logprob)
                bits = -sum(values) / math.log(2)
                rows.append(
                    {
                        "id": e["id"],
                        "status": "ok",
                        "content_bits": bits,
                        "scored_tokens": len(values),
                        "bits_per_token": bits / len(values),
                        "scoring_input_tokens": len(ids),
                        "model_revision": model["revision"],
                        "event_sha256": digest(e),
                        "interpretation": "cross_entropy_of_original_prompt_content_conditioned_on_response_and_JSON_prefix",
                    }
                )
            append_jsonl(dest, rows)
            budget.settle(reservation, sum(len(out.outputs[0].token_ids) for out in outputs))
            print(f"Inverse surprisal: {start + len(batch)}/{len(events)}", flush=True)


if __name__ == "__main__":
    main()
