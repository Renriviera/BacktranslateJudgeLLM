#!/usr/bin/env python
"""Chunk-pooled semantic geometry, using all text rather than encoder truncation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.orbits.io import digest, read_jsonl, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(8)
    runtime = json.loads((args.run_dir / "preflight.json").read_text())
    model_spec = runtime["models"]["embedding"]
    model = SentenceTransformer(model_spec["snapshot"], device=args.device, local_files_only=True)
    # Leave room for WordPiece fragments that expand when decoded/re-tokenized.
    chunk_size = min(192, model.max_seq_length - 16)
    manifest = {r["id"]: r for r in json.loads(args.manifest.read_text())}
    events = read_jsonl(args.events_dir / "events.jsonl")
    texts = {}
    for r in manifest.values():
        texts[digest(r["prompt"])] = r["prompt"]
    for e in events:
        texts[digest(e["text"])] = e["text"]
        if e["reconstructed_prompt"]:
            texts[digest(e["reconstructed_prompt"])] = e["reconstructed_prompt"]
    chunks, owners, weights = [], [], []
    for key, text in texts.items():
        tokens = model.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        for start in range(0, max(len(tokens), 1), chunk_size):
            subset = tokens[start : start + chunk_size]
            chunks.append(model.tokenizer.decode(subset))
            owners.append(key)
            weights.append(max(len(subset), 1))
    if any(
        len(model.tokenizer.encode(chunk, add_special_tokens=True)) > model.max_seq_length
        for chunk in chunks
    ):
        raise RuntimeError(
            "A reconstructed chunk exceeds encoder context; refuse silent truncation"
        )
    vectors = model.encode(
        chunks, batch_size=args.batch_size, normalize_embeddings=False, show_progress_bar=True
    )
    pooled = defaultdict(lambda: np.zeros(vectors.shape[1], dtype=np.float64))
    for owner, weight, vector in zip(owners, weights, vectors, strict=True):
        pooled[owner] += weight * vector
    pooled = {key: value / max(np.linalg.norm(value), 1e-12) for key, value in pooled.items()}
    keys = sorted(pooled)
    np.savez_compressed(
        args.events_dir / "text_embeddings.npz",
        keys=np.array(keys),
        vectors=np.array([pooled[k] for k in keys]),
    )

    def distance(a, b):
        return float(np.clip(1 - np.dot(pooled[digest(a)], pooled[digest(b)]), 0, 2))

    paths = defaultdict(list)
    for e in events:
        paths[(e["item_id"], e["trajectory"])].append(e)
    measurements = []
    for (item_id, trajectory), path in paths.items():
        forward = sorted([e for e in path if e["direction"] == "forward"], key=lambda e: e["step"])
        initial_y = forward[0]["text"]
        initial_x = manifest[item_id]["prompt"]
        for i, e in enumerate(forward):
            x = e["input_messages"][0]["content"]
            measurements.append(
                {
                    "id": e["id"],
                    "item_id": item_id,
                    "group_id": manifest[item_id].get(
                        "analysis_group_id", manifest[item_id]["group_id"]
                    ),
                    "cohort": e["cohort"],
                    "step": e["step"],
                    "trajectory": trajectory,
                    "generation_status": e["status"],
                    "response_distance_from_initial": distance(e["text"], initial_y),
                    "prompt_distance_from_initial": distance(x, initial_x),
                    "response_step_distance": (
                        distance(e["text"], forward[i - 1]["text"]) if i else None
                    ),
                    "prompt_step_distance": (
                        distance(x, forward[i - 1]["input_messages"][0]["content"]) if i else None
                    ),
                }
            )
    write_json(args.events_dir / "geometry.json", measurements)
    write_json(
        args.events_dir / "embedding_provenance.json",
        {
            "model": model_spec,
            "device": args.device,
            "batch_size": args.batch_size,
            "chunk_tokens": chunk_size,
            "pooling": "token-count-weighted mean of chunk embeddings, then L2 normalization",
            "n_texts": len(texts),
            "n_chunks": len(chunks),
            "interpretation": "Geometry only; does not establish retained intent, correctness, or harmfulness.",
        },
    )
    print(
        f"Embedded {len(texts)} texts in {len(chunks)} chunks; {len(measurements)} forward states."
    )


if __name__ == "__main__":
    main()
