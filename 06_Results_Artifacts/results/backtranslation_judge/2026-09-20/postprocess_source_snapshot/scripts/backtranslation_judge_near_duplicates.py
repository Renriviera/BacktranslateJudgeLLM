#!/usr/bin/env python3
"""Reproduce response-independent semantic screening before freezing judge-study splits."""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    import torch
    from sentence_transformers import SentenceTransformer

    repo = Path(__file__).resolve().parents[1]
    root = Path("/mnt/data/hf_cache/hub/models--sentence-transformers--all-MiniLM-L6-v2")
    revision = (root / "refs/main").read_text().strip()
    torch.set_num_threads(4)
    model = SentenceTransformer(
        str(root / "snapshots" / revision), device="cpu", local_files_only=True
    )
    prompts = json.loads((repo / "results/pair_strongreject_olmo3_7b/details.json").read_text())[
        "prompts"
    ]
    vectors = model.encode(
        [p["prompt"] for p in prompts], normalize_embeddings=True, show_progress_bar=False
    )
    similarities = vectors @ vectors.T
    candidates = []
    maximum = -1.0
    nearest = None
    for i in range(len(prompts)):
        for j in range(i):
            value = float(similarities[i, j])
            if value > maximum:
                maximum, nearest = value, [prompts[i]["id"], prompts[j]["id"]]
            if value >= 0.8:
                candidates.append(
                    dict(
                        a=prompts[i]["id"],
                        b=prompts[j]["id"],
                        cosine=round(value, 4),
                        prompt_a=prompts[i]["prompt"],
                        prompt_b=prompts[j]["prompt"],
                    )
                )
    out = repo / "results/backtranslation_judge"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "near_duplicate_candidates.json"
    text = json.dumps(sorted(candidates, key=lambda r: -r["cosine"]), indent=2) + "\n"
    if target.exists() and json.loads(target.read_text()) != json.loads(text):
        raise RuntimeError("Existing candidate screen differs; do not mutate frozen study inputs")
    target.write_text(text)
    metadata = dict(
        model="sentence-transformers/all-MiniLM-L6-v2",
        revision=revision,
        candidate_threshold=0.8,
        candidates=len(candidates),
        nearest_pair=nearest,
        maximum_cosine=maximum,
        note="Automated similarity screen, not human proof of semantic independence",
    )
    (out / "near_duplicate_screen_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
