#!/usr/bin/env python
"""Download the models BRASS needs into the local HF cache (``models/`` by default).

Models:
  - allenai/Olmo-3-1125-32B          (base M)
  - allenai/Olmo-3.1-32B-Instruct    (instruct I)
  - allenai/Olmo-3.1-32B-Instruct-DPO (DPO)
  - lmsys/vicuna-7b-v1.5             (TAO paper target)
  - meta-llama/Llama-2-7b-chat-hf    (TAO paper target; gated)
  - qylu4156/strongreject-15k-v1     (StrongREJECT finetuned judge)
  - cais/HarmBench-Llama-2-13b-cls   (HarmBench judge)
  - sentence-transformers/all-MiniLM-L6-v2 (completion embedder)

Usage:
  python 04_Scripts_Experiments/scripts/download_models.py                 # all of the above
  python 04_Scripts_Experiments/scripts/download_models.py --only base instruct
  python 04_Scripts_Experiments/scripts/download_models.py --list

Set HF_TOKEN in the environment / .env for gated repos (e.g. google/gemma-2b, the base for the
StrongREJECT judge). Honors HF_HOME (defaults to ./models).
"""

from __future__ import annotations

import argparse
import os
import sys

MODELS: dict[str, str] = {
    "qwen3_32b": "Qwen/Qwen3-32B",
    "base": "allenai/Olmo-3-1125-32B",
    "instruct": "allenai/Olmo-3.1-32B-Instruct",
    "dpo": "allenai/Olmo-3.1-32B-Instruct-DPO",
    # 7B counterparts (faster attack iteration; note 7B instruct is "Olmo-3", no "3.1" 7B exists).
    "base_7b": "allenai/Olmo-3-1025-7B",
    "instruct_7b": "allenai/Olmo-3-7B-Instruct",
    "dpo_7b": "allenai/Olmo-3-7B-Instruct-DPO",
    # TAO-Attack AdvBench reproduction targets.
    "tao_vicuna_7b_v1_5": "lmsys/vicuna-7b-v1.5",
    "tao_llama2_7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    # StrongREJECT judge is a LoRA adapter over the gated google/gemma-2b base; fetch both.
    "judge_strongreject_base": "google/gemma-2b",  # gated: needs HF_TOKEN + accepted license
    "judge_strongreject": "qylu4156/strongreject-15k-v1",
    "judge_harmbench": "cais/HarmBench-Llama-2-13b-cls",
    "embedder": "sentence-transformers/all-MiniLM-L6-v2",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=list(MODELS), help="Subset of keys to fetch.")
    parser.add_argument("--list", action="store_true", help="List models and exit.")
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("HF_HOME", str(__import__("pathlib").Path.home() / ".cache/huggingface")),
        help="HF cache dir (default: HF_HOME or ./models).",
    )
    args = parser.parse_args()

    if args.list:
        for k, v in MODELS.items():
            print(f"{k:20s} {v}")
        return 0

    os.environ.setdefault("HF_HOME", args.models_dir)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    keys = args.only or list(MODELS)
    failed = []
    for key in keys:
        repo = MODELS[key]
        print(f"==> Downloading {key}: {repo}")
        try:
            path = snapshot_download(repo_id=repo, token=token)
            print(f"    done: {path}")
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if repo.startswith("meta-llama/") and any(
                marker in message.lower()
                for marker in ("gated", "401", "403", "unauthorized", "forbidden")
            ):
                message = (
                    "Hugging Face access denied. Accept the model license and set HF_TOKEN "
                    "to an account with access."
                )
            print(f"    FAILED: {message}", file=sys.stderr)
            failed.append((key, repo, message))

    if failed:
        print("\nSome downloads failed:", file=sys.stderr)
        for key, repo, err in failed:
            print(f"  - {key} ({repo}): {err}", file=sys.stderr)
        return 1
    print("\nAll requested models downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
