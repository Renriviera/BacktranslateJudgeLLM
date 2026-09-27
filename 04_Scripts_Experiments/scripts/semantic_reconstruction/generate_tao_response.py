#!/usr/bin/env python3
"""Generate one OLMo response for a cached TAO prompt before classifier evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    load_dotenv(repo / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    prompt = args.prompt_file.read_text().strip()
    if not prompt:
        parser.error("prompt file is empty")
    if os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HF_TOKEN"]

    from brass.serving.vllm_engine import ModelSpec, VLLMEngine
    from vllm import SamplingParams

    spec = ModelSpec(
        name="olmo3_7b_instruct",
        hf_id=args.model,
        is_chat=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    with VLLMEngine(spec, seed=args.seed) as engine:
        outputs = engine.llm.chat(
            [{"role": "user", "content": prompt}],
            SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=args.max_tokens,
                seed=args.seed,
            ),
            use_tqdm=False,
        )
    if not outputs or not outputs[0].outputs:
        raise RuntimeError("Target model returned no completion")
    completion = outputs[0].outputs[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(completion.text)
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "seed": args.seed,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": args.max_tokens,
                "finish_reason": completion.finish_reason,
                "source": "TAO optimized prompt, fresh target generation",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
