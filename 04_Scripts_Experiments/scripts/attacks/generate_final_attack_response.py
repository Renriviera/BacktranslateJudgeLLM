#!/usr/bin/env python3
"""Generate and save the target-model response for one finished attack prompt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--seed", type=int, default=235711)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()
    data = json.loads(args.result.read_text())
    if data.get("target_response"):
        print(f"Target response already present in {args.result}")
        return
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9, trust_remote_code=True, seed=args.seed)
    response = llm.chat(
        [{"role": "user", "content": data["attacked_prompt"]}],
        SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed),
        use_tqdm=False,
    )[0]
    data["target_response"] = response.outputs[0].text
    data["target_finish_reason"] = response.outputs[0].finish_reason
    data["target_generation"] = {"model": args.model, "seed": args.seed, "temperature": 1.0,
                                 "top_p": 1.0, "max_tokens": args.max_tokens}
    args.result.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved final target response in {args.result}")


if __name__ == "__main__":
    main()
