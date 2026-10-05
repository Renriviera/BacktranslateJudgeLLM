#!/usr/bin/env python3
"""Regenerate target-only outputs missing after PAP semantic tasks crashed post-generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv


def resolve_run_dir(base_dir: Path, run_id: str) -> Path:
    candidates = [base_dir / f"pap-{run_id}"]
    candidates.extend(
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / f"pap_strongreject_0_{run_id}.report.json").exists()
    )
    for candidate in candidates:
        if (candidate / f"pap_strongreject_0_{run_id}.report.json").exists():
            return candidate
    return base_dir


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    load_dotenv(repo / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    from brass.serving.vllm_engine import ModelSpec, VLLMEngine
    from vllm import SamplingParams

    base_dir = repo / "06_Results_Artifacts/new_runs/semantic_reconstruction"
    out_dir = resolve_run_dir(base_dir, args.run_id)
    missing: list[tuple[int, dict, dict]] = []
    for index in range(313):
        report_path = out_dir / f"pap_strongreject_{index}_{args.run_id}.report.json"
        semantic_path = out_dir / f"pap_semantic_{index}_{args.run_id}.json"
        if semantic_path.exists():
            continue
        report = json.loads(report_path.read_text())
        row = report["per_behavior"][0]
        if row.get("pap_status") == "valid":
            missing.append((index, report, row))
    if len(missing) != 12:
        raise SystemExit(f"Expected 12 valid PAP cases without semantic output; found {len(missing)}")

    spec = ModelSpec(
        name="olmo3_7b_instruct",
        hf_id="allenai/Olmo-3-7B-Instruct",
        is_chat=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    with VLLMEngine(spec, seed=235711) as engine:
        for index, report, row in missing:
            if not row.get("attacked_prompt"):
                raise SystemExit(f"Missing PAP-mutated prompt at index {index}")
            seed = 235711 + index + 1
            outputs = engine.llm.chat(
                [{"role": "user", "content": row["attacked_prompt"]}],
                SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=args.max_tokens,
                    seed=seed,
                ),
                use_tqdm=False,
            )
            if not outputs or not outputs[0].outputs:
                raise RuntimeError(f"No target completion for StrongREJECT index {index}")
            completion = outputs[0].outputs[0]
            result = {
                "run_id": args.run_id,
                "dataset_index": index,
                "behavior_id": row.get("id"),
                "original_prompt": row.get("behavior"),
                "attacked_prompt": row["attacked_prompt"],
                "target_response": completion.text,
                "target_finish_reason": completion.finish_reason,
                "target_model": "allenai/Olmo-3-7B-Instruct",
                "target_generation_config": {
                    "seed": seed,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": args.max_tokens,
                    "max_model_len": args.max_model_len,
                },
                "recovery_note": (
                    "Regenerated for audit with the original task seed/settings because the original "
                    "pilot crashed after generation and did not persist its response. This response "
                    "was not reclassified or pairwise-scored."
                ),
            }
            out_path = out_dir / f"pap_target_recovery_{index}_{args.run_id}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"Recovered target response for StrongREJECT index {index} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
