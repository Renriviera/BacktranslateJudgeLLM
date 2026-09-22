#!/usr/bin/env python
"""Rescore saved BRASS attacked completions with the StrongREJECT Gemma judge."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from omegaconf import OmegaConf

from brass.metrics.judge_asr import StrongRejectFinetunedJudge, aggregate_judge
from brass.serving.vllm_engine import ModelSpec, VLLMEngine
from brass.utils.io import read_json, write_json
from brass.utils.seeding import seed_everything

DEFAULT_RESULTS_DIRS = (
    "06_Results_Artifacts/results/tao_strongreject100_olmo3_7b",
    "06_Results_Artifacts/results/pair_strongreject_olmo3_7b",
    "06_Results_Artifacts/results/pap_logic_strongreject_olmo3_7b",
    "06_Results_Artifacts/results/pap_authority_strongreject_olmo3_7b",
    "06_Results_Artifacts/results/pap_misrep_strongreject_olmo3_7b",
)

logger = logging.getLogger("rescore_strongreject")


def _triples(details: dict[str, Any]) -> list[tuple[str, str, str]]:
    prompts = {prompt["id"]: prompt["prompt"] for prompt in details["prompts"]}
    attacked = details["completions"]["attacked_instruct"]
    return [
        (prompt_id, prompts[prompt_id], completion)
        for prompt_id, completions in attacked.items()
        for completion in completions
    ]


def _payload(results: list[Any]) -> dict[str, Any]:
    return {
        "judge": "strongreject_finetuned",
        "asr": aggregate_judge(results),
        "per_prompt": {
            result.prompt_id: {
                "asr": result.asr,
                "threshold_asr": result.threshold_asr,
                "scores": result.scores,
            }
            for result in results
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dirs",
        nargs="*",
        default=list(DEFAULT_RESULTS_DIRS),
        help="Result directories containing summary.json and details.json.",
    )
    parser.add_argument("--seed", type=int, default=235711)
    parser.add_argument(
        "--log-path",
        default="06_Results_Artifacts/results/strongreject_rescore.json",
        help="Combined output log.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override the judge's vLLM GPU-memory fraction.",
    )
    args = parser.parse_args()

    load_dotenv()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    seed_everything(args.seed, deterministic_torch=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    judge_cfg = OmegaConf.load("04_Scripts_Experiments/configs/judge/strongreject_ft.yaml")
    spec = ModelSpec(**OmegaConf.to_container(judge_cfg.model, resolve=True))
    if args.gpu_memory_utilization is not None:
        spec.gpu_memory_utilization = args.gpu_memory_utilization
    run_log: dict[str, Any] = {
        "judge": "strongreject_finetuned",
        "model": spec.hf_id,
        "adapter": spec.lora_path,
        "seed": args.seed,
        "experiments": {},
    }

    with VLLMEngine(spec, seed=args.seed) as engine:
        judge = StrongRejectFinetunedJudge(
            engine,
            threshold=float(judge_cfg.threshold),
            max_response_length=int(judge_cfg.max_response_length),
        )
        for raw_dir in args.results_dirs:
            results_dir = Path(raw_dir)
            summary_path = results_dir / "summary.json"
            details_path = results_dir / "details.json"
            summary = read_json(summary_path)
            details = read_json(details_path)
            pairs = _triples(details)
            logger.info("Scoring %s completions from %s", len(pairs), results_dir)
            judge_results = judge.score(pairs)
            payload = _payload(judge_results)

            summary.setdefault("judge_asr", {})["strongreject"] = payload["asr"]
            details.setdefault("judges", {})["strongreject"] = payload
            write_json(summary_path, summary)
            write_json(details_path, details)
            write_json(results_dir / "strongreject_rescore.json", payload)

            run_log["experiments"][summary["experiment"]] = {
                "results_dir": str(results_dir),
                "n_prompts": len(judge_results),
                "n_completions": len(pairs),
                "strongreject_asr": payload["asr"],
                "strongreject_threshold_asr": sum(
                    result.threshold_asr or 0.0 for result in judge_results
                )
                / len(judge_results),
            }
            write_json(args.log_path, run_log)
            logger.info("%s StrongREJECT ASR: %.6f", summary["experiment"], payload["asr"])

    write_json(args.log_path, run_log)
    logger.info("Wrote combined log to %s", args.log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
