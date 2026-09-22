#!/usr/bin/env python
"""Run a frozen manifest on the preflight-pinned local OLMo checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.runner import OrbitConfig, VLLMBackend, run_orbits  # noqa: E402


def main():
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = OrbitConfig(**json.loads(args.config.read_text()))
    manifest = json.loads(args.manifest.read_text())
    runtime = json.loads((args.run_dir / "preflight.json").read_text())
    model = runtime["models"]["target"]
    from brass.serving.vllm_engine import ModelSpec, VLLMEngine

    spec = ModelSpec(
        name="orbit_olmo3_7b",
        hf_id=model["snapshot"],
        is_chat=True,
        dtype="bfloat16",
        max_model_len=config.max_model_len,
        gpu_memory_utilization=0.85,
        extra={"enforce_eager": True, "max_num_seqs": 128},
    )
    fingerprint = {"model": model, "versions": runtime["versions"], "dtype": "bfloat16"}
    with VLLMEngine(spec, seed=config.seed) as engine:
        backend = VLLMBackend(engine, fingerprint, config.max_model_len)
        print(
            json.dumps(
                run_orbits(
                    manifest, config, backend, args.output, args.run_dir / "token_budget.jsonl"
                )
            )
        )


if __name__ == "__main__":
    main()
