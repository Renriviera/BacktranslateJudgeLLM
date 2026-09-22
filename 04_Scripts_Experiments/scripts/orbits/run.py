#!/usr/bin/env python
"""Run a frozen manifest on the preflight-pinned local OLMo checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.orbits.io import file_digest, write_json  # noqa: E402
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
    # Preserve implementation provenance for this invocation without copying private config.
    source_files = [*(REPO / "04_Scripts_Experiments/src/brass/orbits").glob("*.py"), Path(__file__).resolve()]
    source_hashes = {str(p.relative_to(REPO)): file_digest(p) for p in source_files}
    for p in source_files:
        target = args.output / "source_snapshot" / p.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(p, target)
    if not (args.output / "source_snapshot/hashes.json").exists():
        write_json(args.output / "source_snapshot/hashes.json", source_hashes)
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
