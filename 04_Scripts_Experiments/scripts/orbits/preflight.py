#!/usr/bin/env python
"""Record GPU/runtime/model provenance without exposing credentials."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.orbits.io import file_digest, write_json  # noqa: E402

MODELS = {
    "target": "allenai/Olmo-3-7B-Instruct",
    "strongreject": "google/gemma-2b",
    "strongreject_adapter": "qylu4156/strongreject-15k-v1",
    "harmbench": "cais/HarmBench-Llama-2-13b-cls",
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
}


def main():
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    models = {}
    for key, repo in MODELS.items():
        root = cache / ("models--" + repo.replace("/", "--"))
        ref = root / "refs/main"
        revision = ref.read_text().strip() if ref.exists() else None
        snapshot = root / "snapshots" / (revision or "missing")
        files = {p.name: file_digest(p) for p in snapshot.glob("*.json") if p.is_file()}
        models[key] = {
            "hf_id": repo,
            "revision": revision,
            "snapshot": str(snapshot),
            "available": snapshot.exists(),
            "metadata_hashes": files,
        }
    versions = {}
    for name in ["vllm", "transformers", "datasets", "huggingface_hub", "torch", "numpy", "scipy"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(models["target"]["snapshot"], local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Orbit preflight."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "versions": versions,
        "models": models,
        "gpu": gpu.stdout.strip(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "disk_free_bytes": shutil.disk_usage(
            args.output.parent if args.output.parent.exists() else REPO
        ).free,
        "effective_forward_template_example": rendered,
        "historical_revision_verified": False,
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {k: report[k] for k in ["versions", "gpu", "cuda_available", "disk_free_bytes"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
