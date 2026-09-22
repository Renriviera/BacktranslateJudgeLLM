"""Shared checkout and model-cache locations; no user-specific filesystem paths."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
DATA = ROOT / "01_Datasets_Benchmarks/data"
RESULTS = ROOT / "06_Results_Artifacts/results"
NEW_RUNS = ROOT / "06_Results_Artifacts/new_runs"
CONFIGS = ROOT / "04_Scripts_Experiments/configs"


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"))).expanduser()


def hf_hub_cache() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE", str(hf_home() / "hub"))).expanduser()


def model_snapshot(repo: str, revision: str | None = None) -> Path:
    directory = hf_hub_cache() / ("models--" + repo.replace("/", "--"))
    if revision is None:
        reference = directory / "refs/main"
        if not reference.is_file():
            raise FileNotFoundError(f"Model is not cached: {repo}. Download it or configure HF_HOME/HF_HUB_CACHE.")
        revision = reference.read_text().strip()
    snapshot = directory / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned model snapshot is not cached: {repo}@{revision}")
    return snapshot


def recorded_path(value: str | Path) -> Path:
    """Resolve historical BRASS paths without changing the archived record or its hash.

    Prefer this checkout even if the old checkout still exists. Unknown absolute
    paths are returned unchanged; they are not guessed or silently redirected.
    """
    value = str(value)
    original = Path(value)
    if original.is_absolute() and original.is_relative_to(ROOT):
        return original
    prefixes = {
        "results/": "06_Results_Artifacts/results/",
        "data/": "01_Datasets_Benchmarks/data/",
        "src/": "04_Scripts_Experiments/src/",
        "scripts/": "04_Scripts_Experiments/scripts/",
        "configs/": "04_Scripts_Experiments/configs/",
        "docs/": "04_Scripts_Experiments/docs/",
    }
    if "/BRASS/" in value:
        value = value.split("/BRASS/", 1)[1]
    for old, new in prefixes.items():
        if value.startswith(old):
            return ROOT / (new + value[len(old):])
    return original if original.is_absolute() else ROOT / original
