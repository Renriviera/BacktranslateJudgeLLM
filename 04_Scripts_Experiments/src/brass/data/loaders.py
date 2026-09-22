"""Benchmark loaders with a unified prompt schema.

Every benchmark is mapped onto :class:`BehaviorPrompt` so the rest of the pipeline is dataset
agnostic. Loaders are registered by name and selected via Hydra (``dataset=<name>``):

- ``strongreject`` - StrongREJECT forbidden prompts (primary harmful holdout).
- ``harmbench``    - HarmBench standard text behaviors.
- ``orbench``      - OR-Bench over-refusal control (benign, safety-adjacent).
- ``xstest``       - XSTest over-refusal control (safe + unsafe contrast set).
- ``jailjudge``    - JAILJUDGE prompts for judge / refusal behavior study.

Loaders prefer the ``datasets`` library / official sources and fall back to a cached CSV download
where a stable raw URL exists. Network failures raise a clear error so callers can decide how to
proceed.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Local cache for any files we download directly (CSV sources without a datasets loader).
_CACHE_DIR = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir()) / "06_Results_Artifacts/results" / "_dataset_cache"


@dataclass(frozen=True)
class BehaviorPrompt:
    """A single behavior probe, normalized across benchmarks.

    Attributes:
        id: Stable identifier (``"{source}:{index_or_native_id}"``).
        prompt: The forbidden / probe instruction shown to the model.
        source: Originating benchmark name.
        category: Optional semantic / functional category.
        context: Optional context string (HarmBench contextual behaviors).
        is_harmful: ``True`` for harmful probes, ``False`` for benign over-refusal controls,
            ``None`` if unknown / mixed.
        metadata: Any extra native fields, kept for auditability.
    """

    id: str
    prompt: str
    source: str
    category: str | None = None
    context: str | None = None
    is_harmful: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


LoaderFn = Callable[..., list[BehaviorPrompt]]
_REGISTRY: dict[str, LoaderFn] = {}


def register_loader(name: str) -> Callable[[LoaderFn], LoaderFn]:
    """Decorator registering a loader under ``name``."""

    def deco(fn: LoaderFn) -> LoaderFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def load_dataset_prompts(name: str, **kwargs: Any) -> list[BehaviorPrompt]:
    """Load a benchmark by registered ``name``.

    Extra kwargs are forwarded to the specific loader (e.g. ``variant`` for OR-Bench).
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def _download_csv(url: str, cache_name: str) -> list[dict[str, str]]:
    """Download (and cache) a CSV, returning a list of row dicts."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / cache_name
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        logger.info("Downloading %s -> %s", url, cache_path)
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (trusted URL)
            text = resp.read().decode("utf-8")
        cache_path.write_text(text, encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _try_load_hf(dataset_id: str, split: str, **kwargs: Any):
    """Load a HF dataset split, raising a clear error if ``datasets`` is missing."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The 'datasets' package is required for this loader.") from exc
    return load_dataset(dataset_id, split=split, **kwargs)


# --------------------------------------------------------------------------------------------- #
# StrongREJECT                                                                                    #
# --------------------------------------------------------------------------------------------- #
_STRONGREJECT_FULL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)
_STRONGREJECT_SMALL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_small_dataset.csv"
)


@register_loader("strongreject")
def load_strongreject(
    *,
    small: bool = False,
    ids_path: str | None = None,
    **_: Any,
) -> list[BehaviorPrompt]:
    """Load StrongREJECT forbidden prompts.

    Tries the ``strong_reject`` package first, then falls back to the canonical CSV on GitHub.
    """
    # Preferred: official package (handles versioning + dedup).
    try:
        from strong_reject.load_datasets import (  # type: ignore
            load_strongreject as _pkg_load,
        )
        from strong_reject.load_datasets import (
            load_strongreject_small as _pkg_load_small,
        )

        ds = _pkg_load_small() if small else _pkg_load()
        rows = [dict(r) for r in ds]
    except Exception:  # noqa: BLE001 - fall back to raw CSV
        url = _STRONGREJECT_SMALL if small else _STRONGREJECT_FULL
        rows = _download_csv(url, f"strongreject_{'small' if small else 'full'}.csv")

    prompts: list[BehaviorPrompt] = []
    for i, r in enumerate(rows):
        text = r.get("forbidden_prompt") or r.get("prompt") or ""
        if not text.strip():
            continue
        prompts.append(
            BehaviorPrompt(
                id=f"strongreject:{i}",
                prompt=text.strip(),
                source="strongreject",
                category=r.get("category"),
                is_harmful=True,
                metadata={"origin": r.get("source")},
            )
        )
    if ids_path:
        ids_obj = json.loads(Path(ids_path).read_text(encoding="utf-8"))
        if isinstance(ids_obj, dict):
            ids = [k for k in ids_obj if isinstance(k, str) and k.startswith("strongreject:")]
        else:
            ids = [str(x) for x in ids_obj]
        ids_set = set(ids)
        prompts = [p for p in prompts if p.id in ids_set]
        # Preserve the file/cache order when possible so runs are reproducible across the attack
        # cache and the evaluator.
        rank = {pid: i for i, pid in enumerate(ids)}
        prompts.sort(key=lambda p: rank.get(p.id, len(rank)))
    return prompts


# --------------------------------------------------------------------------------------------- #
# HarmBench                                                                                       #
# --------------------------------------------------------------------------------------------- #
_HARMBENCH_BEHAVIORS = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "01_Datasets_Benchmarks/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)


@register_loader("harmbench")
def load_harmbench(
    *, functional_category: str | None = "standard", **_: Any
) -> list[BehaviorPrompt]:
    """Load HarmBench text behaviors (defaults to standard behaviors)."""
    rows = _download_csv(_HARMBENCH_BEHAVIORS, "harmbench_behaviors_text_all.csv")
    prompts: list[BehaviorPrompt] = []
    for r in rows:
        if functional_category and r.get("FunctionalCategory", "").strip() != functional_category:
            continue
        text = (r.get("Behavior") or "").strip()
        if not text:
            continue
        ctx = (r.get("ContextString") or "").strip() or None
        prompts.append(
            BehaviorPrompt(
                id=f"harmbench:{r.get('BehaviorID', len(prompts))}",
                prompt=text,
                source="harmbench",
                category=r.get("SemanticCategory"),
                context=ctx,
                is_harmful=True,
                metadata={
                    "functional_category": r.get("FunctionalCategory"),
                    "tags": r.get("Tags"),
                },
            )
        )
    return prompts


# --------------------------------------------------------------------------------------------- #
# OR-Bench (over-refusal control: benign prompts that look unsafe)                                #
# --------------------------------------------------------------------------------------------- #
@register_loader("orbench")
def load_orbench(*, variant: str = "or-bench-hard-1k", **_: Any) -> list[BehaviorPrompt]:
    """Load OR-Bench. ``variant`` in {or-bench-80k, or-bench-hard-1k, or-bench-toxic}."""
    ds = _try_load_hf("bench-llm/or-bench", split="train", name=variant)
    prompts: list[BehaviorPrompt] = []
    for i, r in enumerate(ds):
        text = (r.get("prompt") or "").strip()
        if not text:
            continue
        toxic = variant == "or-bench-toxic"
        prompts.append(
            BehaviorPrompt(
                id=f"orbench:{variant}:{i}",
                prompt=text,
                source="orbench",
                category=r.get("category"),
                is_harmful=toxic,
                metadata={"variant": variant},
            )
        )
    return prompts


# --------------------------------------------------------------------------------------------- #
# XSTest (over-refusal control: safe + contrasting unsafe prompts)                                #
# --------------------------------------------------------------------------------------------- #
@register_loader("xstest")
def load_xstest(**_: Any) -> list[BehaviorPrompt]:
    """Load XSTest (safe vs unsafe contrast prompts)."""
    ds = _try_load_hf("walledai/XSTest", split="test")
    prompts: list[BehaviorPrompt] = []
    for i, r in enumerate(ds):
        text = (r.get("prompt") or "").strip()
        if not text:
            continue
        label = (r.get("label") or "").lower()
        prompts.append(
            BehaviorPrompt(
                id=f"xstest:{i}",
                prompt=text,
                source="xstest",
                category=r.get("type"),
                is_harmful=(label == "unsafe"),
                metadata={"focus": r.get("focus"), "label": label},
            )
        )
    return prompts


# --------------------------------------------------------------------------------------------- #
# JAILJUDGE                                                                                       #
# --------------------------------------------------------------------------------------------- #
@register_loader("jailjudge")
def load_jailjudge(*, split: str = "train", **_: Any) -> list[BehaviorPrompt]:
    """Load JAILJUDGE prompts (used to study judge / refusal evaluation behavior)."""
    ds = _try_load_hf("usail-hkust/JAILJUDGE", split=split)
    prompts: list[BehaviorPrompt] = []
    for i, r in enumerate(ds):
        text = (r.get("prompt") or r.get("question") or "").strip()
        if not text:
            continue
        prompts.append(
            BehaviorPrompt(
                id=f"jailjudge:{i}",
                prompt=text,
                source="jailjudge",
                category=r.get("category"),
                metadata={k: v for k, v in r.items() if k not in {"prompt", "question"}},
            )
        )
    return prompts
