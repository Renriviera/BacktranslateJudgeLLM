"""Adapter for TAO-Attack (https://github.com/ZevineXu/TAO-Attack).

TAO-Attack is a GCG-family white-box suffix optimizer (a fork of the ``llm-attacks`` codebase;
entry point ``experiments/main.py``, AdvBench data under ``01_Datasets_Benchmarks/data/advbench``). It runs its own
gradient-based optimization against the target model to produce an adversarial suffix per
behavior.

Integration contract: the optimizer writes a cache of ``{prompt_id -> {suffix, attacked_prompt,
metadata}}`` (JSON) under ``06_Results_Artifacts/results/attacks/tao/<target>.json``; this adapter loads that cache and
applies the suffix. Running the optimizer itself is a separate step (see
``04_Scripts_Experiments/scripts/`` / repo README) because it needs gradients via HF transformers, not vLLM.
"""

from __future__ import annotations

import logging
from pathlib import Path

from brass.attacks.base import Attack, AttackResult, register_attack
from brass.data import BehaviorPrompt
from brass.utils.io import read_json

logger = logging.getLogger(__name__)


@register_attack("tao")
class TaoAttack(Attack):
    """Apply cached TAO-Attack adversarial suffixes.

    Args:
        cache_path: JSON file mapping ``prompt_id`` (or original prompt) to a record with at least
            an ``attacked_prompt`` or ``suffix`` field.
        suffix_join: String inserted between prompt and suffix when only a suffix is cached.
        strict: If ``True``, raise when a prompt has no cached attack; otherwise fall back to the
            original prompt.
    """

    def __init__(
        self,
        cache_path: str | None = None,
        suffix_join: str = " ",
        strict: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cache_path = cache_path
        self.suffix_join = suffix_join
        self.strict = strict
        self._cache: dict[str, dict] = {}
        if cache_path and Path(cache_path).exists():
            self._cache = read_json(cache_path)
        elif cache_path:
            msg = f"TAO cache not found at {cache_path}; run the optimizer first."
            if strict:
                raise FileNotFoundError(msg)
            logger.warning(msg)

    def transform(self, prompt: BehaviorPrompt) -> AttackResult:
        record = self._cache.get(prompt.id) or self._cache.get(prompt.prompt)
        if record is None:
            if self.strict:
                raise KeyError(f"No cached TAO attack for {prompt.id}")
            logger.warning("No cached TAO attack for %s; using original prompt.", prompt.id)
            attacked = prompt.prompt
            meta = {"cache_hit": False}
        elif "attacked_prompt" in record:
            attacked = record["attacked_prompt"]
            meta = {
                "cache_hit": True,
                **{k: v for k, v in record.items() if k != "attacked_prompt"},
            }
        else:
            attacked = f"{prompt.prompt}{self.suffix_join}{record.get('suffix', '')}"
            meta = {"cache_hit": True, **record}
        return AttackResult(
            prompt_id=prompt.id,
            original_prompt=prompt.prompt,
            attacked_prompt=attacked,
            attack_name=self.name,
            metadata=meta,
        )
