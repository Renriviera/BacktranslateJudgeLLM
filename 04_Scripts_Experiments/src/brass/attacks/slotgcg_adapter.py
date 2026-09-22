"""Adapter for SlotGCG (https://github.com/youai058/SlotGCG).

SlotGCG generalizes GCG beyond suffix-only attacks: a white-box preprocessing pass scores
insertion *slots* inside the behavior (via attention aggregated over upper layers), allocates the
adversarial-token budget across high-scoring slots, then runs GCG over the interleaved tokens. The
repo is structured like HarmBench (``baselines/{gcg,attngcg,...}``, ``check_refusal_utils.py``).

Because tokens are interleaved at chosen positions (not just appended), the cache stores the fully
materialized ``attacked_prompt`` per behavior rather than a single suffix. This adapter loads that
cache; running the optimizer (gradient-based, HF transformers) is a separate step.
"""

from __future__ import annotations

import logging
from pathlib import Path

from brass.attacks.base import Attack, AttackResult, register_attack
from brass.data import BehaviorPrompt
from brass.utils.io import read_json

logger = logging.getLogger(__name__)


@register_attack("slotgcg")
class SlotGCGAttack(Attack):
    """Apply cached SlotGCG attacked prompts.

    Args:
        cache_path: JSON file mapping ``prompt_id`` (or original prompt) to a record containing the
            materialized ``attacked_prompt`` (and optionally slot/loss metadata).
        strict: If ``True``, raise on a cache miss; otherwise fall back to the original prompt.
    """

    def __init__(self, cache_path: str | None = None, strict: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cache_path = cache_path
        self.strict = strict
        self._cache: dict[str, dict] = {}
        if cache_path and Path(cache_path).exists():
            self._cache = read_json(cache_path)
        elif cache_path:
            msg = f"SlotGCG cache not found at {cache_path}; run the optimizer first."
            if strict:
                raise FileNotFoundError(msg)
            logger.warning(msg)

    def transform(self, prompt: BehaviorPrompt) -> AttackResult:
        record = self._cache.get(prompt.id) or self._cache.get(prompt.prompt)
        if record is None:
            if self.strict:
                raise KeyError(f"No cached SlotGCG attack for {prompt.id}")
            logger.warning("No cached SlotGCG attack for %s; using original prompt.", prompt.id)
            return AttackResult(
                prompt_id=prompt.id,
                original_prompt=prompt.prompt,
                attacked_prompt=prompt.prompt,
                attack_name=self.name,
                metadata={"cache_hit": False},
            )
        attacked = record.get("attacked_prompt", prompt.prompt)
        return AttackResult(
            prompt_id=prompt.id,
            original_prompt=prompt.prompt,
            attacked_prompt=attacked,
            attack_name=self.name,
            metadata={
                "cache_hit": True,
                **{k: v for k, v in record.items() if k != "attacked_prompt"},
            },
        )
