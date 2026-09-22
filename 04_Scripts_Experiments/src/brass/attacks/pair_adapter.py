"""Adapter for PAIR (Prompt Automatic Iterative Refinement) attacks.

PAIR is a black-box, *iterative* jailbreak (Chao et al. 2023, arXiv:2310.08419; reference
implementation ``patrickrchao/JailbreakingLLMs``). An attacker LLM proposes an adversarial prompt,
the target answers, an LLM judge scores it 1-10, and the attacker refines over several rounds and
parallel streams, keeping the best prompt. Like the GCG/PAP adapters, the expensive search runs
offline: ``04_Scripts_Experiments/scripts/attacks/run_pair.py`` performs the loop (Qwen3-32B attacker+judge vs OLMo-3-7B
target) and writes a model-scoped cache ``06_Results_Artifacts/results/attacks/pair/<target>.json`` mapping ``prompt_id``
(and raw prompt) to a record with at least ``attacked_prompt`` (the best PAIR prompt). This adapter
loads that cache and applies it.

The cache record also carries ``pair_status`` (``valid`` / ``attacker_failed``), the in-loop judge
``pair_score`` / ``pair_jailbroken``, and ``n_queries``. These flow into ``AttackResult.metadata`` so
``04_Scripts_Experiments/scripts/attacks/pair_funnel.py`` can separate *attacker* failure (Qwen never produced a usable
adversarial prompt) from *target* robustness (OLMo resisted a real PAIR prompt).
"""

from __future__ import annotations

import logging
from pathlib import Path

from brass.attacks.base import Attack, AttackResult, register_attack
from brass.data import BehaviorPrompt
from brass.utils.io import read_json

logger = logging.getLogger(__name__)


@register_attack("pair")
class PairAttack(Attack):
    """Apply cached PAIR (best iteratively-refined) prompts.

    Args:
        cache_path: JSON file mapping ``prompt_id`` (or original prompt) to a record with at least
            an ``attacked_prompt`` field (and optionally ``pair_status`` / ``pair_score`` /
            ``pair_jailbroken`` / ``n_queries``).
        strict: If ``True``, raise when a prompt has no cached PAIR result; otherwise fall back to
            the original prompt (recorded with ``pair_status="missing"``).
    """

    def __init__(
        self,
        cache_path: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cache_path = cache_path
        self.strict = strict
        self._cache: dict[str, dict] = {}
        if cache_path and Path(cache_path).exists():
            self._cache = read_json(cache_path)
        elif cache_path:
            msg = f"PAIR cache not found at {cache_path}; run 04_Scripts_Experiments/scripts/attacks/run_pair.py first."
            if strict:
                raise FileNotFoundError(msg)
            logger.warning(msg)

    def transform(self, prompt: BehaviorPrompt) -> AttackResult:
        record = self._cache.get(prompt.id) or self._cache.get(prompt.prompt)
        if record is None:
            if self.strict:
                raise KeyError(f"No cached PAIR result for {prompt.id}")
            logger.warning("No cached PAIR result for %s; using original prompt.", prompt.id)
            return AttackResult(
                prompt_id=prompt.id,
                original_prompt=prompt.prompt,
                attacked_prompt=prompt.prompt,
                attack_name=self.name,
                metadata={"cache_hit": False, "pair_status": "missing"},
            )
        attacked = record.get("attacked_prompt", prompt.prompt)
        meta = {
            "cache_hit": True,
            **{k: v for k, v in record.items() if k != "attacked_prompt"},
        }
        return AttackResult(
            prompt_id=prompt.id,
            original_prompt=prompt.prompt,
            attacked_prompt=attacked,
            attack_name=self.name,
            metadata=meta,
        )
