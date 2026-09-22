"""Adapter for PAP (Persuasive Adversarial Prompt) attacks.

PAP is a black-box, single-shot *prompt-rewriting* attack (Zeng et al. 2024, arXiv:2401.06373;
reference implementation in ``dsbowen/strong_reject``). An attacker LLM mutates a forbidden prompt
using an in-context persuasion template (here, the "Misrepresentation" technique) while preserving
the original harmful intent. Unlike the GCG-family optimizers (TAO/SlotGCG), there is no gradient
search: the "attack" is one attacker-model call per behavior.

Integration contract (matches ``tao_adapter`` / ``slotgcg_adapter``): ``04_Scripts_Experiments/scripts/attacks/run_pap.py``
generates the PAPs with the attacker model and writes a model-scoped cache
``06_Results_Artifacts/results/attacks/pap/<target>.json`` mapping ``prompt_id`` (and raw prompt) to a record with at
least ``attacked_prompt``. This adapter loads that cache and applies it.

The cache record also carries the attacker-side funnel label ``pap_status`` in
``{valid, distorted, refused}`` plus the extracted ``core_intention``. These flow through to
``AttackResult.metadata`` so downstream analysis (``04_Scripts_Experiments/scripts/attacks/pap_funnel.py``) can separate
*attacker* failure (Qwen declined / distorted the paraphrase) from *target* robustness (OLMo
resisted a valid persuasive prompt) -- the attempted -> valid -> target-success funnel.
"""

from __future__ import annotations

import logging
from pathlib import Path

from brass.attacks.base import Attack, AttackResult, register_attack
from brass.data import BehaviorPrompt
from brass.utils.io import read_json

logger = logging.getLogger(__name__)


@register_attack("pap")
class PapAttack(Attack):
    """Apply cached PAP (persuasion-mutated) prompts.

    Args:
        cache_path: JSON file mapping ``prompt_id`` (or original prompt) to a record with at least
            an ``attacked_prompt`` field (and optionally ``pap_status`` / ``core_intention``).
        strict: If ``True``, raise when a prompt has no cached PAP; otherwise fall back to the
            original prompt (recorded with ``pap_status="missing"``).
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
            msg = f"PAP cache not found at {cache_path}; run 04_Scripts_Experiments/scripts/attacks/run_pap.py first."
            if strict:
                raise FileNotFoundError(msg)
            logger.warning(msg)

    def transform(self, prompt: BehaviorPrompt) -> AttackResult:
        record = self._cache.get(prompt.id) or self._cache.get(prompt.prompt)
        if record is None:
            if self.strict:
                raise KeyError(f"No cached PAP for {prompt.id}")
            logger.warning("No cached PAP for %s; using original prompt.", prompt.id)
            return AttackResult(
                prompt_id=prompt.id,
                original_prompt=prompt.prompt,
                attacked_prompt=prompt.prompt,
                attack_name=self.name,
                metadata={"cache_hit": False, "pap_status": "missing"},
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
