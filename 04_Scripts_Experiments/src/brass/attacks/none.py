"""Identity (no-op) attack: the clean-instruct baseline used by the smoke test.

With ``a = identity`` we have ``E_{a(p)}(I) = E_p(I)``, so BRASS reduces to ``0`` by
construction (numerator equals denominator). This is exactly what validates the plumbing: the
smoke test should report BRASS near zero for the unattacked instruct model.
"""

from __future__ import annotations

from brass.attacks.base import Attack, AttackResult, register_attack
from brass.data import BehaviorPrompt


@register_attack("none")
class NoAttack(Attack):
    def transform(self, prompt: BehaviorPrompt) -> AttackResult:
        return AttackResult(
            prompt_id=prompt.id,
            original_prompt=prompt.prompt,
            attacked_prompt=prompt.prompt,
            attack_name=self.name,
            metadata={"identity": True},
        )
