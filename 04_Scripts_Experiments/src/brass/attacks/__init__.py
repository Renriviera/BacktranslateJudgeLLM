"""Attack transforms: ``prompt -> attacked_prompt``.

Attacks turn a behavior prompt into one or more attacked prompts that are then served through
vLLM for completion sampling. White-box optimizers (TAO-Attack, SlotGCG) run their own
gradient-based search in their vendored repos and cache the resulting adversarial prompts; the
adapters here load those cached artifacts.
"""

from brass.attacks.base import Attack, AttackResult, get_attack, register_attack
from brass.attacks.none import NoAttack
from brass.attacks.pair_adapter import PairAttack
from brass.attacks.pap_adapter import PapAttack
from brass.attacks.slotgcg_adapter import SlotGCGAttack
from brass.attacks.tao_adapter import TaoAttack

__all__ = [
    "Attack",
    "AttackResult",
    "get_attack",
    "register_attack",
    "NoAttack",
    "TaoAttack",
    "SlotGCGAttack",
    "PapAttack",
    "PairAttack",
]
