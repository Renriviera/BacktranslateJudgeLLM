"""Attack interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from brass.data import BehaviorPrompt


@dataclass
class AttackResult:
    """Result of attacking one prompt.

    Attributes:
        prompt_id: Source :class:`~brass.data.BehaviorPrompt` id.
        original_prompt: The unmodified probe.
        attacked_prompt: The prompt actually sent to the target model.
        attack_name: Name of the attack that produced it.
        metadata: Optimizer details (suffix, slots, iterations, loss, ...).
    """

    prompt_id: str
    original_prompt: str
    attacked_prompt: str
    attack_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Attack(ABC):
    """Base class for prompt-transforming attacks."""

    name: str = "attack"

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    @abstractmethod
    def transform(self, prompt: BehaviorPrompt) -> AttackResult:
        """Return the attacked version of ``prompt``."""

    def transform_many(self, prompts: list[BehaviorPrompt]) -> list[AttackResult]:
        return [self.transform(p) for p in prompts]


_REGISTRY: dict[str, type[Attack]] = {}


def register_attack(name: str) -> Callable[[type[Attack]], type[Attack]]:
    def deco(cls: type[Attack]) -> type[Attack]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_attack(name: str, **kwargs: Any) -> Attack:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown attack '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
