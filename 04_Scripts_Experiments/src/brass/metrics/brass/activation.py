"""Activation-recovery BRASS distance (stub for a later phase).

Compares hidden states / residual-stream representations between base, clean instruct, and
attacked instruct models on the same prompt. Requires hidden-state capture (HF
``output_hidden_states`` or vLLM hooks), so this is deferred; the interface is fixed here.
"""

from __future__ import annotations

from typing import Any

from brass.metrics.brass.base import BrassScore


def compute_activation_brass(*args: Any, **kwargs: Any) -> BrassScore:
    """Not yet implemented. See module docstring for the intended contract."""
    raise NotImplementedError(
        "Activation-recovery BRASS is stubbed. Implement residual-stream distance across models."
    )
