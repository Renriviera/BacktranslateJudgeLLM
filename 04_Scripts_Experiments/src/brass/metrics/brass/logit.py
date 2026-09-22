"""Logit-recovery BRASS distance (stub for a later phase).

Compares next-token distributions ``P_M(. | p)``, ``P_I(. | p)``, and ``P_I(. | a(p))`` with
KL-style or Jensen-Shannon divergences. Requires per-position vocabulary distributions from each
model; with vLLM these come from ``prompt_logprobs`` / capped ``logprobs``, so a faithful
implementation needs an engine configured for wide logprob output. Interface is fixed here; the
numerical implementation lands in the logit-recovery phase.
"""

from __future__ import annotations

from typing import Any

from brass.metrics.brass.base import BrassScore


def compute_logit_brass(*args: Any, **kwargs: Any) -> BrassScore:
    """Not yet implemented. See module docstring for the intended contract."""
    raise NotImplementedError(
        "Logit-recovery BRASS is stubbed. Implement next-token KL/JS over P_M, P_I, P_I(a(p))."
    )
