"""Deflection / dummy-response penalty (stub for a later phase).

Detects cases where non-refusal has moved into toy examples, vague educational language, benign
substitutions, or technical-looking nonsense rather than base-like task behavior (cf. the BID2
abliteration case study in ``main.tex``). A first implementation can reuse the transparent,
auditable regex-severity heuristic from the BID2 experiment; the interface is fixed here.
"""

from __future__ import annotations

from typing import Any

from brass.metrics.brass.base import BrassScore


def compute_deflection_brass(*args: Any, **kwargs: Any) -> BrassScore:
    """Not yet implemented. See module docstring for the intended contract."""
    raise NotImplementedError(
        "Deflection/dummy-response penalty is stubbed. Implement the auditable severity heuristic."
    )
