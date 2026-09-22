"""Composite BRASS distance (Eq. (3)).

Combines individual distance signals into a single score:

    d_BRASS = lambda_1 d_emb + lambda_2 d_likelihood + lambda_3 d_logit
              + lambda_4 d_act + lambda_5 d_deflection

Per the proposal, individual variants are reported *before* any composite so failure modes stay
auditable. This helper combines already-computed :class:`BrassScore` distances; it does not hide
the components.
"""

from __future__ import annotations

from collections.abc import Mapping

from brass.metrics.brass.base import BrassScore, brass_from_distances

# Default weights (lambda_1..5). Logit / activation / deflection default to 0 until implemented.
DEFAULT_WEIGHTS: dict[str, float] = {
    "embedding": 1.0,
    "likelihood": 1.0,
    "logit": 0.0,
    "activation": 0.0,
    "deflection": 0.0,
}


def composite_brass(
    components: Mapping[str, BrassScore],
    *,
    weights: Mapping[str, float] | None = None,
    eps: float = 1e-8,
) -> BrassScore:
    """Weighted-sum composite over component BRASS distances.

    Args:
        components: Mapping ``family -> BrassScore`` (keys match :data:`DEFAULT_WEIGHTS`).
        weights: Optional weight overrides; missing families use :data:`DEFAULT_WEIGHTS`.
        eps: Denominator stabilizer.

    Returns:
        A composite :class:`BrassScore` whose distances are the weighted sums of the components'
        distances, with the BRASS score recomputed from those sums.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    d_attacked = sum(w.get(k, 0.0) * c.d_attacked_base for k, c in components.items())
    d_clean = sum(w.get(k, 0.0) * c.d_clean_base for k, c in components.items())
    return BrassScore(
        distance_name="composite",
        brass=brass_from_distances(d_attacked, d_clean, eps=eps),
        d_attacked_base=d_attacked,
        d_clean_base=d_clean,
    )
