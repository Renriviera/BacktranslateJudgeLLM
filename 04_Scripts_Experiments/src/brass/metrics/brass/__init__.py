"""BRASS distance functions and the base-relative recovery score.

Implements Eq. (2) from ``main.tex``:

    BRASS_d(M, I, p, a) = 1 - d(E_{a(p)}(I), E_p(M)) / (d(E_p(I), E_p(M)) + eps)

Each distance family lives in its own module:

- ``embedding``  - distribution distances over completion embeddings (MMD, energy,
  sliced-Wasserstein, centroid-cosine). Primary, scalable default.
- ``likelihood`` - base log-likelihood recovery.
- ``logit``, ``activation``, ``deflection`` - stubbed interfaces for a later phase.
- ``composite``  - weighted combination (Eq. (3)).
"""

from brass.metrics.brass.base import BrassScore, brass_from_distances

__all__ = ["BrassScore", "brass_from_distances"]
