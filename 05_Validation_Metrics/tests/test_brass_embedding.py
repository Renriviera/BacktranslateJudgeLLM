"""Tests for BRASS embedding distances and the recovery score."""

import numpy as np

from brass.metrics.brass.base import brass_from_distances
from brass.metrics.brass.embedding import (
    centroid_cosine,
    compute_embedding_brass,
    energy_distance,
    mmd_rbf,
    sliced_wasserstein,
)

RNG = np.random.default_rng(235711)


def _cloud(center, n=64, dim=16, scale=0.1):
    return RNG.normal(loc=center, scale=scale, size=(n, dim))


def test_distances_zero_for_identical_clouds():
    x = _cloud(0.0)
    for fn in (mmd_rbf, energy_distance, centroid_cosine):
        assert fn(x, x) < 1e-6
    assert sliced_wasserstein(x, x, n_projections=64) < 1e-2


def test_distances_increase_with_separation():
    x = _cloud(0.0)
    near = _cloud(0.2)
    far = _cloud(3.0)
    for fn in (mmd_rbf, energy_distance):
        assert fn(x, near) < fn(x, far)


def test_brass_from_distances_semantics():
    # Attacked exactly recovers base -> BRASS ~ 1.
    assert abs(brass_from_distances(0.0, 1.0) - 1.0) < 1e-6
    # Attacked == clean -> BRASS ~ 0.
    assert abs(brass_from_distances(1.0, 1.0)) < 1e-6
    # Attacked farther than clean -> BRASS < 0.
    assert brass_from_distances(2.0, 1.0) < 0.0


def test_compute_embedding_brass_identity_attack():
    base = _cloud(0.0)
    clean = _cloud(1.0)
    attacked = clean.copy()  # identity attack: E_{a(p)}(I) == E_p(I)
    scores = compute_embedding_brass(base, clean, attacked, distances=["mmd", "energy"])
    for name in ("mmd", "energy"):
        assert abs(scores[name].brass) < 1e-3  # BRASS ~ 0 by construction
