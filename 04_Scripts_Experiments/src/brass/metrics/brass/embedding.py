"""Embedding-distribution BRASS distances.

Embed completions from ``E_p(M)`` (base), ``E_p(I)`` (clean instruct), and ``E_{a(p)}(I)``
(attacked instruct), then compare the resulting point clouds with several distribution distances:

- ``mmd``                - squared Maximum Mean Discrepancy with an RBF kernel.
- ``energy``             - energy distance (Szekely-Rizzo).
- ``sliced_wasserstein`` - sliced 2-Wasserstein (via POT if available, else a NumPy fallback).
- ``centroid_cosine``    - cosine distance between cloud centroids.

These are the most scalable default variant in the proposal. All distances operate on NumPy
arrays of shape ``(n_samples, dim)`` and are symmetric and non-negative.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

from brass.metrics.brass.base import BrassScore, brass_from_distances

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Lazily-cached embedding model (keyed by name+device).
_EMBED_CACHE: dict[tuple[str, str], object] = {}


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Embed a list of texts with a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    key = (model_name, device or "auto")
    model = _EMBED_CACHE.get(key)
    if model is None:
        model = SentenceTransformer(model_name, device=device)
        _EMBED_CACHE[key] = model
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float64)


# --------------------------------------------------------------------------------------------- #
# Distances                                                                                       #
# --------------------------------------------------------------------------------------------- #
def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x**2, axis=1)[:, None]
    y2 = np.sum(y**2, axis=1)[None, :]
    d = x2 + y2 - 2.0 * x @ y.T
    return np.clip(d, 0.0, None)


def _median_heuristic_gamma(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y])
    d = _pairwise_sq_dists(z, z)
    iu = np.triu_indices_from(d, k=1)
    med = np.median(d[iu]) if iu[0].size else 1.0
    med = med if med > 0 else 1.0
    return 1.0 / med


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased-ish squared MMD with an RBF kernel ``k(a,b)=exp(-gamma ||a-b||^2)``."""
    if gamma is None:
        gamma = _median_heuristic_gamma(x, y)
    kxx = np.exp(-gamma * _pairwise_sq_dists(x, x)).mean()
    kyy = np.exp(-gamma * _pairwise_sq_dists(y, y)).mean()
    kxy = np.exp(-gamma * _pairwise_sq_dists(x, y)).mean()
    return float(max(kxx + kyy - 2.0 * kxy, 0.0))


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Szekely-Rizzo energy distance between two samples."""
    dxy = np.sqrt(_pairwise_sq_dists(x, y)).mean()
    dxx = np.sqrt(_pairwise_sq_dists(x, x)).mean()
    dyy = np.sqrt(_pairwise_sq_dists(y, y)).mean()
    return float(max(2.0 * dxy - dxx - dyy, 0.0))


def sliced_wasserstein(
    x: np.ndarray, y: np.ndarray, n_projections: int = 128, seed: int = 0
) -> float:
    """Sliced 2-Wasserstein distance (POT if available, else a NumPy fallback)."""
    try:
        import ot  # POT

        return float(ot.sliced_wasserstein_distance(x, y, n_projections=n_projections, seed=seed))
    except Exception as exc:  # noqa: BLE001 - fallback keeps the metric available
        logger.debug("POT unavailable (%s); using NumPy sliced-Wasserstein fallback.", exc)
        rng = np.random.default_rng(seed)
        dim = x.shape[1]
        total = 0.0
        for _ in range(n_projections):
            v = rng.standard_normal(dim)
            v /= np.linalg.norm(v) + 1e-12
            xp = np.sort(x @ v)
            yp = np.sort(y @ v)
            n = min(len(xp), len(yp))
            # Equal-size 1D W2 via sorted matching (quantile coupling).
            xi = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(xp)), xp)
            yi = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(yp)), yp)
            total += np.mean((xi - yi) ** 2)
        return float(np.sqrt(total / n_projections))


def centroid_cosine(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine distance between cloud centroids (in [0, 2])."""
    cx = x.mean(axis=0)
    cy = y.mean(axis=0)
    denom = (np.linalg.norm(cx) * np.linalg.norm(cy)) + 1e-12
    return float(1.0 - (cx @ cy) / denom)


EMBEDDING_DISTANCES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mmd": mmd_rbf,
    "energy": energy_distance,
    "sliced_wasserstein": sliced_wasserstein,
    "centroid_cosine": centroid_cosine,
}


def compute_embedding_brass(
    base_emb: np.ndarray,
    clean_emb: np.ndarray,
    attacked_emb: np.ndarray,
    *,
    distances: Sequence[str] | None = None,
    eps: float = 1e-8,
) -> dict[str, BrassScore]:
    """Compute BRASS under each requested embedding distance.

    Args:
        base_emb: Embeddings of base-model completions ``E_p(M)``.
        clean_emb: Embeddings of clean instruct completions ``E_p(I)``.
        attacked_emb: Embeddings of attacked instruct completions ``E_{a(p)}(I)``.
        distances: Subset of :data:`EMBEDDING_DISTANCES` keys (defaults to all).
        eps: Stabilizer for the denominator.

    Returns:
        Mapping ``distance_name -> BrassScore``.
    """
    names = list(distances) if distances else list(EMBEDDING_DISTANCES)
    out: dict[str, BrassScore] = {}
    for name in names:
        fn = EMBEDDING_DISTANCES[name]
        d_clean = fn(clean_emb, base_emb)
        d_attacked = fn(attacked_emb, base_emb)
        out[name] = BrassScore(
            distance_name=f"embedding_{name}",
            brass=brass_from_distances(d_attacked, d_clean, eps=eps),
            d_attacked_base=d_attacked,
            d_clean_base=d_clean,
            n_base=len(base_emb),
            n_clean=len(clean_emb),
            n_attacked=len(attacked_emb),
        )
    return out
