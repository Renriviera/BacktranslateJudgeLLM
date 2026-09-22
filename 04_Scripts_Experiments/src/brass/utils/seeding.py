"""Reproducibility helpers.

The default seed is ``235711`` (see :data:`brass.DEFAULT_SEED`) and must be overrideable from
configuration. We seed Python's ``random``, NumPy, and PyTorch (CPU + CUDA) when available.
Full determinism on GPU is not guaranteed; document any nondeterminism at the call site.
"""

from __future__ import annotations

import os
import random

from brass import DEFAULT_SEED


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = False) -> int:
    """Seed all common RNGs.

    Args:
        seed: Seed value. Defaults to the project-wide :data:`brass.DEFAULT_SEED`.
        deterministic_torch: If ``True``, request deterministic cuDNN/cuBLAS kernels. This can
            slow training/inference and is off by default; sampling-based generation remains
            nondeterministic unless the engine is also configured for greedy/seeded decoding.

    Returns:
        The seed that was applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        pass

    return seed
