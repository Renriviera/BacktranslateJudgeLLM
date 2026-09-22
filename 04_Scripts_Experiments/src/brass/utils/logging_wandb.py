"""Thin Weights & Biases wrapper.

Defaults to offline mode (``WANDB_MODE=offline``) so runs never block on a missing API key; set
``WANDB_MODE=online`` in ``.env`` to sync. If ``wandb`` is not installed the helpers degrade to
no-ops, keeping the pipeline runnable in minimal environments / CI.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


class WandbRun:
    """Context-managed W&B run. No-op when wandb is unavailable."""

    def __init__(
        self,
        *,
        project: str | None = None,
        entity: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        mode: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and _WANDB_AVAILABLE
        self._run = None
        self._project = project or os.environ.get("WANDB_PROJECT", "brass")
        self._entity = entity or os.environ.get("WANDB_ENTITY") or None
        self._name = name
        self._config = config or {}
        self._mode = mode or os.environ.get("WANDB_MODE", "offline")
        if enabled and not _WANDB_AVAILABLE:
            logger.warning("wandb not installed; logging is disabled.")

    def __enter__(self) -> WandbRun:
        if self.enabled:
            self._run = wandb.init(
                project=self._project,
                entity=self._entity,
                name=self._name,
                config=self._config,
                mode=self._mode,
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._run is not None:
            self._run.finish(exit_code=0 if exc_type is None else 1)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def log_table(self, key: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """Log tabular data (e.g. per-prompt completions) as a ``wandb.Table``."""
        if self._run is not None:
            table = wandb.Table(columns=list(columns))
            for row in rows:
                table.add_data(*row)
            self._run.log({key: table})

    @property
    def active(self) -> bool:
        return self._run is not None
