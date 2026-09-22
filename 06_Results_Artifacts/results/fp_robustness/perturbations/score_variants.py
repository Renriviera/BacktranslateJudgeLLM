"""Thin wrapper. Canonical scorer: scripts/fp_robustness/score_variants.py."""

from __future__ import annotations

import runpy
from pathlib import Path

_CANONICAL = (
    Path(__file__).resolve().parents[3] / "scripts" / "fp_robustness" / "score_variants.py"
)
runpy.run_path(str(_CANONICAL), run_name="__main__")
