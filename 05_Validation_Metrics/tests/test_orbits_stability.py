import runpy
from pathlib import Path

import numpy as np
import pytest

module = runpy.run_path(
    str(next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir()) / "04_Scripts_Experiments/scripts/orbits/stability_update.py")
)


def test_pairwise_formula_matches_explicit_distances():
    v = np.random.default_rng(1).normal(size=(8, 4))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    expected = np.mean([1 - np.dot(v[i], v[j]) for i in range(8) for j in range(i)])
    assert module["pairwise_dispersion"](v) == pytest.approx(expected)
    assert module["pairwise_dispersion"]([[1, 0], [1, 0]]) == pytest.approx(0)


def test_task_groups_are_the_sampling_units():
    a = [{"group_id": "one", "v": 1.0}] * 8 + [{"group_id": "two", "v": 0.5}]
    b = [{"group_id": "three", "v": 0.2}, {"group_id": "four", "v": 0.0}]
    result = module["contrast"](a, b, "v")
    assert result["attack_groups"] == 2
    assert result["attack_mean"] == 0.75
    assert result["attack_minus_benign"] == pytest.approx(0.65)
