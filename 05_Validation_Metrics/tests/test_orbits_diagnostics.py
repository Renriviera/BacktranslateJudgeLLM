"""Diagnostic reuse must preserve the actual request and clustering unit."""

import json
import runpy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from brass.orbits.io import read_jsonl
from brass.orbits.runner import Generation, OrbitConfig, run_orbits

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
copy_prefix = runpy.run_path(str(REPO / "04_Scripts_Experiments/scripts/orbits/reuse_prefix.py"))["copy_prefix"]
group_interval = runpy.run_path(str(REPO / "04_Scripts_Experiments/scripts/orbits/analyze.py"))["group_interval"]


class Backend:
    fingerprint = {"test": True}

    def __init__(self):
        self.requests = []

    def generate(self, requests):
        self.requests.extend(requests)
        return [
            Generation(
                (
                    '{"prompt":"next task"}'
                    if "assistant_response" in r.messages[0]["content"]
                    else "a response"
                ),
                5,
                10,
                "stop",
            )
            for r in requests
        ]


def test_deep_extension_reuses_actual_prefix_without_charging_it_twice(tmp_path):
    manifest = [{"id": "a", "group_id": "g", "cohort": "benign", "prompt": "task"}]
    cfg = OrbitConfig(round_trips=1, sampled_trajectories=2, greedy=False)
    run_orbits(manifest, cfg, Backend(), tmp_path / "core", tmp_path / "budget.jsonl")
    extended = replace(cfg, round_trips=3, sampled_trajectories=1)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "config.json").write_text(json.dumps(asdict(extended)))
    copy_prefix(
        tmp_path / "core",
        tmp_path / "deep",
        tmp_path / "manifest.json",
        tmp_path / "config.json",
        2,
    )
    backend = Backend()
    run_orbits(manifest, extended, backend, tmp_path / "deep", tmp_path / "budget.jsonl")
    assert len(backend.requests) == 4
    assert len(read_jsonl(tmp_path / "deep/events.jsonl")) == 7


def test_prefix_cannot_reuse_changed_inverse_instruction(tmp_path):
    manifest = [{"id": "a", "group_id": "g", "cohort": "benign", "prompt": "task"}]
    cfg = OrbitConfig(round_trips=1, sampled_trajectories=1, greedy=False)
    run_orbits(manifest, cfg, Backend(), tmp_path / "core", tmp_path / "budget.jsonl")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "config.json").write_text(
        json.dumps(asdict(replace(cfg, inverse_template="minimal")))
    )
    with pytest.raises(ValueError, match="request mismatch"):
        copy_prefix(
            tmp_path / "core",
            tmp_path / "other",
            tmp_path / "manifest.json",
            tmp_path / "config.json",
            2,
        )
    assert not (tmp_path / "other/events.jsonl").exists()


def test_cluster_mean_does_not_treat_repeated_trajectories_as_new_tasks():
    result = group_interval([("many", 1.0)] * 100 + [("one", 0.0)], 7)
    assert result["mean"] == 0.5
    assert result["n_groups"] == 2
