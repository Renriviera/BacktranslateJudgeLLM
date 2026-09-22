"""Guard relocation, fresh-run isolation, and preservation of archived inputs."""
import json
from pathlib import Path
import runpy

import pytest

from brass.orbits.runner import OrbitConfig
from brass.paths import ROOT, RESULTS, hf_hub_cache, recorded_path
from dataclasses import asdict

prepare = runpy.run_path(str(ROOT / "02_Preprocessing/prepare.py"))["new_orbit_study"]


def test_recorded_paths_prefer_new_checkout_even_when_old_tree_exists():
    expected = RESULTS / "orbits/example/events.jsonl"
    assert recorded_path("/old/work/BRASS/results/orbits/example/events.jsonl") == expected
    assert recorded_path("results/orbits/example/events.jsonl") == expected
    assert recorded_path(str(expected)) == expected


def test_explicit_hf_hub_cache_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert hf_hub_cache() == tmp_path / "hub"


def test_new_study_preserves_inputs_and_refuses_overwrite(tmp_path):
    manifest, config = tmp_path / "manifest.json", tmp_path / "config.json"
    manifest.write_text(json.dumps([{"id": "benign:one", "prompt": "Count three apples.", "cohort": "benign"}]))
    config.write_text(json.dumps(asdict(OrbitConfig())))
    out = tmp_path / "new"
    report = prepare(out, manifest, config)
    assert report["items"] == 1 and not report["new_generation_performed"]
    assert (out / "manifest.json").read_bytes() == manifest.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(out, manifest, config)
    with pytest.raises(ValueError, match="historical"):
        prepare(RESULTS / "should-never-be-created", manifest, config)


def test_duplicate_manifest_fails_before_creating_output(tmp_path):
    manifest, config = tmp_path / "manifest.json", tmp_path / "config.json"
    manifest.write_text(json.dumps([{"id": "same"}, {"id": "same"}]))
    config.write_text(json.dumps(asdict(OrbitConfig())))
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="unique"):
        prepare(out, manifest, config)
    assert not out.exists()
