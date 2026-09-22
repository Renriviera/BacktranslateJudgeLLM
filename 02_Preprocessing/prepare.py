#!/usr/bin/env python3
"""Prepare independent study inputs without mutating historical results or launching models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "04_Scripts_Experiments/src"))
from brass.paths import RESULTS, NEW_RUNS


def new_orbit_study(out: Path, manifest: Path, config: Path):
    out = out.resolve()
    if out.is_relative_to(RESULTS.resolve()):
        raise ValueError("Choose a new study directory outside the historical results archive")
    if out.exists():
        raise FileExistsError(f"Study directory already exists: {out}")
    rows = json.loads(manifest.read_text())
    settings = json.loads(config.read_text())
    from brass.orbits.runner import OrbitConfig
    OrbitConfig(**settings)
    ids = [r["id"] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Manifest must contain unique, nonempty study inputs")
    out.mkdir(parents=True)
    for source, name in [(manifest, "manifest.json"), (config, "config.json")]:
        shutil.copyfile(source, out / name)
    record = {
        "status": "inputs_prepared_preflight_required",
        "source_manifest": str(manifest.relative_to(ROOT)) if manifest.is_relative_to(ROOT) else manifest.name,
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "items": len(rows),
        "new_generation_performed": False,
        "notes": ["Reusing a manifest is a replication; it does not create a new held-out split.", "Record local model revisions with orbits/preflight.py before inference.", "Prepare evaluator dependencies before scoring benign code/IFEval tasks."],
    }
    (out / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    orbit = sub.add_parser("orbit", help="Copy a frozen manifest/config into a fresh run")
    orbit.add_argument("--out", type=Path, required=True)
    orbit.add_argument("--manifest", type=Path, default=RESULTS / "orbits/2026-09-18-pilot/main_manifest.json")
    orbit.add_argument("--config", type=Path, default=RESULTS / "orbits/2026-09-18-pilot/main_config.json")
    judge = sub.add_parser("judge", help="Prepare the fixed-response judge cohort from archived evidence")
    judge.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "orbit":
        result = new_orbit_study(args.out, args.manifest.resolve(), args.config.resolve())
    else:
        out = args.out.resolve()
        if out.exists() or out.is_relative_to(RESULTS.resolve()):
            parser.error("Choose a fresh output directory outside the historical archive")
        from brass.backtranslation_judge.data import prepare
        result = prepare(ROOT, out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
