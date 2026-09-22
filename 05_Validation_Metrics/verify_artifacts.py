#!/usr/bin/env python3
"""Verify exported scientific artifacts against the migration inventory."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-imported", action="store_true", help="Also check refactored code against export hashes")
    args = parser.parse_args()
    inventory = json.loads((ROOT / "06_Results_Artifacts/migration_manifest.json").read_text())
    failures, checked = [], 0
    for record in inventory["files"]:
        if not args.all_imported and not record["source"].startswith(("results/", "data/", "brass/")):
            continue
        checked += 1
        path = ROOT / record["destination"]
        if not path.is_file():
            failures.append({"path": record["destination"], "error": "missing"})
        elif file_hash(path) != record["export_sha256"]:
            failures.append({"path": record["destination"], "error": "hash_mismatch_or_unfetched_lfs_object"})
    print(json.dumps({"checked": checked, "failures": failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
