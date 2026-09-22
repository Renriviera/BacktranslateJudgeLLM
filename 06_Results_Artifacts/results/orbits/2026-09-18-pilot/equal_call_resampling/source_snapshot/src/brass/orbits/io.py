"""Small, dependency-free provenance and append-only checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # Fail on a partial final line: never silently lose a budget/checkpoint record.
    # JSONL records are delimited by physical newlines. str.splitlines() also splits
    # legal JSON string characters such as U+2028, corrupting otherwise valid records.
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def stable_seed(root: int, *parts: object) -> int:
    return int(digest([root, *parts])[:8], 16) % (2**31 - 1)
