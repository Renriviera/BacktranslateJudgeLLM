"""Filesystem helpers for run artifacts (JSON / JSONL / dataframes)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):  # numpy arrays / scalars
        return obj.tolist()
    return obj


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    """Write ``data`` to ``path`` as pretty JSON, creating parent dirs."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_to_serializable, ensure_ascii=False)
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    """Write an iterable of records to a JSONL file."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_to_serializable, ensure_ascii=False) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[Any]:
    rows: list[Any] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
