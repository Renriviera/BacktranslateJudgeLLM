"""Write/merge the adapter cache JSON consumed by ``brass.attacks.{tao,slotgcg}_adapter``.

The adapters look up ``prompt.id`` first, then ``prompt.prompt`` (raw text). BRASS prompt ids are
``"{dataset}:{behavior_id}"`` (e.g. ``strongreject:274``, ``harmbench:syn_flood_...``), which match
``"{dataset}:{bid}"`` reconstructed from the converted datasets. We index each record under BOTH
keys so a lookup hits regardless of how the pipeline keys it.

Cache file is model-scoped (e.g. ``06_Results_Artifacts/results/attacks/tao/olmo31_instruct.json``) and merged across
datasets, so one file serves both the StrongREJECT and HarmBench experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def merge_cache(out_path: str | Path, dataset: str, items: list[dict[str, Any]]) -> int:
    """Merge ``items`` into the model-scoped cache at ``out_path``.

    Each item: ``{"bid", "behavior", "attacked_prompt", "extra": {...}}``.
    Returns the number of behaviors written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Any] = {}
    if out_path.exists():
        cache = json.loads(out_path.read_text(encoding="utf-8"))

    for it in items:
        record = {"attacked_prompt": it["attacked_prompt"], **(it.get("extra") or {})}
        cache[f"{dataset}:{it['bid']}"] = record
        if it.get("behavior"):
            cache[it["behavior"]] = record

    out_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return len(items)
