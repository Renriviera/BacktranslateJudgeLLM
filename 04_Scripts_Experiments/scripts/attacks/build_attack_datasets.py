"""Convert BRASS benchmarks into the input formats the vendored attacks expect.

Produces, for each requested dataset (default: strongreject, harmbench):

  TAO-Attack  -> ``<out>/tao/<dataset>.json``        : list[{"id","behavior","target"}]
  SlotGCG     -> ``<out>/slotgcg/<dataset>_behaviors.csv`` (HarmBench schema)
              -> ``<out>/slotgcg/<dataset>_targets.json``  : {BehaviorID: target}

GCG-family attacks optimize a cross-entropy loss toward an *affirmative target* string
(e.g. "Sure, here is ..."). HarmBench ships official targets; we use those when available
(keyed by BehaviorID) and otherwise synthesize a transparent affirmative target from the
behavior text. StrongREJECT has no official targets, so all of its targets are synthesized.

Run with the main BRASS venv (needs ``brass`` + ``datasets`` + network):

    .venv/bin/python 04_Scripts_Experiments/scripts/attacks/build_attack_datasets.py --max-behaviors 100
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import urllib.request
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
TAO_IGCG_PATH = (
    REPO
    / "04_Scripts_Experiments/src"
    / "brass"
    / "attacks"
    / "external"
    / "TAO-Attack"
    / "data"
    / "advbench"
    / "igcg.json"
)
ADVBENCH_EXPECTED_BEHAVIORS = 50
ADVBENCH_SEED_ID = "50"
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))

from brass import DEFAULT_SEED  # noqa: E402
from brass.data.loaders import BehaviorPrompt, load_dataset_prompts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_attack_datasets")

_HARMBENCH_TARGETS_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "01_Datasets_Benchmarks/data/optimizer_targets/harmbench_targets_text.json"
)

# Leading imperative verbs we strip so "Write a tutorial ..." -> "Sure, here is a tutorial ...".
_IMPERATIVE = {
    "write",
    "create",
    "develop",
    "give",
    "explain",
    "describe",
    "provide",
    "generate",
    "design",
    "list",
    "compose",
    "draft",
    "produce",
    "suggest",
    "outline",
    "detail",
    "demonstrate",
    "show",
    "teach",
    "tell",
    "offer",
    "share",
    "devise",
    "formulate",
    "construct",
    "build",
    "make",
    "prepare",
    "craft",
    "name",
    "identify",
    "recommend",
    "propose",
    "illustrate",
}


def affirmative_target(behavior: str) -> str:
    """Synthesize an affirmative GCG target from an imperative behavior string."""
    b = behavior.strip()
    b = re.sub(r"\s+", " ", b).rstrip(".")
    words = b.split()
    if words and words[0].lower() in _IMPERATIVE:
        rest = " ".join(words[1:]).strip()
        if rest:
            rest = rest[0].lower() + rest[1:]
            return f"Sure, here is {rest}"
    # Fallback for questions / non-imperative phrasing.
    return f"Sure, here is the information you requested regarding: {b}"


def _safe_behavior_id(prompt: BehaviorPrompt, fallback_idx: int) -> str:
    """Filesystem-safe BehaviorID (SlotGCG uses it as a directory name)."""
    native = prompt.id.split(":", 1)[-1]
    bid = native if native else f"{prompt.source}_{fallback_idx}"
    bid = re.sub(r"[^0-9A-Za-z_.-]+", "_", bid).strip("_")
    return bid or f"{prompt.source}_{fallback_idx}"


def _load_harmbench_official_targets() -> dict[str, str]:
    try:
        with urllib.request.urlopen(_HARMBENCH_TARGETS_URL, timeout=60) as r:  # noqa: S310
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch official HarmBench targets (%s); synthesizing all.", exc)
        return {}


def load_advbench_tao_records(path: Path = TAO_IGCG_PATH) -> list[dict[str, object]]:
    """Load and validate the paper's curated 50-behavior I-GCG AdvBench split."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing TAO AdvBench subset at {path}; clone the TAO-Attack external repo first."
        )

    raw_records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected a JSON list in {path}")

    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_behaviors: set[str] = set()
    for index, raw in enumerate(raw_records, start=1):
        behavior = str(raw.get("behavior") or raw.get("behaviour") or "").strip()
        target = str(raw.get("target") or "").strip()
        behavior_id = str(raw.get("id", index))
        if not behavior or not target:
            raise ValueError(f"AdvBench record {index} is missing behavior or target")
        if behavior_id in seen_ids:
            raise ValueError(f"Duplicate AdvBench id: {behavior_id}")
        if behavior in seen_behaviors:
            raise ValueError(f"Duplicate AdvBench behavior at id {behavior_id}")
        seen_ids.add(behavior_id)
        seen_behaviors.add(behavior)
        records.append(
            {
                "id": behavior_id,
                "behavior": behavior,
                "target": target,
                "is_seed": behavior_id == ADVBENCH_SEED_ID,
            }
        )

    if len(records) != ADVBENCH_EXPECTED_BEHAVIORS:
        raise ValueError(
            f"Expected {ADVBENCH_EXPECTED_BEHAVIORS} curated AdvBench behaviors, "
            f"found {len(records)} in {path}"
        )
    if sum(bool(record["is_seed"]) for record in records) != 1:
        raise ValueError(f"Expected exactly one easy-to-hard seed with id {ADVBENCH_SEED_ID}")
    return records


def build_for_dataset(name: str, max_behaviors: int, seed: int, out_dir: Path) -> dict[str, int]:
    if name == "advbench":
        records = load_advbench_tao_records()
        if max_behaviors > 0:
            records = records[:max_behaviors]
        (out_dir / "tao").mkdir(parents=True, exist_ok=True)
        tao_path = out_dir / "tao" / "advbench.json"
        tao_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        logger.info(
            "advbench: %d curated I-GCG behaviors (order preserved) -> %s",
            len(records),
            tao_path.relative_to(REPO) if tao_path.is_relative_to(REPO) else tao_path,
        )
        return {"behaviors": len(records), "official_targets": len(records)}

    prompts = load_dataset_prompts(name)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    if max_behaviors > 0:
        prompts = prompts[:max_behaviors]

    official = _load_harmbench_official_targets() if name == "harmbench" else {}

    tao_records: list[dict[str, str]] = []
    sg_rows: list[dict[str, str]] = []
    sg_targets: dict[str, str] = {}
    n_official = 0

    for i, p in enumerate(prompts):
        bid = _safe_behavior_id(p, i)
        native_id = p.id.split(":", 1)[-1]
        target = official.get(native_id)
        if target:
            n_official += 1
        else:
            target = affirmative_target(p.prompt)

        tao_records.append({"id": bid, "behavior": p.prompt, "target": target})
        sg_rows.append(
            {
                "Behavior": p.prompt,
                "FunctionalCategory": (p.metadata or {}).get("functional_category", "standard")
                or "standard",
                "SemanticCategory": p.category or "",
                "Tags": (p.metadata or {}).get("tags", "") or "",
                "ContextString": p.context or "",
                "BehaviorID": bid,
            }
        )
        sg_targets[bid] = target

    (out_dir / "tao").mkdir(parents=True, exist_ok=True)
    (out_dir / "slotgcg").mkdir(parents=True, exist_ok=True)

    tao_path = out_dir / "tao" / f"{name}.json"
    tao_path.write_text(json.dumps(tao_records, indent=2), encoding="utf-8")

    sg_csv = out_dir / "slotgcg" / f"{name}_behaviors.csv"
    with sg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Behavior",
                "FunctionalCategory",
                "SemanticCategory",
                "Tags",
                "ContextString",
                "BehaviorID",
            ],
        )
        writer.writeheader()
        writer.writerows(sg_rows)

    sg_tgt = out_dir / "slotgcg" / f"{name}_targets.json"
    sg_tgt.write_text(json.dumps(sg_targets, indent=2), encoding="utf-8")

    logger.info(
        "%s: %d behaviors (%d official HarmBench targets, %d synthesized) -> %s | %s | %s",
        name,
        len(prompts),
        n_official,
        len(prompts) - n_official,
        tao_path.relative_to(REPO),
        sg_csv.relative_to(REPO),
        sg_tgt.relative_to(REPO),
    )
    return {"behaviors": len(prompts), "official_targets": n_official}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="strongreject,harmbench")
    ap.add_argument(
        "--max-behaviors",
        type=int,
        default=100,
        help="Cap behaviors per dataset (main.tex main-run scale = 100). 0 = all.",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default=str(REPO / "01_Datasets_Benchmarks/data" / "attacks"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        build_for_dataset(name, args.max_behaviors, args.seed, out_dir)


if __name__ == "__main__":
    main()
