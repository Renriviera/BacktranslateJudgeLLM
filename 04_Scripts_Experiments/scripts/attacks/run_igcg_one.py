#!/usr/bin/env python3
"""Run one copied I-GCG behavior and save only its final prompt and response."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
IGCG_ROOT = REPO / "04_Scripts_Experiments/src/brass/attacks/external/I-GCG"
IGCG = IGCG_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behavior-index", type=int, required=True, help="One-based row in copied I-GCG AdvBench-50.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--force", action="store_true", help="Rerun optimization even if a saved trajectory exists.")
    args = ap.parse_args()
    source = REPO.parent / "attacks/igcg/benchmarks/advbench-50/behaviors.json"
    rows = json.loads(source.read_text())
    if not 1 <= args.behavior_index <= len(rows):
        ap.error(f"--behavior-index must be between 1 and {len(rows)}")
    row = rows[args.behavior_index - 1]
    configs = json.loads((IGCG / "configs/behaviors_ours_config_init.json").read_text())
    config_rows = configs if isinstance(configs, list) else list(configs.values())
    config_row = dict(config_rows[args.behavior_index - 1])
    config_row["behaviour"] = row["behaviour"]
    config_row["target"] = row["target"]
    config_row["step"] = args.steps
    config_row["batch_size"] = 256
    config_row["top_k"] = 256
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / "behavior.json"
    config_rows = [dict(r) for r in config_rows]
    config_rows[args.behavior_index - 1] = config_row
    config.write_text(json.dumps(config_rows, indent=2) + "\n")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    command = [sys.executable, "attack_llm_core_best_update_our_target.py", "--model_path", args.model,
               "--id", str(row["id"]), "--device", "0", "--K", "3", "--defense", "without_defense",
               "--behaviors_config", str(config), "--output_path", str(args.output), "--incremental_token_num", "3"]
    logs_path = args.output / "3" / "log" / f"result_{row['id']}.json"
    if args.force or not logs_path.exists():
        subprocess.run(command, cwd=IGCG, env=env, check=True)
    records = json.loads(logs_path.read_text())
    final = records[-1]
    result = {"behavior_id": str(row["id"]), "original_prompt": row["behaviour"],
              "attacked_prompt": row["behaviour"] + final["adv_suffix"],
              "target_response": None, "final_attack_step": final["step"],
              "attack_loss": final.get("loss"), "attack": "I-GCG", "dataset": "I-GCG AdvBench-50"}
    (args.output / "final_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output / "final_result.json")


if __name__ == "__main__":
    main()
