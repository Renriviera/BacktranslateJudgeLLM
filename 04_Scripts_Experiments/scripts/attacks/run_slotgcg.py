"""Run the (patched) SlotGCG optimizer on a converted BRASS dataset and write the adapter cache.

SlotGCG (``GCG_posinit_attention``) interleaves adversarial tokens at attention-selected slots
inside the behavior. This wrapper generates a self-contained method config (explicit OLMo
``target_model`` + ``use_prefix_cache: False`` so the transformers-5.x cache layout never hits the
legacy tuple-indexing path, + ``attn_implementation: eager`` so the VSS step gets attention
weights), runs ``generate_test_cases.py`` in ``.venv-attacks``, then merges the materialized
attacked prompts into ``06_Results_Artifacts/results/attacks/slotgcg/<model_tag>.json``.

Examples
--------
Dry run (1 behavior, few steps):
    .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_slotgcg.py --dataset strongreject --dry-run

Full run (launch yourself; long):
    .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_slotgcg.py --dataset harmbench --num-steps 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
SCRIPTS_ATTACKS = REPO / "04_Scripts_Experiments/scripts" / "attacks"
SG_DIR = REPO / "04_Scripts_Experiments/src" / "brass" / "attacks" / "external" / "SlotGCG"
VENV_PY = REPO / ".venv-attacks" / "bin" / "python"

sys.path.insert(0, str(SCRIPTS_ATTACKS))
from cache_io import merge_cache  # noqa: E402
from run_tao import DEFAULT_MODEL, DEFAULT_SEED, model_tag  # noqa: E402  (reuse the same tag mapping)
from worker_utils import assign_gpus, run_parallel, slice_ranges  # noqa: E402

EXPERIMENT = "olmo_slotgcg"


def write_method_config(path: Path, model: str, targets_path: Path, args) -> None:
    cfg = {
        "default_method_hyperparameters": {
            "num_test_cases_per_behavior": args.num_test_cases,
            "num_steps": args.num_steps,
            "adv_string_init": "a",
            "allow_non_ascii": False,
            "search_width": args.search_width,
            "num_adv_string": args.num_adv_string,
            "attention_temp": args.attention_temp,
            "use_prefix_cache": False,  # transformers 5.x DynamicCache != legacy tuple layout
            "targets_path": str(targets_path),
            "eval_steps": args.eval_steps,
            # Cheap, unconditional backstop: stop only when loss is ~0 (almost surely jailbroken).
            "early_stopping": args.early_stopping,
            "early_stopping_min_loss": args.early_stopping_min_loss,
            # Verified stop: every eval_steps, once loss < threshold, generate and stop iff the
            # completion is not a refusal (avoids over-counting on a deceptively low target loss).
            "eval_with_check_refusal": args.eval_with_check_refusal,
            "check_refusal_min_loss": args.check_refusal_min_loss,
        },
        EXPERIMENT: {
            "target_model": {
                "model_name_or_path": model,
                "dtype": "bfloat16",
                "use_fast_tokenizer": True,
                "trust_remote_code": False,
                "attn_implementation": "eager",  # needed for outputs.attentions (VSS slot scoring)
            },
            "targets_path": str(targets_path),
        },
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default="strongreject", choices=["strongreject", "harmbench"])
    ap.add_argument("--num-steps", type=int, default=500)
    ap.add_argument("--search-width", type=int, default=512)
    ap.add_argument("--num-adv-string", type=int, default=20)
    ap.add_argument("--attention-temp", type=float, default=8)
    ap.add_argument("--num-test-cases", type=int, default=1)
    ap.add_argument("--max-behaviors", type=int, default=0, help="0 = all behaviors")
    # Early-stopping controls (default OFF -> run the full num_steps budget).
    ap.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="Steps between eval/refusal checks (lower => verified stop can trigger sooner).",
    )
    ap.add_argument(
        "--eval-with-check-refusal",
        action="store_true",
        help="Verified stop: every --eval-steps, once loss < --check-refusal-min-loss, generate a "
        "completion and stop the behavior iff it is NOT a refusal.",
    )
    ap.add_argument("--check-refusal-min-loss", type=float, default=0.1)
    ap.add_argument(
        "--early-stopping",
        action="store_true",
        help="Cheap backstop: stop as soon as loss < --early-stopping-min-loss (proxy, unverified). "
        "Keep its threshold below --check-refusal-min-loss so the verified stop stays primary.",
    )
    ap.add_argument("--early-stopping-min-loss", type=float, default=0.01)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel per-behavior workers over disjoint behavior slices. NOTE: on a single GPU "
        "this gives ~no speedup (search_width=512 already saturates compute); set >1 only with "
        "multiple GPUs via --gpu '0,1,...' (workers are assigned round-robin).",
    )
    ap.add_argument(
        "--gpu",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="GPU id, or comma-separated ids for multi-GPU workers (e.g. '0,1,2,3').",
    )
    ap.add_argument("--data-dir", default=str(REPO / "01_Datasets_Benchmarks/data" / "attacks"))
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute every behavior. Default (off) resumes: skip behaviors already saved.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="1 behavior, 3 steps, search-width 64, num-adv-string 5 - validate the pipeline.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Passed through to generate_test_cases.py (overrideable; project default 235711).",
    )
    args = ap.parse_args()

    if args.dry_run:
        args.num_steps, args.search_width, args.num_adv_string, args.max_behaviors = 3, 64, 5, 1
        args.num_workers = 1

    behaviors_csv = Path(args.data_dir) / "slotgcg" / f"{args.dataset}_behaviors.csv"
    targets_json = (Path(args.data_dir) / "slotgcg" / f"{args.dataset}_targets.json").resolve()
    if not behaviors_csv.exists() or not targets_json.exists():
        sys.exit(
            f"Missing converted SlotGCG inputs for '{args.dataset}'; run build_attack_datasets.py."
        )

    tag = model_tag(args.model)
    run_dir = REPO / "06_Results_Artifacts/results" / "attacks" / "_native" / "slotgcg" / f"{tag}_{args.dataset}"
    save_dir = run_dir / "test_cases"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "method_config.gen.yaml"
    write_method_config(cfg_path, args.model, targets_json, args)
    out_path = (
        Path(args.out) if args.out else REPO / "06_Results_Artifacts/results" / "attacks" / "slotgcg" / f"{tag}.json"
    )

    # Behavior selection (CSV order).
    with behaviors_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bid_to_behavior = {r["BehaviorID"]: r["Behavior"] for r in rows}

    total = len(rows)
    if args.max_behaviors and args.max_behaviors > 0:
        total = min(args.max_behaviors, total)
    # Disjoint contiguous behavior slices, one per parallel worker (all share the GPU + save_dir;
    # SlotGCG writes per-behavior subdirs so concurrent workers never collide).
    shards = slice_ranges(total, args.num_workers)

    def build_cmd(start: int, end: int) -> list[str]:
        cmd = [
            str(VENV_PY),
            "generate_test_cases.py",
            "--method_name",
            "GCG_posinit_attention",
            "--experiment_name",
            EXPERIMENT,
            "--method_config_file",
            str(cfg_path.resolve()),
            "--models_config_file",
            str(SG_DIR / "configs" / "model_configs" / "models.yaml"),
            "--behaviors_path",
            str(behaviors_csv.resolve()),
            "--save_dir",
            str(save_dir.resolve()),
            "--behavior_start_idx",
            str(start),
            "--behavior_end_idx",
            str(end),
            "--seed",
            str(args.seed),
            "--verbose",
        ]
        # Default = resume: behaviors with an existing test_cases.json are skipped (crash-safe via
        # the incremental per-behavior save patch). --overwrite forces a full recompute.
        if args.overwrite:
            cmd.append("--overwrite")
        return cmd

    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = os.pathsep.join(
        [str(SCRIPTS_ATTACKS), str(SG_DIR), base_env.get("PYTHONPATH", "")]
    )
    base_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmds = [build_cmd(s, e) for s, e in shards]
    gpu_ids = assign_gpus(len(cmds), args.gpu)
    envs = []
    for gid in gpu_ids:
        e = dict(base_env)
        e["CUDA_VISIBLE_DEVICES"] = gid
        envs.append(e)
    log_paths = [run_dir / f"worker_{i}.log" for i in range(len(cmds))]
    print(
        f"[run_slotgcg] {len(cmds)} worker(s) over {total} behaviors, slices={shards}, "
        f"gpus={gpu_ids}, search_width={args.search_width}, steps={args.num_steps}, "
        f"seed={args.seed}"
    )
    if len(cmds) == 1:
        subprocess.run(cmds[0], cwd=str(SG_DIR), env=envs[0], check=True)
    else:
        rcs = run_parallel(cmds, cwd=str(SG_DIR), envs=envs, log_paths=log_paths)
        print(f"[run_slotgcg] worker exit codes: {rcs} (logs in {run_dir})")
        if any(rc != 0 for rc in rcs):
            sys.exit(f"[run_slotgcg] {sum(rc != 0 for rc in rcs)} worker(s) failed; see logs.")

    # Collect per-behavior outputs: save_dir/test_cases_individual_behaviors/<bid>/test_cases.json
    indiv = save_dir / "test_cases_individual_behaviors"
    if not indiv.exists():
        sys.exit(f"[run_slotgcg] no outputs at {indiv}")

    items = []
    for bid_dir in sorted(indiv.iterdir()):
        tc_file = bid_dir / "test_cases.json"
        if not tc_file.exists():
            continue
        tc = json.loads(tc_file.read_text(encoding="utf-8"))
        for bid, cases in tc.items():
            if not cases:
                continue
            items.append(
                {
                    "bid": bid,
                    "behavior": bid_to_behavior.get(bid, ""),
                    "attacked_prompt": cases[0],
                    "extra": {"attack": "slotgcg", "all_test_cases": cases},
                }
            )

    if not items:
        sys.exit("[run_slotgcg] produced no test cases")
    n = merge_cache(out_path, args.dataset, items)
    print(f"[run_slotgcg] merged {n} behaviors ({args.dataset}) -> {out_path}")


if __name__ == "__main__":
    main()
