"""Stall-watchdog wrapper around run_tao.py for unattended multi-day runs.

Same design as ``run_slotgcg_watchdog.py``: monitors run-log byte growth, kills on stall,
relaunches in resume mode (``run_tao.py`` skips behaviors already in ``tao_results.jsonl``).

Example:
    setsid nohup .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_tao_watchdog.py \
        --log 06_Results_Artifacts/results/attacks/logs/tao_sr_7b.log --stall-timeout 600 -- \
        --dataset strongreject --model allenai/Olmo-3-7B-Instruct \
        --num-steps 500 --batch-size 256 --num-workers 1 --gpu 0 &
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
PY = REPO / ".venv" / "bin" / "python"
RUNNER = REPO / "04_Scripts_Experiments/scripts" / "attacks" / "run_tao.py"
# Matches run_tao.FATAL_EXIT_CODE: seed jailbreak failure and other doomed-run aborts.
FATAL_EXIT_CODES = {78}


def log_size(log: Path) -> int:
    try:
        return log.stat().st_size
    except OSError:
        return -1


def hb(hb_path: Path, msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(hb_path, "a", encoding="utf-8") as f:
        f.write(f"[watchdog {stamp}] {msg}\n")


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--log", required=True, help="Run log file the runner writes to (progress source)."
    )
    ap.add_argument(
        "--heartbeat-log",
        default=None,
        help="Watchdog event/heartbeat log (default: <log>.watchdog).",
    )
    ap.add_argument(
        "--stall-timeout",
        type=int,
        default=600,
        help="Restart if the run log has not grown for N s.",
    )
    ap.add_argument("--poll", type=int, default=20, help="Stall-check interval (s).")
    ap.add_argument(
        "--heartbeat-every",
        type=int,
        default=120,
        help="Emit an 'alive' heartbeat at most every N s.",
    )
    ap.add_argument("--max-restarts", type=int, default=100)
    ap.add_argument(
        "--cooldown", type=int, default=15, help="Seconds to let the GPU free after a kill."
    )
    ap.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Everything after `--` is passed through to run_tao.py (resume mode).",
    )
    args = ap.parse_args()

    passthrough = list(args.runner_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    hb_path = (
        Path(args.heartbeat_log)
        if args.heartbeat_log
        else log_path.with_suffix(log_path.suffix + ".watchdog")
    )

    env = dict(os.environ)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    restarts = 0
    while True:
        hb(
            hb_path,
            f"launch attempt {restarts} (stall_timeout={args.stall_timeout}s, poll={args.poll}s)",
        )
        out = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
        proc = subprocess.Popen(
            [str(PY), "-u", str(RUNNER), *passthrough],
            cwd=str(REPO),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        last_size = log_size(log_path)
        last_change = time.time()
        last_hb = 0.0
        stalled = False
        while proc.poll() is None:
            time.sleep(args.poll)
            now = time.time()
            size = log_size(log_path)
            if size > last_size:
                last_size = size
                last_change = now
            idle = now - last_change
            if now - last_hb >= args.heartbeat_every:
                hb(
                    hb_path,
                    f"alive pid={proc.pid} log_size={size} idle={idle:.0f}s/{args.stall_timeout}s",
                )
                last_hb = now
            if idle > args.stall_timeout:
                hb(
                    hb_path,
                    f"STALL: run log unchanged for {idle:.0f}s -> killing process group & restarting",
                )
                kill_tree(proc)
                stalled = True
                break
        out.close()
        rc = proc.poll()

        if not stalled and rc == 0:
            hb(hb_path, "runner exited 0 -> DONE")
            return 0
        if not stalled and rc in FATAL_EXIT_CODES:
            hb(hb_path, f"runner exited {rc} FATAL abort -> not retrying")
            return rc
        hb(
            hb_path,
            f"{'stalled+killed' if stalled else f'runner exited {rc}'} -> retry after {args.cooldown}s",
        )
        time.sleep(args.cooldown)
        restarts += 1
        if restarts > args.max_restarts:
            hb(hb_path, f"exceeded max-restarts={args.max_restarts}; giving up")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
