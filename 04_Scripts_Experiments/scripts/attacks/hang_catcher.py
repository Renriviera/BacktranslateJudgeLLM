"""One-shot hang diagnostic for the SlotGCG run.

Polls GPU utilization and the run-log byte size. When the worker is alive but the GPU sits idle and
the log stops growing (i.e. a hang, not slow compute), it snapshots nvidia-smi + /proc state and
sends SIGABRT to the worker so its PYTHONFAULTHANDLER prints all-thread Python tracebacks into the
run log. Captures one event then exits (the watchdog will relaunch the worker).

Run alongside the watchdog:
    nohup .venv/bin/python 04_Scripts_Experiments/scripts/attacks/hang_catcher.py \
        --log 06_Results_Artifacts/results/attacks/logs/slotgcg_sr_7b.log \
        --out 06_Results_Artifacts/results/attacks/logs/hang_diag.txt &
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

WORKER_PAT = "venv-attacks/bin/python generate_test_cases.py"


def gpu_util() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return 100  # on error, assume busy (don't false-trigger)


def worker_pid(pattern: str) -> int | None:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
        return int(out.strip().splitlines()[0])
    except subprocess.CalledProcessError:
        return None


def log_size(log: Path) -> int:
    try:
        return log.stat().st_size
    except OSError:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--util-threshold", type=int, default=15, help="GPU%% below this counts as idle.")
    ap.add_argument("--stall-secs", type=int, default=60, help="Log must be frozen this long to fire.")
    ap.add_argument("--poll", type=int, default=8)
    ap.add_argument(
        "--worker-pattern",
        default=WORKER_PAT,
        help="pgrep -f pattern identifying the GPU worker process.",
    )
    args = ap.parse_args()

    log, out = Path(args.log), Path(args.out)
    last_size, last_change, seen_growth = -1, time.time(), False

    while True:
        time.sleep(args.poll)
        pid = worker_pid(args.worker_pattern)
        if pid is None:  # between worker restarts
            last_size, last_change, seen_growth = -1, time.time(), False
            continue
        size = log_size(log)
        if size > last_size:
            if last_size >= 0:
                seen_growth = True
            last_size, last_change = size, time.time()
        log_idle = time.time() - last_change
        util = gpu_util()
        if seen_growth and util < args.util_threshold and log_idle > args.stall_secs:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(out, "a", encoding="utf-8") as f:
                f.write(f"\n==== HANG DETECTED {stamp} pid={pid} util={util}% log_idle={log_idle:.0f}s ====\n")
                f.write(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
                try:
                    f.write(f"\n--- /proc/{pid}/status ---\n")
                    f.write(Path(f"/proc/{pid}/status").read_text())
                    f.write("\n--- per-thread wchan ---\n")
                    for t in sorted(Path(f"/proc/{pid}/task").iterdir()):
                        f.write(f"{t.name}: {(t / 'wchan').read_text()}\n")
                except OSError:
                    pass
                f.write(f"\nsending SIGABRT to {pid} -> faulthandler tracebacks go to the run log\n")
            os.kill(pid, signal.SIGABRT)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
