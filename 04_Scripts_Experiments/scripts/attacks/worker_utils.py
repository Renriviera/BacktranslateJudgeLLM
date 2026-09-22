"""Helpers for running attack optimizers as parallel per-behavior workers on one GPU.

The GCG-family optimizers are single-behavior (no native cross-behavior batching, since behaviors
have different token lengths). With a small 7B target, several model copies fit on a 96GB GPU, so we
get true behavior-level parallelism by launching N worker processes over disjoint behavior slices
that share the GPU. Each ~7B bf16 copy is ~14GB + a few GB of working memory, so 3-5 workers fit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def slice_ranges(n_items: int, n_workers: int) -> list[tuple[int, int]]:
    """Split ``range(n_items)`` into up to ``n_workers`` contiguous [start, end) slices."""
    n_workers = max(1, min(n_workers, max(1, n_items)))
    base, rem = divmod(n_items, n_workers)
    ranges, start = [], 0
    for i in range(n_workers):
        size = base + (1 if i < rem else 0)
        if size <= 0:
            continue
        ranges.append((start, start + size))
        start += size
    return ranges


def run_parallel(
    cmds: list[list[str]], cwd: str, envs: list[dict], log_paths: list[Path]
) -> list[int]:
    """Launch all ``cmds`` concurrently (one env each), stream to logs, wait, return exit codes."""
    procs, files = [], []
    for cmd, env, lp in zip(cmds, envs, log_paths, strict=True):
        lp.parent.mkdir(parents=True, exist_ok=True)
        f = open(lp, "w", encoding="utf-8")  # noqa: SIM115
        files.append(f)
        procs.append(subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT))
    try:
        return [p.wait() for p in procs]
    finally:
        for f in files:
            f.close()


def assign_gpus(n_workers: int, gpu_spec: str) -> list[str]:
    """Map each worker to a GPU id, round-robin over the comma-separated ``gpu_spec`` (e.g. "0,1").

    With a single GPU this co-locates all workers on it (no speedup for compute-bound GCG); with
    multiple GPUs it spreads workers out for near-linear throughput.
    """
    gpus = [g.strip() for g in str(gpu_spec).split(",") if g.strip()] or ["0"]
    return [gpus[i % len(gpus)] for i in range(n_workers)]
