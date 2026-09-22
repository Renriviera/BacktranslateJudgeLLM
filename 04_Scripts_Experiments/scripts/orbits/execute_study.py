#!/usr/bin/env python
"""Execute a frozen, budget-limited queue with durable stage status and local logs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.orbits.io import append_jsonl, digest, file_digest, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    root = Path(queue["run_dir"])
    errors = []

    def stop_requested(signum, frame):
        raise KeyboardInterrupt("Study stop requested")

    signal.signal(signal.SIGTERM, stop_requested)
    with (root / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def step(job, stage, script, flags):
            for path, expected in {**queue["source_hashes"], **queue["frozen_file_hashes"]}.items():
                if file_digest(Path(path)) != expected:
                    raise RuntimeError(f"Frozen input/source changed: {path}")
            if stage in job.get("skip_stages", []):
                print(f"Reusing audited completed stage {job['name']}: {stage}", flush=True)
                return True
            argv = [sys.executable, str(REPO / "04_Scripts_Experiments/scripts/orbits" / script), *map(str, flags)]
            marker = (
                root
                / queue.get("marker_directory", "execution_markers")
                / f"{job['name']}_{stage}.json"
            )
            signature = digest([argv, queue["source_hashes"], queue["frozen_file_hashes"]])
            if marker.exists():
                if json.loads(marker.read_text())["signature"] != signature:
                    raise RuntimeError("Completed stage differs from current frozen queue")
                return True
            record = {
                "job": job["name"],
                "stage": stage,
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "command": argv,
                "status": "running",
            }
            write_json(root / "study_progress.json", record)
            log = root / "execution_logs" / f"{job['name']}_{stage}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"Starting {job['name']}: {stage}", flush=True)
            with log.open("a") as out:
                worker = subprocess.Popen(
                    argv, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
                )
                record["worker_pid"] = worker.pid
                write_json(root / "study_progress.json", record)
                try:
                    returncode = worker.wait()
                except BaseException:
                    os.killpg(worker.pid, signal.SIGTERM)
                    try:
                        worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid, signal.SIGKILL)
                    write_json(
                        root / "study_progress.json",
                        {
                            **record,
                            "status": "interrupted",
                            "resume": "Rerun the same frozen queue; completed calls are checkpointed.",
                        },
                    )
                    raise
            record.update(
                returncode=returncode,
                finished_at=datetime.now(timezone.utc).isoformat(),
                status="complete" if returncode == 0 else "failed",
            )
            append_jsonl(root / "execution_history.jsonl", [record])
            if returncode:
                errors.append(record)
                print(f"Stage failed; inspect {log}", flush=True)
                return False
            write_json(marker, {"signature": signature, **record})
            return True

        for job in queue["jobs"]:
            if job.get("skip_generation"):
                continue
            shared = [
                "--manifest",
                job["manifest"],
                "--config",
                job["config"],
                "--output",
                job["output"],
            ]
            if job.get("reuse"):
                if not step(
                    job,
                    "reuse",
                    "reuse_prefix.py",
                    [
                        *shared,
                        "--source",
                        root / job["reuse"],
                        "--max-stage",
                        job["reuse_max_stage"],
                    ],
                ):
                    break
            if job.get("seed_archives"):
                if not step(
                    job, "archive_import", "seed_archives.py", ["--run-dir", root, *shared]
                ):
                    break
            if not step(job, "generation", "run.py", ["--run-dir", root, *shared]):
                break  # No unbounded retries or budget expansion.
            step(
                job,
                "validity",
                "summarize.py",
                ["--manifest", job["manifest"], "--events-dir", job["output"]],
            )

        # Score observed data even if a later generation job hit the hard resource ceiling.
        for job in queue["jobs"]:
            if not (Path(job["output"]) / "events.jsonl").exists():
                continue
            shared = [
                "--run-dir",
                root,
                "--manifest",
                job["manifest"],
                "--events-dir",
                job["output"],
            ]
            for judge in ["strongreject", "harmbench"]:
                step(job, judge, "score_harmful.py", [*shared, "--judge", judge])
            if job["name"] == "main_core":
                step(
                    job,
                    "human_review",
                    "export_review.py",
                    [
                        "--manifest",
                        job["manifest"],
                        "--events-dir",
                        job["output"],
                        "--output",
                        Path(job["output"]) / "human_review",
                    ],
                )
            step(job, "inverse_surprisal", "score_inverse_surprisal.py", shared)
            step(job, "benign", "score_benign.py", [*shared, "--code-image", queue["code_image"]])
            step(
                job,
                "embeddings",
                "embed_orbits.py",
                [
                    *shared,
                    "--device",
                    queue.get("embedding_device", "cpu"),
                    "--batch-size",
                    queue.get("embedding_batch_size", 64),
                ],
            )
            step(
                job,
                "analysis",
                "analyze.py",
                ["--manifest", job["manifest"], "--events-dir", job["output"]],
            )
        write_json(
            root / "study_progress.json",
            {
                "status": "queue_complete_with_errors" if errors else "queue_complete",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "errors": errors,
                "scientific_status": "Descriptive computation only; human labels, confirmatory statistics and detection validation remain pending.",
                "pending_optional_diagnostics": queue["pending_optional_diagnostics"],
            },
        )
        print("Frozen computational queue finished.", flush=True)


if __name__ == "__main__":
    main()
