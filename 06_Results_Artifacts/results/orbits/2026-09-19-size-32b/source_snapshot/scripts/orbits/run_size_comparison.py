#!/usr/bin/env python
"""Run the frozen, resumable two-cycle size study with sequential GPU stages."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.io import append_jsonl, digest, file_digest, read_jsonl, write_json  # noqa: E402
from brass.orbits.runner import (  # noqa: E402
    Budget,
    OrbitConfig,
    Request,
    VLLMBackend,
    parse_inverse,
    run_orbits,
)


def verify(root):
    for name, expected in json.loads((root / "frozen/hashes.json").read_text()).items():
        if file_digest(Path(name)) != expected:
            raise RuntimeError(f"Frozen input/source changed: {name}")


def fixed_inverse(root, config, backend):
    dest = root / "fixed_7b_response_inverse_32b"
    dest.mkdir(exist_ok=True)
    fingerprint = {
        "backend": backend.fingerprint,
        "source_sha256": file_digest(root / "baseline_7b/events.jsonl"),
        "config": asdict(config),
    }
    if (dest / "run_spec.json").exists() and json.loads(
        (dest / "run_spec.json").read_text()
    ) != fingerprint:
        raise RuntimeError("Fixed-response resume mismatch")
    write_json(dest / "run_spec.json", fingerprint)
    source = [
        e for e in read_jsonl(root / "baseline_7b/events.jsonl") if e["direction"] == "inverse"
    ]
    done = {e["id"] for e in read_jsonl(dest / "events.jsonl")}
    pending = [e for e in source if e["id"] not in done]
    budget = Budget(root / "token_budget.jsonl", config.global_generated_token_limit)
    for start in range(0, len(pending), config.batch_size):
        batch = pending[start : start + config.batch_size]
        requests = [
            Request(
                e["id"],
                e["input_messages"],
                e["seed"],
                0.0 if e["trajectory"] == "greedy" else config.inverse_temperature,
                config.inverse_max_tokens,
                config.top_p,
            )
            for e in batch
        ]
        if any(
            digest(asdict(r)) != e["request_sha256"] for r, e in zip(requests, batch, strict=True)
        ):
            raise RuntimeError("Fixed-response request not identical to 7B request")
        reservation = digest([str(dest), [e["id"] for e in batch], time.time_ns()])
        budget.reserve(reservation, len(batch) * config.inverse_max_tokens)
        outputs = backend.generate(requests)
        rows = []
        for e, g in zip(batch, outputs, strict=True):
            reconstructed = None
            if g.finish_reason == "stop":
                reconstructed, status = parse_inverse(g.text)
            else:
                status = {
                    "length": "generation_truncated",
                    "context_overflow": "context_overflow",
                }.get(g.finish_reason, "generation_failure")
            row = {
                k: e[k]
                for k in [
                    "id",
                    "item_id",
                    "group_id",
                    "trajectory",
                    "stage",
                    "step",
                    "direction",
                    "cohort",
                    "parent_id",
                    "request_sha256",
                    "seed",
                    "input_messages",
                ]
            }
            row.update(
                asdict(g),
                status=status,
                reconstructed_prompt=reconstructed,
                reservation_id=reservation,
                source_inverse_sha256=digest(e),
                diagnostic="same_7b_response_different_inverse_model",
            )
            rows.append(row)
        append_jsonl(dest / "events.jsonl", rows)
        budget.settle(reservation, sum(g.output_tokens for g in outputs))
        write_json(
            dest / "progress.json",
            {
                "completed": len(done) + start + len(batch),
                "planned": len(source),
                "global_tokens_charged": budget.used(),
            },
        )
        print(
            f"fixed-response inversions {len(done) + start + len(batch)}/{len(source)}", flush=True
        )
    write_json(dest / "completion.json", {"completed": True, "events": len(source)})


def generate(root):
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from brass.serving.vllm_engine import ModelSpec, VLLMEngine

    runtime = json.loads((root / "preflight.json").read_text())
    model = runtime["models"]["target"]
    config = OrbitConfig(**json.loads((root / "frozen/config.json").read_text()))
    engine_settings = {
        "enforce_eager": True,
        "max_num_seqs": 64,
        "override_generation_config": {"eos_token_id": [100265, 100257]},
    }
    spec = ModelSpec(
        name="orbit_olmo31_32b_matched",
        hf_id=model["snapshot"],
        is_chat=True,
        dtype="bfloat16",
        max_model_len=config.max_model_len,
        gpu_memory_utilization=0.90,
        extra=engine_settings,
    )
    fingerprint = {
        "model": model,
        "versions": runtime["versions"],
        "dtype": "bfloat16",
        "matched_template_sha256": runtime["matched_template_sha256"],
        "engine_settings": engine_settings,
        "gpu_memory_utilization": 0.90,
    }
    with VLLMEngine(spec, seed=config.seed) as engine:
        engine.tokenizer.chat_template = (root / "frozen/matched_chat_template.jinja").read_text()
        backend = VLLMBackend(engine, fingerprint, config.max_model_len)
        smoke = [
            {
                "id": "smoke:constraints",
                "group_id": "smoke:constraints",
                "cohort": "benign",
                "prompt": "Write exactly two numbered tips for keeping a library book dry. Use a calm librarian voice.",
            },
            {
                "id": "smoke:arithmetic",
                "group_id": "smoke:arithmetic",
                "cohort": "benign",
                "prompt": "A box holds 7 blue marbles and 5 red marbles. How many marbles are in the box? Explain briefly.",
            },
        ]
        smoke_config = OrbitConfig(**{**asdict(config), "sampled_trajectories": 1})
        run_orbits(smoke, smoke_config, backend, root / "smoke", root / "token_budget.jsonl")
        smoke_events = read_jsonl(root / "smoke/events.jsonl")
        if len(smoke_events) != 20 or any(e["status"] != "ok" for e in smoke_events):
            raise RuntimeError("32B smoke gate failed; inspect events before changing any settings")
        manifest = json.loads((root / "frozen/manifest.json").read_text())
        run_orbits(manifest, config, backend, root / "main_32b", root / "token_budget.jsonl")
        fixed_inverse(root, config, backend)


def orchestrate(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(root)
        runtime = json.loads((root / "preflight.json").read_text())
        snapshot = Path(runtime["models"]["target"]["snapshot"])
        shards = set(
            json.loads((snapshot / "model.safetensors.index.json").read_text())[
                "weight_map"
            ].values()
        )
        deadline = time.monotonic() + 4 * 3600
        while any(not (snapshot / s).is_file() for s in shards):
            write_json(
                root / "study_progress.json",
                {
                    "status": "waiting_for_pinned_weights",
                    "pid": os.getpid(),
                    "available_shards": sum((snapshot / s).is_file() for s in shards),
                    "total_shards": len(shards),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if time.monotonic() > deadline:
                raise TimeoutError("Weight download did not finish within four hours")
            time.sleep(10)
        protocol = json.loads((root / "frozen/protocol.json").read_text())
        shared = [
            "--run-dir",
            str(root),
            "--manifest",
            str(root / "frozen/manifest.json"),
            "--events-dir",
            str(root / "main_32b"),
        ]
        stages = [
            ("generation", "run_size_comparison.py", ["--run-dir", str(root), "--generate"]),
            (
                "validity",
                "summarize.py",
                [
                    "--manifest",
                    str(root / "frozen/manifest.json"),
                    "--events-dir",
                    str(root / "main_32b"),
                ],
            ),
            ("embeddings", "embed_orbits.py", [*shared, "--device", "cuda", "--batch-size", "256"]),
            ("benign", "score_benign.py", [*shared, "--code-image", protocol["code_image"]]),
            ("strongreject", "score_harmful.py", [*shared, "--judge", "strongreject"]),
            ("harmbench", "score_harmful.py", [*shared, "--judge", "harmbench"]),
            (
                "paired_analysis",
                "analyze_size_comparison.py",
                ["--run-dir", str(root), "--device", "cuda"],
            ),
        ]
        for name, script, args in stages:
            verify(root)
            marker = root / "markers" / f"{name}.json"
            if marker.exists():
                continue
            command = [sys.executable, str(REPO / "scripts/orbits" / script), *args]
            record = {
                "status": "running",
                "stage": name,
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "command": command,
            }
            write_json(root / "study_progress.json", record)
            log = root / "logs" / f"{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"Starting {name}", flush=True)
            with log.open("a") as output:
                worker = subprocess.Popen(
                    command,
                    cwd=REPO,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                record["worker_pid"] = worker.pid
                write_json(root / "study_progress.json", record)
                try:
                    result = worker.wait()
                except BaseException:
                    os.killpg(worker.pid, signal.SIGTERM)
                    try:
                        worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid, signal.SIGKILL)
                    raise
            record.update(
                returncode=result,
                finished_at=datetime.now(timezone.utc).isoformat(),
                status="complete" if result == 0 else "failed",
            )
            append_jsonl(root / "execution_history.jsonl", [record])
            if result:
                write_json(root / "study_progress.json", record)
                raise RuntimeError(f"{name} failed; see {log}")
            write_json(marker, record)
        write_json(
            root / "study_progress.json",
            {
                "status": "complete",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "report": str(root / "comparison_report.md"),
                "scientific_status": "Paired checkpoint comparison with retention proxies; no causal parameter-count claim.",
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.generate:
        verify(args.run_dir)
        generate(args.run_dir)
    else:
        try:
            orchestrate(args.run_dir)
        except Exception as exc:
            path = args.run_dir / "study_progress.json"
            record = json.loads(path.read_text()) if path.exists() else {}
            write_json(
                path,
                {
                    **record,
                    "status": "failed",
                    "error": str(exc),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise


if __name__ == "__main__":
    main()
