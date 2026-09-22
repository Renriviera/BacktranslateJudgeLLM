#!/usr/bin/env python
"""Freeze the matched two-cycle size comparison before generating 32B outcomes."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.io import append_jsonl, file_digest, read_jsonl, write_json  # noqa: E402
from brass.orbits.runner import inverse_messages  # noqa: E402

OLD = REPO / "results/orbits/2026-09-18-pilot"
ROOT = REPO / "results/orbits/2026-09-19-size-32b"
MODEL_ID = "allenai/Olmo-3.1-32B-Instruct"
REV = "ac0587e4a7744a551c059d8cd17ba220bc940dae"


def main():
    from transformers import AutoTokenizer

    if (ROOT / "frozen/protocol.json").exists():
        raise RuntimeError("Already frozen; do not overwrite the protocol")
    frozen = ROOT / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OLD / "frozen/main_core_manifest.json", frozen / "manifest.json")
    manifest = json.loads((frozen / "manifest.json").read_text())
    config = json.loads((OLD / "frozen/main_core_config.json").read_text())
    config.update(round_trips=2, global_generated_token_limit=64_000_000)
    write_json(frozen / "config.json", config)
    old_runtime = json.loads((OLD / "preflight.json").read_text())
    snapshot = (
        Path("/mnt/data/hf_cache/hub")
        / ("models--" + MODEL_ID.replace("/", "--"))
        / "snapshots"
        / REV
    )
    t7 = AutoTokenizer.from_pretrained(
        old_runtime["models"]["target"]["snapshot"], local_files_only=True
    )
    t32 = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    native_template = t32.chat_template
    t32.chat_template = t7.chat_template
    (frozen / "matched_chat_template.jinja").write_text(t7.chat_template)
    (frozen / "native_32b_chat_template.jinja").write_text(native_template)
    events = [e for e in read_jsonl(OLD / "main_core/events.jsonl") if e["stage"] <= 4]
    messages = [[{"role": "user", "content": r["prompt"]}] for r in manifest]
    messages += [
        inverse_messages(
            'A test response with numbers 123 and JSON {"key": "value"}.',
            config["inverse_template"],
        )
    ]
    for m in messages:
        if t7.apply_chat_template(
            m, tokenize=True, add_generation_prompt=True, return_dict=False
        ) != t32.apply_chat_template(
            m, tokenize=True, add_generation_prompt=True, return_dict=False
        ):
            raise RuntimeError("Tokenization differs despite matched template")
    baseline = ROOT / "baseline_7b"
    baseline.mkdir(exist_ok=True)
    append_jsonl(baseline / "events.jsonl", events)
    ids = {e["id"] for e in events}
    write_json(
        baseline / "geometry.json",
        [r for r in json.loads((OLD / "main_core/geometry.json").read_text()) if r["id"] in ids],
    )
    for name in ["benign_scores.jsonl", "strongreject_scores.jsonl", "harmbench_scores.jsonl"]:
        append_jsonl(
            baseline / name, [e for e in read_jsonl(OLD / "main_core" / name) if e["id"] in ids]
        )
    shutil.copy2(OLD / "main_core/text_embeddings.npz", baseline / "text_embeddings.npz")
    shutil.copy2(
        OLD / "main_core/embedding_provenance.json", baseline / "embedding_provenance.json"
    )
    write_json(
        baseline / "provenance.json",
        {
            "reused_without_new_inference": True,
            "max_stage": 4,
            "source_run": str(OLD / "main_core"),
            "source_events_sha256": file_digest(OLD / "main_core/events.jsonl"),
            "source_model": old_runtime["models"]["target"],
            "source_spec_sha256": file_digest(OLD / "main_core/run_spec.json"),
            "n_events": len(events),
        },
    )
    # Reuse the exact versioned benchmark evaluators, including NLTK data.
    (ROOT / "evaluators").symlink_to(OLD / "evaluators", target_is_directory=True)
    runtime = dict(old_runtime)
    runtime["models"] = dict(old_runtime["models"])
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    runtime["models"]["target"] = {
        "hf_id": MODEL_ID,
        "revision": REV,
        "snapshot": str(snapshot),
        "metadata_hashes": {p.name: file_digest(p) for p in snapshot.glob("*.json") if p.is_file()},
        "total_parameters": index["metadata"]["total_parameters"],
    }
    runtime["timestamp"] = datetime.now(timezone.utc).isoformat()
    runtime["versions"] = {k: importlib.metadata.version(k) for k in old_runtime["versions"]}
    runtime["effective_forward_template_example"] = t32.apply_chat_template(
        messages[0], tokenize=False, add_generation_prompt=True
    )
    # Avoid putting an actual attack prompt in an otherwise shareable preflight field.
    runtime["effective_forward_template_example"] = t32.apply_chat_template(
        [{"role": "user", "content": "Orbit preflight."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    runtime["matched_template_sha256"] = file_digest(frozen / "matched_chat_template.jinja")
    runtime["template_tokenization_checks"] = len(messages)
    runtime["old_versions_match"] = runtime["versions"] == old_runtime["versions"]
    write_json(ROOT / "preflight.json", runtime)
    frame_count = sum(r.get("control_type") == "benign_framed" for r in manifest)
    primary = [
        {"cohort": cohort, "metric": metric}
        for cohort in ["attack", "benign"]
        for metric in ["response_drift", "prompt_drift"]
    ] + [
        {"cohort": "attack", "metric": "content_similarity"},
        {"cohort": "control", "subgroup": "benign_framed", "metric": "frame_lexical_recall"},
        {"cohort": "control", "subgroup": "benign_framed", "metric": "frame_similarity"},
    ]
    protocol = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(ROOT),
        "question": "Does the larger checkpoint preserve prompt content/framing better over two response-only inversion cycles?",
        "n_prompts": len(manifest),
        "cohorts": dict(Counter(r["cohort"] for r in manifest)),
        "sampled_trajectories": 8,
        "greedy_trajectories": 1,
        "round_trips": 2,
        "primary_arm": "32B forward and 32B inverse; same 491 prompts and per-call seeds as the 7B reference",
        "fixed_response_diagnostic": "32B inverse on the identical saved 7B forward responses at stages 0 and 2, paired with original 7B inversions; no extra recursive cycles",
        "sampling": "F temperature 1.0, G 0.7, top_p 1; greedy separate; BF16; F cap 4096, G cap 512; context 8192",
        "chat_control": "Use pinned 7B chat template and eos_token_id [100265,100257] for both models; user messages and inverse wording unchanged",
        "primary_tests": primary,
        "primary_step": 2,
        "primary_mode": "sampled",
        "inference": "Pair trajectory IDs, average within prompt then underlying analysis_group_id; 10000 paired task bootstrap samples; paired t-tests; Holm correction across seven primary tests. Same seeds do not imply equivalent random draws across models.",
        "fixed_response_tests": "Separate Holm family at first inversion: content similarity for attacks and benign; frame lexical recall and similarity for known benign wrappers. Second inversion and greedy are descriptive.",
        "retention": "Content is archived unwrapped behavior for attacks, original task for benign, and exact parent task for benign framing controls. Framing is the exact 40 known benign wrapper spans; frame word recall excludes words in core task. No claimed gold framing decomposition for PAP/PAIR/SlotGCG.",
        "failures": "Invalid/truncated/missing states are censored, never convergence; report all-planned-path coverage and paired complete-case comparisons. Frame/content metrics can use valid inversion even when subsequent forward fails.",
        "subgroups": "Report every attack family and all eight benign benchmarks; controls separately. All archived-positive prompts retained regardless of 32B transfer success. Refusal/quality labels and judge coverage are sensitivity diagnostics, not filtering criteria.",
        "limits": [
            "7B vs 32B checkpoints differ in training and architecture, so not a causal parameter-count estimate",
            "Response-only inversion cannot recover framing absent from the response",
            "Cosine similarity and lexical recall are proxies, not entropy or semantic entailment",
            "Attack successes selected on 7B; transfer to 32B not guaranteed",
            "Two cycles cannot establish stationarity or asymptotic stability",
        ],
        "new_generated_token_ceiling": 64_000_000,
        "new_judge_call_ceiling": 200_000,
        "main_worst_case_tokens": 491 * 9 * (3 * 4096 + 2 * 512),
        "fixed_inverse_worst_case_tokens": 491 * 9 * 2 * 512,
        "n_known_framing_controls": frame_count,
        "engine": {
            "gpu_memory_utilization": 0.90,
            "max_num_seqs": 64,
            "enforce_eager": True,
            "dtype": "bfloat16",
        },
        "model_source": "https://huggingface.co/allenai/Olmo-3.1-32B-Instruct",
        "code_image": json.loads((OLD / "execution_queue_recovery_v1.json").read_text())[
            "code_image"
        ],
    }
    write_json(frozen / "protocol.json", protocol)
    files = list(frozen.iterdir()) + [ROOT / "preflight.json"] + list(baseline.glob("*"))
    source_files = (
        list((REPO / "src/brass/orbits").glob("*.py"))
        + list((REPO / "scripts/orbits").glob("*.py"))
        + [REPO / "src/brass/serving/vllm_engine.py"]
    )
    for p in source_files:
        dest = ROOT / "source_snapshot" / p.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    write_json(
        frozen / "hashes.json",
        {str(p): file_digest(p) for p in files + source_files if p.is_file()},
    )
    print(
        json.dumps(
            {
                "run_dir": str(ROOT),
                "n_prompts": len(manifest),
                "baseline_states": len(events),
                "tokenization_checks": len(messages),
                "primary_tests": len(primary),
                "protocol_sha256": file_digest(frozen / "protocol.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
