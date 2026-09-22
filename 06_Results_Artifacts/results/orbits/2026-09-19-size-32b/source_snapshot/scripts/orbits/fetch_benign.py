#!/usr/bin/env python
"""Download pinned public benchmark sources and normalize evaluation-only fields."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brass.orbits.io import digest, file_digest, write_json  # noqa: E402

HF = {
    "gsm8k": ("openai/gsm8k", "main", "test", "740312add88f781978c0658806c59bc2815b9866"),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "test", "210d026faf9955653af8916fad021475a3f00453"),
    "squad": ("rajpurkar/squad", None, "validation", "7b6d24c440a36b6815f21b70d25016731768db1f"),
    "ifeval": ("google/IFEval", None, "train", "966cd89545d6b6acfd7638bc708b98261ca58e84"),
    "dolly": (
        "databricks/databricks-dolly-15k",
        None,
        "train",
        "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
    ),
    "orbench": (
        "bench-llm/or-bench",
        "or-bench-hard-1k",
        "train",
        "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272",
    ),
}
BBH_REV = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
XSTEST_REV = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
BBH_TASKS = [
    "logical_deduction_three_objects",
    "tracking_shuffled_objects_three_objects",
    "date_understanding",
    "boolean_expressions",
    "object_counting",
    "word_sorting",
]


def record(source, index, prompt, evaluation, *, stratum=None, group=None, category=None):
    key = f"{source}:{index}"
    return {
        "id": key,
        "group_id": group or key,
        "cohort": "benign",
        "source": source,
        "stratum": stratum or source,
        "category": category,
        "prompt": prompt,
        "prompt_sha256": digest(prompt),
        "evaluation": evaluation,
        "label_status": "benchmark_benign_pending_content_audit",
    }


def main():
    import requests
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.output / "sources"
    raw.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    rows, provenance = [], {}
    for source, (repo, name, split, revision) in HF.items():
        saved = raw / f"{source}.json"
        if saved.exists():
            data = json.loads(saved.read_text())
        else:
            ds = load_dataset(
                repo, name, split=split, revision=revision, cache_dir=str(raw / "hf_datasets")
            )
            data = [dict(r) for r in ds]
            write_json(saved, data)
        provenance[source] = {
            "repo": repo,
            "config": name,
            "split": split,
            "revision": revision,
            "source_sha256": file_digest(saved),
            "n_source": len(data),
        }
        for i, r in enumerate(data):
            if source == "gsm8k":
                rows.append(
                    record(
                        source,
                        i,
                        r["question"],
                        {
                            "kind": "numeric",
                            "answer": r["answer"].split("####")[-1].strip(),
                            "reference": r["answer"],
                        },
                    )
                )
            elif source == "arc":
                choices = "\n".join(
                    f"{a}. {b}"
                    for a, b in zip(r["choices"]["label"], r["choices"]["text"], strict=True)
                )
                prompt = (
                    r["question"] + "\n\n" + choices + "\nGive your answer with its option label."
                )
                rows.append(
                    record(
                        source,
                        r["id"],
                        prompt,
                        {"kind": "choice", "answer": r["answerKey"], "choices": r["choices"]},
                    )
                )
            elif source == "squad":
                prompt = (
                    "Read the passage and answer the question.\n\nPassage:\n"
                    + r["context"]
                    + "\n\nQuestion: "
                    + r["question"]
                )
                rows.append(
                    record(
                        source,
                        r["id"],
                        prompt,
                        {"kind": "qa", "answers": r["answers"]["text"], "context": r["context"]},
                        group="squad-passage:" + digest(r["context"]),
                    )
                )
            elif source == "ifeval":
                # Select length-feasible tasks before observing any target output.
                if any((kw.get("num_words") or 0) > 450 for kw in r["kwargs"]):
                    continue
                rows.append(
                    record(
                        source,
                        r.get("key", i),
                        r["prompt"],
                        {
                            "kind": "ifeval",
                            "instruction_id_list": r["instruction_id_list"],
                            "kwargs": r["kwargs"],
                        },
                    )
                )
            elif source == "dolly":
                if r["category"] not in ("creative_writing", "brainstorming", "open_qa"):
                    continue
                prompt = r["instruction"] + ("\n\n" + r["context"] if r.get("context") else "")
                rows.append(
                    record(
                        source,
                        i,
                        prompt,
                        {"kind": "rubric", "reference": r["response"]},
                        category=r["category"],
                    )
                )
            elif source == "orbench":
                rows.append(
                    record(
                        source,
                        i,
                        r["prompt"],
                        {"kind": "benign_safety"},
                        stratum="safety",
                        category=r.get("category"),
                    )
                )
        print(f"Loaded {source}: {len(data)} source items", flush=True)

    def download(name, url):
        path = raw / name
        if not path.exists():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        provenance[name] = {"url": url, "sha256": file_digest(path)}
        return path.read_bytes()

    for task in BBH_TASKS:
        url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{BBH_REV}/bbh/{task}.json"
        data = json.loads(download(f"bbh_{task}.json", url))
        for i, r in enumerate(data["examples"]):
            rows.append(
                record(
                    "bbh",
                    f"{task}:{i}",
                    r["input"],
                    {"kind": "bbh", "answer": r["target"], "task": task},
                    category=task,
                )
            )
    data = download(
        "xstest.csv",
        f"https://raw.githubusercontent.com/paul-rottger/xstest/{XSTEST_REV}/xstest_prompts.csv",
    ).decode()
    for i, r in enumerate(csv.DictReader(io.StringIO(data))):
        # Official XSTest uses 'contrast_*' type names for unsafe contrasts.
        if r["type"].startswith("contrast"):
            continue
        rows.append(
            record(
                "xstest",
                r.get("id", i),
                r["prompt"],
                {"kind": "benign_safety"},
                stratum="safety",
                category=r["type"],
            )
        )
    url = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"
    data = gzip.decompress(download("HumanEvalPlus-v0.1.10.jsonl.gz", url)).decode()
    for line in data.splitlines():
        r = json.loads(line)
        # Tests and canonical solutions are never exposed to the generation runner.
        rows.append(
            record(
                "humaneval",
                r["task_id"],
                "Complete the following Python function. Return the complete function in a Python code block.\n\n"
                + r["prompt"],
                {"kind": "code", "task_id": r["task_id"], "spec": r},
            )
        )
    assert len({r["id"] for r in rows}) == len(rows)
    write_json(args.output / "benign_inventory.json", rows)
    write_json(args.output / "dataset_provenance.json", provenance)
    from collections import Counter

    print(json.dumps(dict(Counter(r["stratum"] for r in rows)), indent=2))


if __name__ == "__main__":
    main()
