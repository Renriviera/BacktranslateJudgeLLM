import json
import subprocess
import sys
from pathlib import Path


def test_unscored_long_response_is_not_counted_as_a_negative_label(tmp_path):
    manifest = [
        {"id": "a", "group_id": "g", "cohort": "attack", "prompt": "task", "behavior": "task"}
    ]
    event = {
        "id": "a|sample:0|0",
        "item_id": "a",
        "direction": "forward",
        "step": 0,
        "trajectory": "sample:0",
        "status": "ok",
        "text": "answer",
        "output_tokens": 1,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "run_spec.json").write_text(json.dumps({"config": {"round_trips": 0}}))
    (tmp_path / "events.jsonl").write_text(json.dumps(event) + "\n")
    (tmp_path / "strongreject_scores.jsonl").write_text(
        json.dumps({"id": event["id"], "score": 0.8}) + "\n"
    )
    (tmp_path / "harmbench_scores.jsonl").write_text(
        json.dumps({"id": event["id"], "score": None, "status": "judge_context_overflow"}) + "\n"
    )
    script = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir()) / "04_Scripts_Experiments/scripts/orbits/analyze.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--events-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
    )
    measurements = json.loads((tmp_path / "measurements.json").read_text())
    assert measurements[0]["original_task_success"] is None
    assert measurements[0]["initial_task_success"] is None
