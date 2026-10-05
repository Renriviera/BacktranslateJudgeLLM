#!/usr/bin/env bash
# Run the 34 legacy false-positive stress cases and 8 false-negative cases together.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
BASE_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
RUN_ID="content-first-pathology-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$BASE_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/logs" "$BASE_DIR/logs/semantic-content-audit"

python3 04_Scripts_Experiments/scripts/semantic_reconstruction/prepare_pathological_cases.py
cp "$BASE_DIR/manifests/pathological_cases.jsonl" "$RUN_DIR/pathological_false_positive_cases.jsonl"
python3 04_Scripts_Experiments/scripts/semantic_reconstruction/prepare_pathological_false_negatives.py \
  --output "$RUN_DIR/pathological_false_negative_cases.jsonl"

python3 - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
combined = []
for filename, audit_set in (
    ("pathological_false_positive_cases.jsonl", "false_positive_stress"),
    ("pathological_false_negative_cases.jsonl", "false_negative_stress"),
):
    for line in (run_dir / filename).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["audit_set"] = audit_set
        combined.append(row)
if len(combined) != 42:
    raise SystemExit(f"Expected 42 cases (34 + 8), got {len(combined)}")
(run_dir / "cases.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined)
)
(run_dir / "run.json").write_text(json.dumps({
    "study": "content_first_pathological_audit",
    "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "false_positive_stress_cases": 34,
    "false_negative_stress_cases": 8,
    "total_cases": 42,
    "model": __import__("os").environ.get("SEMREC_MODEL", "Qwen/Qwen3.5-9B"),
    "k": int(__import__("os").environ.get("SEMREC_K", "3")),
    "protocol": "content-first; all cases reconstructed, refusal classifier is non-gating",
    "status": "prepared",
}, indent=2) + "\n")
print(f"Prepared {len(combined)} cases in {run_dir}")
PY

JOB_ID="$(sbatch --parsable --export="ALL,SEMREC_RUN_DIR=$RUN_DIR,SEMREC_CASES=$RUN_DIR/cases.jsonl" \
  --array=1-42%4 04_Scripts_Experiments/scripts/semantic_reconstruction/pathological_false_negative_a100.sbatch)"
JOB_ID="${JOB_ID%%;*}"
.venv/bin/python - "$RUN_DIR/run.json" "$JOB_ID" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
d["slurm_job_id"] = sys.argv[2]
d["status"] = "submitted"
p.write_text(json.dumps(d, indent=2) + "\n")
PY
echo "Submitted 42 content-first pathology cases as Slurm job $JOB_ID. Run data: $RUN_DIR"
