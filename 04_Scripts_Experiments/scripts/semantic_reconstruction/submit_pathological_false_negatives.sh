#!/usr/bin/env bash
# Submit the curated PAP refusal-gate false-negative audit cases.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
BASE_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
RUN_ID="pathological-false-negatives-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$BASE_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/logs" "$BASE_DIR/logs/semantic-content-audit"
python3 04_Scripts_Experiments/scripts/semantic_reconstruction/prepare_pathological_false_negatives.py \
  --output "$RUN_DIR/pathological_false_negative_cases.jsonl"
cat > "$RUN_DIR/run.json" <<EOF
{
  "study": "pathological_false_negative_audit",
  "source_job_id": "1729137",
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "cases": 8,
  "model": "${SEMREC_MODEL:-Qwen/Qwen3.5-9B}",
  "k": ${SEMREC_K:-3},
  "status": "submitted"
}
EOF
JOB_ID="$(sbatch --parsable --export="ALL,SEMREC_RUN_DIR=$RUN_DIR" \
  --array=1-8%4 04_Scripts_Experiments/scripts/semantic_reconstruction/pathological_false_negative_a100.sbatch)"
JOB_ID="${JOB_ID%%;*}"
.venv/bin/python - "$RUN_DIR/run.json" "$JOB_ID" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
d["slurm_job_id"] = sys.argv[2]
p.write_text(json.dumps(d, indent=2) + "\n")
PY
echo "Submitted 8 pathological false-negative cases as Slurm job $JOB_ID. Run data: $RUN_DIR"
