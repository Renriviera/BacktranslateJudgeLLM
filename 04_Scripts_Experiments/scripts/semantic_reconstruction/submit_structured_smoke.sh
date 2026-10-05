#!/usr/bin/env bash
# Run the structured-output pipeline on a few pathological cases across judge backbones.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
BASE_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
SOURCE_CASES="$BASE_DIR/content-first-pathology-v2-20260927T185900Z/cases.jsonl"
CASE_INDICES="${SEMREC_CASE_INDICES:-3,10,13,24,26,34,35,38}"
RUN_DIR="$BASE_DIR/structured-smoke${SEMREC_LABEL:+-$SEMREC_LABEL}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR/logs"
cp "$SOURCE_CASES" "$RUN_DIR/cases.jsonl"
# sbatch --export splits on commas, so the index list travels as a file.
echo "$CASE_INDICES" > "$RUN_DIR/case_indices.txt"
# Override with SEMREC_MODELS (space-separated) and SEMREC_PYTHON (interpreter path).
printf '%s\n' ${SEMREC_MODELS:-Qwen/Qwen3.5-9B mistralai/Ministral-3-8B-Instruct-2512-BF16 allenai/Olmo-3-7B-Instruct} \
  > "$RUN_DIR/models.txt"
# Absolute path without resolving symlinks: realpath would escape the venv.
PYTHON_REL="${SEMREC_PYTHON:-.venv/bin/python}"
PYTHON_BIN="$(cd "$(dirname "$PYTHON_REL")" && pwd)/$(basename "$PYTHON_REL")"
echo "$PYTHON_BIN" > "$RUN_DIR/python.txt"
N_MODELS="$(wc -l < "$RUN_DIR/models.txt" | tr -d ' ')"

JOB_ID="$(sbatch --parsable --array="1-${N_MODELS}%${SEMREC_CONCURRENCY:-4}" \
  --output="$RUN_DIR/logs/smoke-%A_%a.out" \
  --export="ALL,SEMREC_RUN_DIR=$RUN_DIR" \
  04_Scripts_Experiments/scripts/semantic_reconstruction/structured_smoke_a100.sbatch)"
JOB_ID="${JOB_ID%%;*}"

python3 - "$RUN_DIR" "$JOB_ID" "$CASE_INDICES" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
run = Path(sys.argv[1])
(run / "run.json").write_text(json.dumps({
    "study": "structured_output_smoke",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": sys.argv[2],
    "status": "submitted",
    "protocol": "v3_structured_reason_first",
    "source_cases": "content-first-pathology-v2-20260927T185900Z/cases.jsonl",
    "case_indices": [int(x) for x in sys.argv[3].split(",")],
    "models": (run / "models.txt").read_text().split(),
    "python": (run / "python.txt").read_text().strip(),
    "k": 3,
    "audit_mode": "diagnostic",
    "comparison_baseline": "legacy v2 outputs case_<i>_1731322.jsonl (Qwen3.5-9B, unconstrained JSON)",
}, indent=2) + "\n")
PY
echo "Submitted structured smoke array $JOB_ID. Results: $RUN_DIR"
