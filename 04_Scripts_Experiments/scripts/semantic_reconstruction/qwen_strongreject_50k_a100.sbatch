#!/usr/bin/env bash
# Score original semantic-pipeline PAP target responses with Qwen and a 50k generation cap.
#SBATCH --job-name=pap-qwen-sr-50k
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-313%4
#SBATCH --output=06_Results_Artifacts/new_runs/semantic_reconstruction/logs/qwen-sr-50k-%A_%a.out
#SBATCH --chdir=.

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$REPO_ROOT"
module load python3
module load cuda13.0/toolkit/13.0.2

if [[ ! -x .venv/bin/python || ! -f .env ]]; then
  echo "Missing .venv or repository .env." >&2
  exit 2
fi
set -a
source .env
set +a
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is unset in .env." >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

RUN_DIR="${PAP_JUDGE_RUN_DIR:?PAP_JUDGE_RUN_DIR is not set}"
MANIFEST="${PAP_JUDGE_SOURCE_MANIFEST:?PAP_JUDGE_SOURCE_MANIFEST is not set}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
SOURCE_RESULT="$(sed -n "${TASK_ID}p" "$MANIFEST")"
if [[ -z "$SOURCE_RESULT" || ! -f "$SOURCE_RESULT" ]]; then
  echo "Missing source result for task $TASK_ID." >&2
  exit 2
fi
BASE="$(basename "$SOURCE_RESULT")"
DATASET_INDEX="${BASE#pap_rubric_}"
DATASET_INDEX="${DATASET_INDEX%%_*}"
SEED="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$SOURCE_RESULT")"
OUT_DIR="$RUN_DIR/qwen35_9b_strongreject_rubric"
OUTPUT="$OUT_DIR/pap_rubric_${DATASET_INDEX}_${SLURM_ARRAY_JOB_ID}.json"
mkdir -p "$OUT_DIR"

if ! .venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/strongreject_rubric_pilot.py \
  --source-result "$SOURCE_RESULT" \
  --model Qwen/Qwen3.5-9B \
  --model-label qwen35_9b_strongreject_rubric \
  --seed "$SEED" \
  --threshold 0.5 \
  --judge-max-tokens 50000 \
  --max-model-len 65536 \
  --output "$OUTPUT"; then
  .venv/bin/python - "$SOURCE_RESULT" "$OUTPUT" <<'PY'
import json,sys
from pathlib import Path
src,out=map(Path,sys.argv[1:])
d=json.loads(src.read_text())
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"method":"strongreject_exact_template_expected_digit_score","status":"failed","source_result":str(src),"original_prompt":d.get("original_prompt"),"pap_status":d.get("pap_status"),"attack_success":False,"failure":"See matching Slurm log."},indent=2)+"\n")
PY
fi
