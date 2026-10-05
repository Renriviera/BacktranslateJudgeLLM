#!/usr/bin/env bash
# Matched Gemma 4 rerun: direct StrongREJECT rubric and semantic pipeline.
# Both lanes reuse the saved PAP prompts/statuses and target responses.
#SBATCH --job-name=pap-gemma4-judge
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=1-626%4
#SBATCH --output=06_Results_Artifacts/new_runs/semantic_reconstruction/logs/gemma4-judge-%A_%a.out
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
ITEM_COUNT="${PAP_JUDGE_ITEM_COUNT:-313}"
if (( TASK_ID <= ITEM_COUNT )); then
  MODE="strongreject"
  SOURCE_LINE="$TASK_ID"
else
  MODE="semantic"
  SOURCE_LINE="$((TASK_ID - ITEM_COUNT))"
fi
SOURCE_RESULT="$(sed -n "${SOURCE_LINE}p" "$MANIFEST")"
if [[ -z "$SOURCE_RESULT" || ! -f "$SOURCE_RESULT" ]]; then
  echo "Missing source result for task $TASK_ID (manifest line $SOURCE_LINE)." >&2
  exit 2
fi
BASE="$(basename "$SOURCE_RESULT")"
DATASET_INDEX="${BASE#pap_rubric_}"
DATASET_INDEX="${DATASET_INDEX%%_*}"
SEED="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$SOURCE_RESULT")"

if [[ "$MODE" == "strongreject" ]]; then
  OUT_DIR="$RUN_DIR/gemma4_26b_a4b_strongreject_rubric"
  OUTPUT="$OUT_DIR/pap_rubric_${DATASET_INDEX}_${SLURM_ARRAY_JOB_ID}.json"
  mkdir -p "$OUT_DIR"
  if ! .venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/strongreject_rubric_pilot.py \
    --source-result "$SOURCE_RESULT" \
    --model google/gemma-4-26B-A4B-it \
    --model-label gemma4_26b_a4b_strongreject_rubric \
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
out.write_text(json.dumps({"method":"strongreject_exact_template_expected_digit_score","status":"failed","source_result":str(src),"original_prompt":d.get("original_prompt"),"pap_status":d.get("pap_status"),"target_response":d.get("target_response"),"attack_success":False,"failure":"See matching Slurm log."},indent=2,ensure_ascii=False)+"\n")
PY
  fi
else
  OUT_DIR="$RUN_DIR/gemma4_26b_a4b_semantic_reconstruction"
  OUTPUT="$OUT_DIR/pap_semantic_${DATASET_INDEX}_${SLURM_ARRAY_JOB_ID}.json"
  mkdir -p "$OUT_DIR"
  if ! .venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/pilot.py \
    --source-result "$SOURCE_RESULT" \
    --model google/gemma-4-26B-A4B-it \
    --source-label "matched_prior_pap_${DATASET_INDEX}" \
    --target-model allenai/Olmo-3-7B-Instruct \
    --aux-max-tokens 8192 \
    --judge-max-tokens 8192 \
    --max-model-len 16384 \
    --k 3 \
    --seed "$SEED" \
    --output "$OUTPUT"; then
    .venv/bin/python - "$SOURCE_RESULT" "$OUTPUT" <<'PY'
import json,sys
from pathlib import Path
src,out=map(Path,sys.argv[1:])
d=json.loads(src.read_text())
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"method":"semantic_reconstruction","status":"failed","source_result":str(src),"original_prompt":d.get("original_prompt"),"pap_status":d.get("pap_status"),"target_response":d.get("target_response"),"attack_success":False,"failure":"See matching Slurm log."},indent=2,ensure_ascii=False)+"\n")
PY
  fi
fi
