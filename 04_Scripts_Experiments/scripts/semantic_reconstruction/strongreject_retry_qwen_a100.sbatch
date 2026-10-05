#!/usr/bin/env bash
# Retry only saved responses whose first Qwen StrongREJECT judgment did not yield a score.
#SBATCH --job-name=sr-qwen-retry
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-20%4
#SBATCH --output=06_Results_Artifacts/new_runs/iterative_attacks/logs/sr-qwen-retry-%A_%a.out
#SBATCH --chdir=.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
module load python3
module load cuda13.0/toolkit/13.0.2
if [[ ! -x .venv/bin/python || ! -f .env ]]; then
  echo "Missing .venv or .env." >&2
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

RUN_DIR="${SR_RETRY_RUN_DIR:?SR_RETRY_RUN_DIR is not set}"
MANIFEST="${SR_RETRY_MANIFEST:?SR_RETRY_MANIFEST is not set}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
SOURCE_RESULT="$(sed -n "${TASK_ID}p" "$MANIFEST")"
if [[ -z "$SOURCE_RESULT" || ! -f "$SOURCE_RESULT" ]]; then
  echo "Missing saved attack result for retry task $TASK_ID." >&2
  exit 2
fi
BASE="$(basename "$SOURCE_RESULT" .json)"
ATTACK="${BASE%%_*}"
BEHAVIOR="${BASE#*_}"
SEED="$((235711 + BEHAVIOR))"
OUTPUT="$RUN_DIR/$ATTACK/behavior_$BEHAVIOR/strongreject_qwen35_9b_retry_100k.json"
mkdir -p "$(dirname "$OUTPUT")"

.venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/strongreject_rubric_pilot.py \
  --source-result "$SOURCE_RESULT" \
  --model Qwen/Qwen3.5-9B \
  --model-label qwen35_9b_strongreject_rubric \
  --seed "$SEED" \
  --threshold 0.5 \
  --judge-max-tokens 100000 \
  --max-model-len 131072 \
  --no-pap-gate \
  --output "$OUTPUT"
