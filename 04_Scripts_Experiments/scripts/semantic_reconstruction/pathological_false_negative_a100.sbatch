#!/usr/bin/env bash
# Audit one saved pathological false-negative response on one A100.
# Submit with submit_pathological_false_negatives.sh.
#SBATCH --job-name=semantic-content-audit
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-42%4
#SBATCH --output=06_Results_Artifacts/new_runs/semantic_reconstruction/logs/semantic-content-audit/%x-%A_%a.out
#SBATCH --chdir=.

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$REPO_ROOT"
module load python3
module load cuda13.0/toolkit/13.0.2
if [[ ! -x .venv/bin/python || ! -f .env ]]; then
  echo "Set up the uv environment and repository .env before submitting." >&2
  exit 2
fi
set -a
source .env
set +a
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is unset in .env; refusing to start model download." >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

RUN_DIR="${SEMREC_RUN_DIR:?SEMREC_RUN_DIR is required}"
mkdir -p "$RUN_DIR/logs" 06_Results_Artifacts/new_runs/semantic_reconstruction/logs/semantic-content-audit
CASES="${SEMREC_CASES:-$RUN_DIR/pathological_false_negative_cases.jsonl}"
.venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/pathological_stress_adversarial.py \
  --cases "$CASES" \
  --case-index "$SLURM_ARRAY_TASK_ID" \
  --model "${SEMREC_MODEL:-Qwen/Qwen3.5-9B}" \
  --k "${SEMREC_K:-3}" \
  --seed 20260927 \
  --output "$RUN_DIR/case_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.jsonl"
