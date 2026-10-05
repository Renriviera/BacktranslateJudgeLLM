#!/usr/bin/env bash
# Evaluate saved SlotGCG/PAIR pathological responses on one A100.
# Submit from the repository root with submit_pathological_stress.sh.
#SBATCH --job-name=semantic-recon-path
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-34%4
#SBATCH --output=06_Results_Artifacts/new_runs/semantic_reconstruction/logs/pathological/%x-%A_%a.out
#SBATCH --chdir=.

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$REPO_ROOT"

module load python3
module load cuda13.0/toolkit/13.0.2

if [[ ! -x .venv/bin/python || ! -f .env ]]; then
  echo "Set up the uv environment and repository .env before submission." >&2
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
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" 06_Results_Artifacts/new_runs/semantic_reconstruction/logs/pathological

OUT_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
CASES="$OUT_DIR/manifests/pathological_cases.jsonl"
.venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/pathological_stress.py \
  --cases "$CASES" \
  --case-index "$SLURM_ARRAY_TASK_ID" \
  --model "${SEMREC_MODEL:-Qwen/Qwen3.5-9B}" \
  --k "${SEMREC_K:-3}" \
  --seed 20260927 \
  --output "$OUT_DIR/pathological_case_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID:-local}.jsonl"
