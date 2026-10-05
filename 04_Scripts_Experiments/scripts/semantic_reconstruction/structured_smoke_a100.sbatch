#!/usr/bin/env bash
# Structured-output smoke test: one backbone per array task over a fixed case subset.
# Submit with submit_structured_smoke.sh from the repository root.
#SBATCH --job-name=semrec-structured-smoke
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --chdir=.

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$REPO_ROOT"
module load python3
module load cuda13.0/toolkit/13.0.2

RUN_DIR="${SEMREC_RUN_DIR:?SEMREC_RUN_DIR is not set}"
PYTHON_BIN="$(cat "$RUN_DIR/python.txt")"
if [[ ! -x "$PYTHON_BIN" || ! -f .env ]]; then
  echo "Missing $PYTHON_BIN or repository .env." >&2
  exit 2
fi
set -a
source .env
set +a
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUN_DIR/models.txt")"
LABEL="$(basename "$MODEL")"
"$PYTHON_BIN" 04_Scripts_Experiments/scripts/semantic_reconstruction/semrec_structured.py \
  --cases "$RUN_DIR/cases.jsonl" \
  --case-indices "$(cat "$RUN_DIR/case_indices.txt")" \
  --model "$MODEL" \
  --k 3 \
  --audit diagnostic \
  --output "$RUN_DIR/${LABEL}.jsonl"
