#!/usr/bin/env bash
# Recover target-only responses for the 12 PAP cases lost after downstream pilot errors.
#SBATCH --job-name=pap-response-recovery
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=06_Results_Artifacts/new_runs/semantic_reconstruction/logs/pap-recovery/%x-%j.out
#SBATCH --chdir=.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
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
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
mkdir -p 06_Results_Artifacts/new_runs/semantic_reconstruction/logs/pap-recovery
.venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/recover_pap_responses.py --run-id 1729137
