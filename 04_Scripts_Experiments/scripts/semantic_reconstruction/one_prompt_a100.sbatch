#!/usr/bin/env bash
# Full StrongREJECT PAP attack/reconstruction array, one A100 per behavior; max four concurrent.
# Submit from the repository root with submit_one_prompt.sh.
#SBATCH --job-name=pap-semantic-reconstruction
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-313%4
#SBATCH --output=06_Results_Artifacts/new_runs/semantic_reconstruction/logs/pap/%x-%A_%a.out
#SBATCH --chdir=.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$REPO_ROOT"

if ! command -v module >/dev/null 2>&1; then
  echo "Environment modules are unavailable; activate Python 3.12 before submission." >&2
  exit 2
fi
module load python3
module load cuda13.0/toolkit/13.0.2

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Set up the uv environment from README.md before submitting." >&2
  exit 2
fi

if [[ ! -f .env ]]; then
  echo "Missing repository .env containing HF_TOKEN." >&2
  exit 2
fi
# Export credentials to the child process without printing or tracing the environment.
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
mkdir -p 06_Results_Artifacts/new_runs/semantic_reconstruction/logs/pap

OUT_DIR="${SEMREC_RUN_DIR:-06_Results_Artifacts/new_runs/semantic_reconstruction/local}"
mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
DATASET_INDEX="$((TASK_ID - 1))"
BEHAVIOR_ID="strongreject:${DATASET_INDEX}"
PAP_CACHE="$OUT_DIR/pap_strongreject_${DATASET_INDEX}_${SLURM_ARRAY_JOB_ID:-local}.json"
PAP_REPORT="$OUT_DIR/pap_strongreject_${DATASET_INDEX}_${SLURM_ARRAY_JOB_ID:-local}.report.json"

if [[ -n "${SEMREC_PAP_REPORT:-}" ]]; then
  PAP_REPORT="$SEMREC_PAP_REPORT"
  if [[ ! -f "$PAP_REPORT" ]]; then
    echo "SEMREC_PAP_REPORT does not exist: $PAP_REPORT" >&2
    exit 2
  fi
  echo "Reusing PAP report: $PAP_REPORT"
else
  # Match the tracked PAP runner defaults used in the collaborator's earlier runs:
  # Qwen3-32B attacker, 512 generation tokens, and the OLMo-3-7B-Instruct target.
  .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_pap.py \
    --dataset strongreject \
    --technique misrepresentation \
    --behavior-id "$BEHAVIOR_ID" \
    --seed 235711 \
    --attacker-model "${SEMREC_ATTACKER_MODEL:-Qwen/Qwen3-32B}" \
    --model allenai/Olmo-3-7B-Instruct \
    --max-tokens 512 \
    --temperature 0.7 \
    --top-p 0.8 \
    --max-model-len 8192 \
    --gpu-mem 0.90 \
    --tp 1 \
    --dtype bfloat16 \
    --out "$PAP_CACHE" \
    --report "$PAP_REPORT"
fi

.venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/pilot.py \
  --model "${SEMREC_MODEL:-Qwen/Qwen3.5-9B}" \
  --target-model allenai/Olmo-3-7B-Instruct \
  --pap-report "$PAP_REPORT" \
  --source-label "pap_misrepresentation_strongreject_${BEHAVIOR_ID}" \
  --target-max-tokens 8192 \
  --aux-max-tokens 8192 \
  --judge-max-tokens 8192 \
  --max-model-len 16384 \
  --k "${SEMREC_K:-3}" \
  --seed "$((235711 + TASK_ID))" \
  --output "$OUT_DIR/pap_semantic_${DATASET_INDEX}_${SLURM_ARRAY_JOB_ID:-local}.json"
