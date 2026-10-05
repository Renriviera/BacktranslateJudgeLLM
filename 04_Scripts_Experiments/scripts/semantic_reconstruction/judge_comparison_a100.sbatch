#!/usr/bin/env bash
# Five full 313-item PAP lanes; one global array cap enforces at most four A100 jobs.
# The task maps to one lane and one StrongREJECT behavior.
#SBATCH --job-name=pap-judge-compare
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-1565%4
#SBATCH --output=06_Results_Artifacts/new_runs/semantic_reconstruction/logs/pap-judge-compare-%A_%a.out
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
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
LANE_INDEX="$(((TASK_ID - 1) / 313))"
DATASET_INDEX="$(((TASK_ID - 1) % 313))"
BEHAVIOR_TASK_ID="$((DATASET_INDEX + 1))"
BEHAVIOR_ID="strongreject:${DATASET_INDEX}"
SEED="$((235711 + BEHAVIOR_TASK_ID))"

case "$LANE_INDEX" in
  0) LANE="qwen35_9b_strongreject_rubric"; JUDGE_MODEL="Qwen/Qwen3.5-9B"; MODE="rubric" ;;
  1) LANE="gemma4_12b_strongreject_rubric"; JUDGE_MODEL="google/gemma-4-12B-it"; MODE="rubric" ;;
  2) LANE="gemma4_12b_semantic_reconstruction"; JUDGE_MODEL="google/gemma-4-12B-it"; MODE="semantic" ;;
  3) LANE="ministral3_8b_strongreject_rubric"; JUDGE_MODEL="mistralai/Ministral-3-8B-Instruct-2512-BF16"; MODE="rubric" ;;
  4) LANE="ministral3_8b_semantic_reconstruction"; JUDGE_MODEL="mistralai/Ministral-3-8B-Instruct-2512-BF16"; MODE="semantic" ;;
  *) echo "Invalid lane index: $LANE_INDEX" >&2; exit 2 ;;
esac

OUT_DIR="$RUN_DIR/$LANE"
mkdir -p "$OUT_DIR"
PAP_CACHE="$OUT_DIR/pap_strongreject_${DATASET_INDEX}_${SLURM_ARRAY_JOB_ID}.json"
PAP_REPORT="$OUT_DIR/pap_strongreject_${DATASET_INDEX}_${SLURM_ARRAY_JOB_ID}.report.json"

.venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_pap.py \
  --dataset strongreject \
  --technique misrepresentation \
  --behavior-id "$BEHAVIOR_ID" \
  --seed 235711 \
  --attacker-model Qwen/Qwen3-32B \
  --validity-model Qwen/Qwen3-32B \
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

if [[ "$MODE" == "semantic" ]]; then
  .venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/pilot.py \
    --model "$JUDGE_MODEL" \
    --target-model allenai/Olmo-3-7B-Instruct \
    --pap-report "$PAP_REPORT" \
    --source-label "pap_misrepresentation_${BEHAVIOR_ID}" \
    --target-max-tokens 8192 \
    --aux-max-tokens 8192 \
    --judge-max-tokens 8192 \
    --max-model-len 16384 \
    --k 3 \
    --seed "$SEED" \
    --output "$OUT_DIR/pap_semantic_${DATASET_INDEX}_${SLURM_ARRAY_JOB_ID}.json"
else
  .venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/strongreject_rubric_pilot.py \
    --pap-report "$PAP_REPORT" \
    --model "$JUDGE_MODEL" \
    --model-label "$LANE" \
    --target-model allenai/Olmo-3-7B-Instruct \
    --seed "$SEED" \
    --threshold 0.5 \
    --target-max-tokens 8192 \
    --max-model-len 16384 \
    --output "$OUT_DIR/pap_rubric_${DATASET_INDEX}_${SLURM_ARRAY_JOB_ID}.json"
fi
