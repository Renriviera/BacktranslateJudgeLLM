#!/usr/bin/env bash
# Run one attack on one benchmark behavior, then label the final saved response with the existing pipeline.
#SBATCH --job-name=iterative-attack
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=06_Results_Artifacts/new_runs/iterative_attacks/logs/%x-%A_%a.out
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

RUN_DIR="${ITER_ATTACK_RUN_DIR:?ITER_ATTACK_RUN_DIR is required}"
ATTACK="${ITER_ATTACK_NAME:?ITER_ATTACK_NAME is required}"
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TARGET_MODEL="${ITER_ATTACK_TARGET_MODEL:-allenai/Olmo-3-7B-Instruct}"
SEED="$((235711 + TASK + ${ITER_ATTACK_SEED_OFFSET:-0}))"
OUT="$RUN_DIR/$ATTACK/behavior_$TASK"
mkdir -p "$OUT"

if [[ "$ATTACK" == "igcg" ]]; then
  ICG_ARGS=()
  if [[ "${ITER_ATTACK_FORCE_OPTIMIZATION:-0}" == "1" ]]; then
    ICG_ARGS+=(--force)
  fi
  .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_igcg_one.py \
    --behavior-index "$TASK" --steps 100 --gpu 0 --model "$TARGET_MODEL" --output "$OUT" "${ICG_ARGS[@]}"
else
  .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_slotgcg.py \
    --dataset advbench --behavior-index "$TASK" --num-steps 100 --search-width 512 \
    --num-adv-string 20 --gpu 0 --seed "$SEED" \
    --data-dir 01_Datasets_Benchmarks/data/attacks --native-dir "$OUT/native" \
    --out "$OUT/adapter_cache.json"
  BEHAVIOR_FILE="../attacks/slotgcg/benchmarks/advbench-50/behaviors.csv"
  python_bin=".venv/bin/python"
  "$python_bin" - "$BEHAVIOR_FILE" "$TASK" "$OUT/native/test_cases/test_cases_individual_behaviors" "$OUT/final_result.json" <<'PY'
import csv,json,sys
from pathlib import Path
behaviors,task,native,out=Path(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4])
with behaviors.open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
row=rows[task-1]
bid=row["BehaviorID"]
tc=json.loads((native/bid/"test_cases.json").read_text())[bid][0]
result={"behavior_id":bid,"original_prompt":row["Behavior"],"attacked_prompt":tc,
        "target_response":None,"attack":"SlotGCG","dataset":"SlotGCG AdvBench-50"}
out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
PY
fi
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/generate_final_attack_response.py \
  --result "$OUT/final_result.json" --model "$TARGET_MODEL" --seed "$SEED" \
  --max-model-len 16384 --max-tokens 8192

if [[ "${ITER_ATTACK_CLASSIFIER:-semantic_reconstruction}" == "strongreject" ]]; then
  .venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/strongreject_rubric_pilot.py \
    --source-result "$OUT/final_result.json" --model Qwen/Qwen3.5-9B \
    --model-label qwen35_9b_strongreject_rubric --seed "$SEED" \
    --threshold 0.5 --judge-max-tokens "${ITER_ATTACK_JUDGE_TOKENS:-50000}" \
    --max-model-len "${ITER_ATTACK_MAX_MODEL_LEN:-65536}" --no-pap-gate \
    --output "$OUT/strongreject_qwen35_9b.json"
else
  .venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/pilot.py \
    --source-result "$OUT/final_result.json" --model Qwen/Qwen3.5-9B \
    --source-label "${ATTACK}_advbench_${TASK}" --target-model "$TARGET_MODEL" \
    --k 3 --seed "$SEED" --max-model-len 16384 --target-max-tokens 8192 \
    --aux-max-tokens 8192 --judge-max-tokens "${ITER_ATTACK_JUDGE_TOKENS:-8192}" \
    --output "$OUT/semantic_label.json"
fi
