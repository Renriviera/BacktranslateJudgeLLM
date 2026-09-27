#!/usr/bin/env bash
# One unified job: optimize all TAO prompts, then classify them on the same two-A100 allocation.
# TAO uses GPU 0; after its cache is complete, two local workers split classifier items across GPUs.
#SBATCH --job-name=tao-strongreject-full
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=06_Results_Artifacts/new_runs/semantic_reconstruction/tao-full-%j.out
#SBATCH --chdir=.

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$REPO_ROOT"
module load python3
module load cuda13.0/toolkit/13.0.2
if [[ ! -x .venv/bin/python || ! -f .env ]]; then
  echo "Missing shared .venv or repository .env containing HF_TOKEN." >&2
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
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-/tmp/$USER}/uv-cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OUT_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$OUT_DIR"

RUN_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
OUT_CACHE="$OUT_DIR/tao_cache_${RUN_ID}.json"
ALLOCATED_GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r GPU0 GPU1 _ <<< "$ALLOCATED_GPUS"
if [[ -z "${GPU0:-}" || -z "${GPU1:-}" ]]; then
  echo "Expected two allocated A100s; CUDA_VISIBLE_DEVICES=$ALLOCATED_GPUS" >&2
  exit 2
fi

echo "Starting TAO optimization on allocated GPU $GPU0; run id $RUN_ID."
# Match the tracked OLMo-3-7B StrongREJECT collaborator launcher:
# easy-to-hard initialization, behavior 186 seed (1000 steps), then 500 steps/behavior,
# batch size 512, one worker. Run-specific native output prevents collisions with old runs.
CUDA_VISIBLE_DEVICES="$GPU0" .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_tao.py \
  --attack-python "$REPO_ROOT/.venv/bin/python" \
  --dataset strongreject \
  --model allenai/Olmo-3-7B-Instruct \
  --run-tag "semantic_${RUN_ID}" \
  --max-behaviors 0 \
  --init-mode easy-to-hard \
  --seed-behavior-id 186 \
  --seed 235711 \
  --seed-steps 1000 \
  --num-steps 500 \
  --batch-size 512 \
  --topk 256 \
  --cl-threshold 1.0 \
  --temperature 0.5 \
  --alpha 0.2 \
  --beta 0.2 \
  --refusal-set-size 3 \
  --revert-after 3 \
  --success-judge local \
  --num-workers 1 \
  --gpu "$GPU0" \
  --out "$OUT_CACHE"

.venv/bin/python - "$OUT_CACHE" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
cache = json.loads(p.read_text())
ids = [k for k, v in cache.items() if k.startswith('strongreject:') and isinstance(v, dict) and v.get('attack') == 'tao']
if len(ids) != 313:
    raise SystemExit(f'TAO produced {len(ids)} canonical behavior entries; expected 313')
print(f'Validated TAO cache {p}: {len(ids)} canonical behavior entries')
PY

echo "TAO cache is ready; starting two classifier workers on GPUs $GPU0 and $GPU1."
( set -e; export CUDA_VISIBLE_DEVICES="$GPU0"; for task_id in $(seq 1 2 313); do bash 04_Scripts_Experiments/scripts/semantic_reconstruction/tao_evaluate_task.sh "$task_id" "$RUN_ID" "$OUT_CACHE"; done ) &
WORKER0=$!
( set -e; export CUDA_VISIBLE_DEVICES="$GPU1"; for task_id in $(seq 2 2 313); do bash 04_Scripts_Experiments/scripts/semantic_reconstruction/tao_evaluate_task.sh "$task_id" "$RUN_ID" "$OUT_CACHE"; done ) &
WORKER1=$!
wait "$WORKER0"
STATUS0=$?
wait "$WORKER1"
STATUS1=$?
set -e
if [[ "$STATUS0" -ne 0 || "$STATUS1" -ne 0 ]]; then
  echo "Classifier workers failed (GPU $GPU0 status=$STATUS0; GPU $GPU1 status=$STATUS1)." >&2
  exit 1
fi
echo "TAO optimization and full semantic classification completed for all 313 behaviors."
