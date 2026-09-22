#!/usr/bin/env bash
# Run the paper-configured TAO AdvBench-50 attacks sequentially on one GPU.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
LOG_DIR="06_Results_Artifacts/results/attacks/logs"
mkdir -p "$LOG_DIR"

gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' \
  2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' \
  2>/dev/null || true

run_model() {
  local tag="$1"
  local model="$2"
  local log="$LOG_DIR/tao_advbench_${tag}.log"

  if [[ -f "$log" ]]; then
    mv "$log" "$log.$(date +%Y%m%d_%H%M%S).bak"
  fi
  date '+START %Y-%m-%d %H:%M:%S' | tee "${log}.meta"

  systemd-inhibit \
    --what=sleep:idle \
    --who="BRASS TAO AdvBench" \
    --why="long GPU attack run" \
    --mode=block \
    .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_tao_watchdog.py \
      --log "$log" \
      --stall-timeout 900 \
      --poll 20 \
      --heartbeat-every 60 \
      -- \
      --dataset advbench \
      --model "$model" \
      --init-mode easy-to-hard \
      --seed 235711 \
      --seed-steps 1000 \
      --num-steps 500 \
      --batch-size 256 \
      --topk 256 \
      --cl-threshold 1.0 \
      --temperature 0.5 \
      --alpha 0.2 \
      --beta 0.2 \
      --refusal-set-size 3 \
      --revert-after 3 \
      --success-judge local \
      --stop-on-success \
      --num-workers 1 \
      --gpu 0
}

run_model "vicuna_7b_v1_5" "lmsys/vicuna-7b-v1.5"
run_model "llama_2_7b_chat" "meta-llama/Llama-2-7b-chat-hf"
