#!/usr/bin/env bash
# TAO-Attack on full StrongREJECT / Vicuna-7B-v1.5 with seed abort + stall watchdog.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
LOG_DIR="06_Results_Artifacts/results/attacks/logs"
LOG="$LOG_DIR/tao_strongreject_vicuna_7b_v1_5.log"
mkdir -p "$LOG_DIR"

gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' \
  2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' \
  2>/dev/null || true

if [[ -f "$LOG" ]]; then
  mv "$LOG" "$LOG.$(date +%Y%m%d_%H%M%S).bak"
fi
date '+START %Y-%m-%d %H:%M:%S' | tee "${LOG}.meta"

exec systemd-inhibit \
  --what=sleep:idle \
  --who="BRASS TAO StrongREJECT Vicuna" \
  --why="long GPU attack run" \
  --mode=block \
  .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_tao_watchdog.py \
    --log "$LOG" \
    --stall-timeout 900 \
    --poll 20 \
    --heartbeat-every 60 \
    -- \
    --dataset strongreject \
    --model lmsys/vicuna-7b-v1.5 \
    --init-mode easy-to-hard \
    --seed-behavior-id 186 \
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
    --abort-on-seed-failure \
    --seed-jailbreak-criterion both \
    --num-workers 1 \
    --gpu 0
