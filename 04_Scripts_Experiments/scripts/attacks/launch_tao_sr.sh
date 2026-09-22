#!/usr/bin/env bash
# Launch TAO-Attack on StrongREJECT / Olmo-3-7B-Instruct with suspend + stall safeguards.
set -eu

REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO"
LOG=06_Results_Artifacts/results/attacks/logs/tao_sr_7b.log

gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' 2>/dev/null || true

export HF_HOME=${HOME}/.cache/huggingface HF_HUB_CACHE=${HOME}/.cache/huggingface/hub
mkdir -p 06_Results_Artifacts/results/attacks/logs
[ -f "$LOG" ] && mv "$LOG" "$LOG.$(date +%Y%m%d_%H%M%S).bak" || true
date '+START %Y-%m-%d %H:%M:%S' | tee 06_Results_Artifacts/results/attacks/logs/tao_sr_7b.meta

exec systemd-inhibit --what=sleep:idle --who="BRASS TAO" --why="long GPU attack run" --mode=block \
  .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_tao_watchdog.py \
    --log "$LOG" --stall-timeout 600 --poll 20 --heartbeat-every 60 -- \
    --dataset strongreject --model allenai/Olmo-3-7B-Instruct \
    --num-steps 500 --batch-size 512 --num-workers 1 --gpu 0
