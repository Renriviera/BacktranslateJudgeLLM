#!/usr/bin/env bash
# TAO-Attack on AdvBench-50 / Llama-2-7B-chat with multi-seed search.
#
# The first AdvBench run used paper seed 50 only, stayed in stage 0, and transferred
# a failed suffix (0% GPT ASR). This launcher mirrors the StrongREJECT Llama-2 fixes:
# multi-seed search, abandon stage-0 candidates after 250 steps, and abort if no
# seed jailbreaks. AdvBench seeds rarely reach TAO stage 1 on Llama-2 within 250
# steps; seed 44 (phishing email) can pass the 256-token local judge, so acceptance
# uses the local criterion rather than stage1-ever.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
LOG_DIR="06_Results_Artifacts/results/attacks/logs"
LOG="$LOG_DIR/tao_advbench_llama_2_7b_chat.log"
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
  --who="BRASS TAO AdvBench Llama-2" \
  --why="long GPU attack run" \
  --mode=block \
  .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_tao_watchdog.py \
    --log "$LOG" \
    --stall-timeout 900 \
    --poll 20 \
    --heartbeat-every 60 \
    -- \
    --dataset advbench \
    --model meta-llama/Llama-2-7b-chat-hf \
    --init-mode easy-to-hard \
    --seed-behavior-id 50 \
    --seed-candidate-ids 44,20,36,29,37,50 \
    --seed 235711 \
    --seed-steps 1000 \
    --seed-abandon-stage0-after 250 \
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
    --seed-jailbreak-criterion local \
    --num-workers 1 \
    --gpu 0
