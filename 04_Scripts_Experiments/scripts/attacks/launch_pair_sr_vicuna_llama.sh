#!/usr/bin/env bash
# PAIR on full StrongREJECT vs Vicuna-7B-v1.5 then Llama-2-7B-Chat (one GPU; Qwen3-32B attacker).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO"

if [[ -f "$REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
fi
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
LOG_DIR="06_Results_Artifacts/results/attacks/logs"
mkdir -p "$LOG_DIR"

gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' \
  2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' \
  2>/dev/null || true

INHIBIT=(systemd-inhibit --what=sleep:idle --who="BRASS PAIR StrongREJECT" --why="long GPU attack run" --mode=block)
if ! "${INHIBIT[@]}" true >/dev/null 2>&1; then
  echo "[launch] systemd-inhibit unavailable; relying on gsettings auto-suspend disable."
  INHIBIT=()
fi

run_model() {
  local tag="$1"
  local model="$2"
  local log="$LOG_DIR/pair_sr_${tag}.log"

  if [[ -f "$log" ]]; then
    mv "$log" "$log.$(date +%Y%m%d_%H%M%S).bak"
  fi
  date '+START %Y-%m-%d %H:%M:%S' | tee "${log}.meta"

  "${INHIBIT[@]}" \
    .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_pair.py \
      --dataset strongreject \
      --attacker-model Qwen/Qwen3-32B \
      --target-model "$model" \
      --n-streams 5 \
      --n-iterations 5 \
      --seed 235711 \
      2>&1 | tee "$log"
}

run_model "vicuna_7b_v1_5" "lmsys/vicuna-7b-v1.5"
run_model "llama_2_7b_chat" "meta-llama/Llama-2-7b-chat-hf"
