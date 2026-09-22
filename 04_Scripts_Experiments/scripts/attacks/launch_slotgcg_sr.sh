#!/usr/bin/env bash
# Launch the SlotGCG StrongREJECT run on Olmo-3-7B-Instruct, hardened against the GNOME idle
# auto-suspend that was breaking the GPU job. Root cause of the earlier "hangs": a 100%-GPU job
# does NOT reset GNOME's *input*-idle timer, so the desktop auto-suspended after 20 min of no
# keyboard/mouse activity (sleep-inactive-ac-timeout=1200), freezing/corrupting the CUDA context.
#
# Two layers of defense: (1) disable GNOME idle auto-suspend, (2) hold a systemd sleep inhibitor for
# the whole run. The run itself is resume-mode under the stall watchdog (completed behaviors are
# skipped, so this is safe to re-run).
set -eu

REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO"
LOG=06_Results_Artifacts/results/attacks/logs/slotgcg_sr_7b.log

# (1) Disable GNOME idle auto-suspend (idempotent; the real fix).
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' 2>/dev/null || true

export HF_HOME=${HOME}/.cache/huggingface HF_HUB_CACHE=${HOME}/.cache/huggingface/hub
mkdir -p 06_Results_Artifacts/results/attacks/logs
# Rotate the previous run log but KEEP the native dir so resume skips already-finished behaviors.
[ -f "$LOG" ] && mv "$LOG" "$LOG.$(date +%Y%m%d_%H%M%S).bak" || true
date '+START %Y-%m-%d %H:%M:%S' | tee 06_Results_Artifacts/results/attacks/logs/slotgcg_sr_7b.meta

# (2) Run under the stall watchdog (resume mode; no --overwrite). A systemd sleep inhibitor is
# belt-and-suspenders on top of the gsettings fix above, but it requires a privileged/login
# session; in non-login shells it returns "Failed to inhibit: Access denied". Probe it and only
# wrap with it when actually permitted, otherwise rely on the gsettings auto-suspend disable so the
# launch never dies on an unavailable inhibitor.
INHIBIT=(systemd-inhibit --what=sleep:idle --who="BRASS SlotGCG" --why="long GPU attack run" --mode=block)
if ! "${INHIBIT[@]}" true >/dev/null 2>&1; then
  echo "[launch] systemd-inhibit unavailable (Access denied); relying on gsettings auto-suspend disable."
  INHIBIT=()
fi

exec "${INHIBIT[@]}" \
  .venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_slotgcg_watchdog.py \
    --log "$LOG" --stall-timeout 600 --poll 20 --heartbeat-every 60 -- \
    --dataset strongreject --model allenai/Olmo-3-7B-Instruct \
    --num-steps 300 --search-width 256 --num-adv-string 20 \
    --eval-steps 25 --eval-with-check-refusal --check-refusal-min-loss 0.1 \
    --early-stopping --early-stopping-min-loss 0.01 --num-workers 1 --gpu 0
