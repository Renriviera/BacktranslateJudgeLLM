#!/usr/bin/env bash
# Judges-only PAP (3 techniques) + PAIR eval on Vicuna-7B-v1.5 and Llama-2-7B-Chat, then funnels.
# PAP reuses the existing StrongREJECT caches; PAIR requires the caches from
# 04_Scripts_Experiments/scripts/attacks/launch_pair_sr_vicuna_llama.sh.
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

INHIBIT=(systemd-inhibit --what=sleep:idle --who="BRASS PAP/PAIR eval" --why="long GPU eval run" --mode=block)
if ! "${INHIBIT[@]}" true >/dev/null 2>&1; then
  echo "[launch] systemd-inhibit unavailable; relying on gsettings auto-suspend disable."
  INHIBIT=()
fi

EXPERIMENTS=(
  pap_misrep_strongreject_vicuna_7b
  pap_logic_strongreject_vicuna_7b
  pap_authority_strongreject_vicuna_7b
  pap_misrep_strongreject_llama2_7b
  pap_logic_strongreject_llama2_7b
  pap_authority_strongreject_llama2_7b
  pair_strongreject_vicuna_7b
  pair_strongreject_llama2_7b
)

for exp in "${EXPERIMENTS[@]}"; do
  log="$LOG_DIR/eval_${exp}.log"
  if [[ -f "$log" ]]; then
    mv "$log" "$log.$(date +%Y%m%d_%H%M%S).bak"
  fi
  date "+START %Y-%m-%d %H:%M:%S  experiment=${exp}" | tee "${log}.meta"
  "${INHIBIT[@]}" \
    .venv/bin/python -m brass.pipeline.run_experiment "experiment=${exp}" \
    2>&1 | tee "$log"
done

# PAP funnels: shared attacker reports, per-target --out so OLMo funnels are not overwritten.
python_funnel() {
  .venv/bin/python "$@"
}

python_funnel 04_Scripts_Experiments/scripts/attacks/pap_funnel.py \
  --report 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.report.json \
  --details 06_Results_Artifacts/results/pap_misrep_strongreject_vicuna_7b/details.json \
  --out 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.vicuna_7b.funnel.json
python_funnel 04_Scripts_Experiments/scripts/attacks/pap_funnel.py \
  --report 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.logical_appeal.report.json \
  --details 06_Results_Artifacts/results/pap_logic_strongreject_vicuna_7b/details.json \
  --out 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.logical_appeal.vicuna_7b.funnel.json
python_funnel 04_Scripts_Experiments/scripts/attacks/pap_funnel.py \
  --report 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.authority_endorsement.report.json \
  --details 06_Results_Artifacts/results/pap_authority_strongreject_vicuna_7b/details.json \
  --out 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.authority_endorsement.vicuna_7b.funnel.json

python_funnel 04_Scripts_Experiments/scripts/attacks/pap_funnel.py \
  --report 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.report.json \
  --details 06_Results_Artifacts/results/pap_misrep_strongreject_llama2_7b/details.json \
  --out 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.llama2_7b.funnel.json
python_funnel 04_Scripts_Experiments/scripts/attacks/pap_funnel.py \
  --report 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.logical_appeal.report.json \
  --details 06_Results_Artifacts/results/pap_logic_strongreject_llama2_7b/details.json \
  --out 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.logical_appeal.llama2_7b.funnel.json
python_funnel 04_Scripts_Experiments/scripts/attacks/pap_funnel.py \
  --report 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.authority_endorsement.report.json \
  --details 06_Results_Artifacts/results/pap_authority_strongreject_llama2_7b/details.json \
  --out 06_Results_Artifacts/results/attacks/pap/olmo3_7b_instruct.authority_endorsement.llama2_7b.funnel.json

# PAIR funnels: per-target reports (default out is fine). StrongREJECT-ft default + HarmBench.
python_funnel 04_Scripts_Experiments/scripts/attacks/pair_funnel.py \
  --report 06_Results_Artifacts/results/attacks/pair/vicuna_7b_v1_5.report.json \
  --details 06_Results_Artifacts/results/pair_strongreject_vicuna_7b/details.json
python_funnel 04_Scripts_Experiments/scripts/attacks/pair_funnel.py \
  --report 06_Results_Artifacts/results/attacks/pair/vicuna_7b_v1_5.report.json \
  --details 06_Results_Artifacts/results/pair_strongreject_vicuna_7b/details.json \
  --judge-key harmbench \
  --out 06_Results_Artifacts/results/attacks/pair/vicuna_7b_v1_5.report.harmbench.funnel.json

python_funnel 04_Scripts_Experiments/scripts/attacks/pair_funnel.py \
  --report 06_Results_Artifacts/results/attacks/pair/llama_2_7b_chat.report.json \
  --details 06_Results_Artifacts/results/pair_strongreject_llama2_7b/details.json
python_funnel 04_Scripts_Experiments/scripts/attacks/pair_funnel.py \
  --report 06_Results_Artifacts/results/attacks/pair/llama_2_7b_chat.report.json \
  --details 06_Results_Artifacts/results/pair_strongreject_llama2_7b/details.json \
  --judge-key harmbench \
  --out 06_Results_Artifacts/results/attacks/pair/llama_2_7b_chat.report.harmbench.funnel.json
