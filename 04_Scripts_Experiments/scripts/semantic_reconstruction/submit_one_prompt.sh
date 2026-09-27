#!/usr/bin/env bash
# Submit StrongREJECT PAP semantic-classifier items, capped at four concurrent A100 tasks.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p 06_Results_Artifacts/new_runs/semantic_reconstruction
OUT_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
BEHAVIOR_COUNT="$(.venv/bin/python -c 'from brass.data import load_dataset_prompts; print(len(load_dataset_prompts("strongreject")))')"
TASKS=()
for ((task_id=1; task_id<=BEHAVIOR_COUNT; task_id++)); do
  behavior_index="$((task_id - 1))"
  if [[ "${SEMREC_RERUN_ALL:-0}" != "1" ]] && compgen -G "$OUT_DIR/pap_semantic_${behavior_index}_*.json" >/dev/null; then
    continue
  fi
  TASKS+=("$task_id")
done
if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "All $BEHAVIOR_COUNT StrongREJECT items already have classifier outputs."
  exit 0
fi
ARRAY_SPEC="$(IFS=,; echo "${TASKS[*]}")%4"
echo "Submitting ${#TASKS[@]} items from $BEHAVIOR_COUNT StrongREJECT behaviors (array $ARRAY_SPEC)."
exec sbatch --array="$ARRAY_SPEC" 04_Scripts_Experiments/scripts/semantic_reconstruction/one_prompt_a100.sbatch
