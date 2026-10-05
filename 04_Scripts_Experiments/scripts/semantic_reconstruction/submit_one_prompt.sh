#!/usr/bin/env bash
# Submit StrongREJECT PAP semantic-reconstruction items, capped at four concurrent A100 tasks.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
OUT_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
mkdir -p "$OUT_DIR/logs/pap"
RUN_ID="pap-strongreject-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUT_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/logs"
BEHAVIOR_COUNT="$(.venv/bin/python -c 'from brass.data import load_dataset_prompts; print(len(load_dataset_prompts("strongreject")))')"
TASKS=()
for ((task_id=1; task_id<=BEHAVIOR_COUNT; task_id++)); do
  behavior_index="$((task_id - 1))"
  if [[ "${SEMREC_RERUN_ALL:-0}" != "1" ]] && compgen -G "$OUT_DIR/*/pap_semantic_${behavior_index}_*.json" >/dev/null; then
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
DEPENDENCY_ARGS=()
if [[ -n "${SEMREC_AFTER_JOB_ID:-}" ]]; then
  DEPENDENCY_ARGS+=(--dependency="afterany:${SEMREC_AFTER_JOB_ID}")
  echo "This array will start after job ${SEMREC_AFTER_JOB_ID} finishes."
fi
JOB_ID="$(sbatch --parsable "${DEPENDENCY_ARGS[@]}" --output="$RUN_DIR/logs/pap-semantic-%A_%a.out" \
  --export="ALL,SEMREC_RUN_DIR=$RUN_DIR" --array="$ARRAY_SPEC" \
  04_Scripts_Experiments/scripts/semantic_reconstruction/one_prompt_a100.sbatch)"
JOB_ID="${JOB_ID%%;*}"
python3 - "$RUN_DIR/run.json" "$JOB_ID" "$BEHAVIOR_COUNT" "${#TASKS[@]}" "$ARRAY_SPEC" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "study": "pap_strongreject_semantic_reconstruction",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": sys.argv[2],
    "strongreject_behaviors": int(sys.argv[3]),
    "submitted_tasks": int(sys.argv[4]),
    "array_spec": sys.argv[5],
    "pap_attacker_model": os.environ.get("SEMREC_ATTACKER_MODEL", "Qwen/Qwen3-32B"),
    "target_model": "allenai/Olmo-3-7B-Instruct",
    "evaluation_model": os.environ.get("SEMREC_MODEL", "Qwen/Qwen3.5-9B"),
    "k": int(os.environ.get("SEMREC_K", "3")),
    "status": "submitted",
    "dependency": os.environ.get("SEMREC_AFTER_JOB_ID")
}, indent=2) + "\n")
PY
echo "Submitted PAP semantic run $JOB_ID. Run artifacts and logs: $RUN_DIR"
