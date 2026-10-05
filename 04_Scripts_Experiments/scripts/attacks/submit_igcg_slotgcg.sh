#!/usr/bin/env bash
# Submit I-GCG first, then SlotGCG; each array has four active behavior jobs at most.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ ! -x .venv/bin/python || ! -f .env ]]; then
  echo "Missing .venv or .env." >&2
  exit 2
fi
RUN_BASE="06_Results_Artifacts/new_runs/iterative_attacks"
RUN_ID="igcg-slotgcg-advbench-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$RUN_BASE/$RUN_ID"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/igcg" "$RUN_DIR/slotgcg"
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/build_attack_datasets.py --datasets advbench --max-behaviors 0
IGCG_JOB="$(sbatch --parsable --array=1-50%4 --export="ALL,ITER_ATTACK_NAME=igcg,ITER_ATTACK_RUN_DIR=$RUN_DIR" \
  04_Scripts_Experiments/scripts/attacks/igcg_slotgcg_task_a100.sbatch)"
IGCG_JOB="${IGCG_JOB%%;*}"
SLOTGCG_JOB="$(sbatch --parsable --dependency="afterany:$IGCG_JOB" --array=1-50%4 \
  --export="ALL,ITER_ATTACK_NAME=slotgcg,ITER_ATTACK_RUN_DIR=$RUN_DIR" \
  04_Scripts_Experiments/scripts/attacks/igcg_slotgcg_task_a100.sbatch)"
SLOTGCG_JOB="${SLOTGCG_JOB%%;*}"
.venv/bin/python - "$RUN_DIR/run.json" "$IGCG_JOB" "$SLOTGCG_JOB" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1])
p.write_text(json.dumps({
  "study":"iterative_attack_final_response_semantic_labels",
  "created_utc":datetime.now(timezone.utc).isoformat(),"status":"submitted",
  "dataset":"separate paper-owned AdvBench-50 snapshots (I-GCG and SlotGCG)",
  "target_model":"allenai/Olmo-3-7B-Instruct","classifier_model":"Qwen/Qwen3.5-9B",
  "classifier":"existing semantic reconstruction pilot.py; K=3",
  "label_feedback":"recorded after each attack completes; does not guide or stop attack",
  "steps_per_behavior":100,"behaviors_per_attack":50,"max_concurrent_jobs":4,
  "arrays":{"igcg":{"job_id":sys.argv[2]},"slotgcg":{"job_id":sys.argv[3],"dependency":"afterany:"+sys.argv[2]}},
  "results":"one final attacked prompt, target response, and semantic label per behavior",
  "run_root":str(p.parent)
},indent=2)+"\n")
PY
echo "Submitted I-GCG array $IGCG_JOB and SlotGCG array $SLOTGCG_JOB (50 tasks each, %4; SlotGCG waits for I-GCG). Results: $RUN_DIR"
