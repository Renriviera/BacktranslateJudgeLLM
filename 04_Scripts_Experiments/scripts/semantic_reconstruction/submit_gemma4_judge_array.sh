#!/usr/bin/env bash
# Submit paired Gemma 4 judge lanes over the already-saved 313 PAP responses.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
SOURCE_RUN="06_Results_Artifacts/new_runs/semantic_reconstruction/pap-qwen-strongreject-50k-20260930T035637Z"
SOURCE_MANIFEST="$SOURCE_RUN/source_results.txt"
BASE_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
RUN_ID="pap-gemma4-judge-comparison-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$BASE_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/logs"

ITEM_COUNT="$(wc -l < "$SOURCE_MANIFEST" | tr -d ' ')"
if [[ "$ITEM_COUNT" != "313" ]]; then
  echo "Expected 313 saved PAP responses in $SOURCE_MANIFEST; got $ITEM_COUNT." >&2
  exit 2
fi
cp "$SOURCE_MANIFEST" "$RUN_DIR/source_results.txt"

JOB_ID="$(sbatch --parsable \
  --array="1-$((ITEM_COUNT * 2))%4" \
  --output="$RUN_DIR/logs/gemma4-judge-%A_%a.out" \
  --export="ALL,PAP_JUDGE_RUN_DIR=$RUN_DIR,PAP_JUDGE_SOURCE_MANIFEST=$RUN_DIR/source_results.txt,PAP_JUDGE_ITEM_COUNT=$ITEM_COUNT" \
  04_Scripts_Experiments/scripts/semantic_reconstruction/gemma4_judge_array_a100.sbatch)"
JOB_ID="${JOB_ID%%;*}"

python3 - "$RUN_DIR/run.json" "$JOB_ID" "$SOURCE_RUN" "$ITEM_COUNT" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]); n=int(sys.argv[4])
p.write_text(json.dumps({
  "study":"matched_gemma4_strongreject_and_semantic_pipeline_judge_comparison",
  "created_utc":datetime.now(timezone.utc).isoformat(),
  "slurm_job_id":sys.argv[2],"status":"submitted",
  "source_run":sys.argv[3],"source_result_count":n,"tasks_per_lane":n,"total_tasks":2*n,
  "max_concurrent_tasks":4,
  "model":"google/gemma-4-26B-A4B-it",
  "lanes":[
    {"name":"gemma4_26b_a4b_strongreject_rubric","method":"Exact STRONGREJECT_FINETUNED_TEMPLATE raw prompt; no adapter; expected digit score after generated reasoning; 50k generation cap; threshold 0.5."},
    {"name":"gemma4_26b_a4b_semantic_reconstruction","method":"Existing what-only semantic pipeline; K=3; auxiliary and pairwise stages at 8192 tokens; existing prompts/parser."}
  ],
  "matching":"Both lanes reuse the same saved original prompt, attacked prompt, PAP status, target response, and seed from the original 313-item PAP run. No PAP or target response is regenerated.",
  "strongreject_rubric":"Unchanged from the Qwen baseline.",
  "run_root":str(p.parent)
},indent=2)+"\n")
PY
echo "Submitted Gemma 4 paired judge array $JOB_ID ($ITEM_COUNT paired items, concurrency %4). Results: $RUN_DIR"
