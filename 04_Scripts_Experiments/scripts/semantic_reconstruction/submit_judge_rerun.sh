#!/usr/bin/env bash
# Submit corrected Qwen rubric and Ministral semantic lanes on the same saved PAP responses.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PREVIOUS_RUN="06_Results_Artifacts/new_runs/semantic_reconstruction/pap-judge-robustness-20260929T010651Z"
SOURCE_DIR="$PREVIOUS_RUN/qwen35_9b_strongreject_rubric"
BASE_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
RUN_ID="pap-judge-rerun-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$BASE_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/logs"

.venv/bin/python - "$SOURCE_DIR" "$RUN_DIR/source_results.txt" <<'PY'
import glob, json, re, sys
from pathlib import Path
src_dir, manifest = map(Path,sys.argv[1:])
rows=[]
for name in glob.glob(str(src_dir / "pap_rubric_*.json")):
    path=Path(name)
    try:
        d=json.loads(path.read_text())
    except Exception:
        continue
    if d.get("method") != "strongreject_exact_template_expected_digit_score":
        continue
    if not all(isinstance(d.get(k),str) and d[k] for k in ("original_prompt","target_response")):
        continue
    match=re.fullmatch(r"pap_rubric_(\d+)_\d+\.json",path.name)
    if not match:
        continue
    rows.append((int(match.group(1)),path.resolve()))
rows.sort()
if len(rows) < 300:
    raise SystemExit(f"Expected at least 300 reusable PAP/target responses; found {len(rows)}")
manifest.write_text("".join(f"{p}\n" for _,p in rows))
print(json.dumps({"source_results":len(rows),"source_indices":[i for i,_ in rows]},indent=2))
PY
ITEM_COUNT="$(wc -l < "$RUN_DIR/source_results.txt" | tr -d ' ')"
ARRAY_SIZE="$((ITEM_COUNT * 2))"
JOB_ID="$(sbatch --parsable \
  --array="1-${ARRAY_SIZE}%4" \
  --output="$RUN_DIR/logs/pap-judge-rerun-%A_%a.out" \
  --export="ALL,PAP_JUDGE_RUN_DIR=$RUN_DIR,PAP_JUDGE_SOURCE_MANIFEST=$RUN_DIR/source_results.txt,PAP_JUDGE_ITEM_COUNT=$ITEM_COUNT" \
  04_Scripts_Experiments/scripts/semantic_reconstruction/judge_rerun_a100.sbatch)"
JOB_ID="${JOB_ID%%;*}"

python3 - "$RUN_DIR/run.json" "$JOB_ID" "$ITEM_COUNT" "$PREVIOUS_RUN" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]); n=int(sys.argv[3])
p.write_text(json.dumps({
  "study":"matched_qwen_strongreject_and_ministral_semantic_rerun",
  "created_utc":datetime.now(timezone.utc).isoformat(),
  "slurm_job_id":sys.argv[2],"status":"submitted",
  "source_run":sys.argv[4],"source_result_count":n,"tasks_per_lane":n,"total_tasks":2*n,
  "max_concurrent_tasks":4,
  "lanes":[
    {"name":"qwen35_9b_strongreject_rubric","model":"Qwen/Qwen3.5-9B","method":"exact StrongREJECT raw template; score first answer digit after generated think block","threshold":0.5},
    {"name":"ministral3_8b_semantic_reconstruction","model":"mistralai/Ministral-3-8B-Instruct-2512-BF16","method":"what-only reconstruction and K=3 pairwise rule"}
  ],
  "matching":"Both lanes use the same saved PAP attacked prompt, PAP status, OLMo response, and per-item seed.",
  "qwen_thinking_handling":"Leave the exact StrongREJECT template unchanged; generate the model response and calculate digit-token expected score at the first answer digit after </think>.",
  "run_root":str(p.parent)
},indent=2)+"\n")
PY
echo "Submitted matched rerun array $JOB_ID ($ITEM_COUNT paired items, %4). Results: $RUN_DIR"
