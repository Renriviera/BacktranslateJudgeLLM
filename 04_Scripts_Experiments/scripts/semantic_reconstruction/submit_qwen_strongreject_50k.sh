#!/usr/bin/env bash
# Submit a Qwen StrongREJECT pass over all saved PAP/target responses from pipeline run 1729137.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
SOURCE_RUN="06_Results_Artifacts/new_runs/semantic_reconstruction/pap-1729137"
BASE_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
RUN_ID="pap-qwen-strongreject-50k-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$BASE_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/source_inputs"

.venv/bin/python - "$SOURCE_RUN" "$RUN_DIR/source_inputs" "$RUN_DIR/source_results.txt" <<'PY'
import glob,json,re,sys
from pathlib import Path
source=Path(sys.argv[1]); outdir=Path(sys.argv[2]); manifest=Path(sys.argv[3])
by_index={}
for name in glob.glob(str(source/"pap_semantic_*.json")):
    p=Path(name); d=json.loads(p.read_text())
    m=re.fullmatch(r"pap_semantic_(\d+)_1729137\.json",p.name)
    if not m: continue
    i=int(m.group(1)); meta=d.get("attack_metadata") or {}
    by_index[i]={"original_prompt":d["original_prompt"],"attacked_prompt":d.get("attacked_prompt"),
      "target_response":d["target_response"],"target_finish_reason":d.get("target_finish_reason"),
      "pap_status":meta.get("pap_status","unknown"),"seed":d.get("seed",235711+i+1),
      "target_model":d.get("target_generation_config",{}).get("model","allenai/Olmo-3-7B-Instruct"),
      "source_artifact":str(p)}
for name in glob.glob(str(source/"pap_target_recovery_*.json")):
    p=Path(name); d=json.loads(p.read_text()); i=int(d["dataset_index"])
    report=source/f"pap_strongreject_{i}_1729137.report.json"
    rows=json.loads(report.read_text()).get("per_behavior",[]) if report.exists() else []
    if len(rows)!=1: raise SystemExit(f"Missing PAP report for recovered index {i}")
    row=rows[0]
    by_index[i]={"original_prompt":d["original_prompt"],"attacked_prompt":d["attacked_prompt"],
      "target_response":d["target_response"],"target_finish_reason":d.get("target_finish_reason"),
      "pap_status":row.get("pap_status","unknown"),"seed":235711+i+1,
      "target_model":d.get("target_model","allenai/Olmo-3-7B-Instruct"),
      "source_artifact":str(p)}
if sorted(by_index)!=list(range(313)):
    missing=sorted(set(range(313))-set(by_index)); extra=sorted(set(by_index)-set(range(313)))
    raise SystemExit(f"Expected all 313 original responses; missing={missing}, extra={extra}")
paths=[]
for i in range(313):
    d=by_index[i]
    target=outdir/f"pap_rubric_{i}_1729137.json"
    target.write_text(json.dumps(d,indent=2,ensure_ascii=False)+"\n")
    paths.append(str(target.resolve()))
manifest.write_text("\n".join(paths)+"\n")
print(json.dumps({"source_semantic_outputs":len(glob.glob(str(source/"pap_semantic_*.json"))),
 "recovered_target_responses":len(glob.glob(str(source/"pap_target_recovery_*.json"))),
 "prepared_total":len(paths)},indent=2))
PY

JOB_ID="$(sbatch --parsable \
  --array="1-313%4" \
  --output="$RUN_DIR/logs/qwen-sr-50k-%A_%a.out" \
  --export="ALL,PAP_JUDGE_RUN_DIR=$RUN_DIR,PAP_JUDGE_SOURCE_MANIFEST=$RUN_DIR/source_results.txt" \
  04_Scripts_Experiments/scripts/semantic_reconstruction/qwen_strongreject_50k_a100.sbatch)"
JOB_ID="${JOB_ID%%;*}"

python3 - "$RUN_DIR/run.json" "$JOB_ID" "$SOURCE_RUN" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1])
p.write_text(json.dumps({
  "study":"qwen_strongreject_on_original_semantic_pipeline_responses",
  "created_utc":datetime.now(timezone.utc).isoformat(),"slurm_job_id":sys.argv[2],"status":"submitted",
  "source_run":sys.argv[3],"items":313,"source_counts":{"semantic_pipeline_outputs":301,"recovered_target_responses":12},
  "max_concurrent_tasks":4,"judge_model":"Qwen/Qwen3.5-9B",
  "judge_max_generation_tokens":50000,"judge_max_model_len":65536,
  "judge_prompt":"Exact STRONGREJECT_FINETUNED_TEMPLATE raw prompt, unchanged; no added instructions or adapter.",
  "thinking_handling":"Generate without per-token logprob retention; then score expected digit-token probabilities at first post-</think> answer position, conditioned on the generated prefix.",
  "target_responses":"Reuse original pipeline run responses and 12 recovered target responses; no PAP or target regeneration.",
  "pap_refusals":"Retained as unsuccessful at the PAP gate.","threshold":0.5,"run_root":str(p.parent)
},indent=2)+"\n")
PY
echo "Submitted Qwen StrongREJECT job $JOB_ID (313 original pipeline responses, %4). Results: $RUN_DIR"
