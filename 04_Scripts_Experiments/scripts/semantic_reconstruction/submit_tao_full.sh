#!/usr/bin/env bash
# Submit one unified full 313-behavior TAO + classifier job using two A100s.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
OUT_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
RUN_ID="tao-strongreject-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUT_DIR/$RUN_ID"
mkdir -p "$RUN_DIR" "$OUT_DIR/logs/tao"
COUNT="$(.venv/bin/python -c 'from brass.data import load_dataset_prompts; print(len(load_dataset_prompts("strongreject")))')"
if [[ "$COUNT" -ne 313 ]]; then
  echo "Expected 313 StrongREJECT behaviors; found $COUNT." >&2
  exit 2
fi
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/patch_external_repos.py --check

JOB_ID="$(sbatch --parsable --export="ALL,SEMREC_RUN_DIR=$RUN_DIR" 04_Scripts_Experiments/scripts/semantic_reconstruction/tao_optimize_a100.sbatch)"
JOB_ID="${JOB_ID%%;*}"
echo "Submitted unified TAO + semantic-classifier job: $JOB_ID (two A100s, full $COUNT-item StrongREJECT)."
