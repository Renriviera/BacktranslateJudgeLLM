#!/usr/bin/env bash
# Submit the saved-response pathological-case stress run.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p 06_Results_Artifacts/new_runs/semantic_reconstruction
python3 04_Scripts_Experiments/scripts/semantic_reconstruction/prepare_pathological_cases.py
CASE_COUNT="$(wc -l < 06_Results_Artifacts/new_runs/semantic_reconstruction/pathological_cases.jsonl)"
if [[ "$CASE_COUNT" -lt 1 ]]; then
  echo "No pathological cases were generated." >&2
  exit 2
fi
SBATCH_ARGS=(--array="1-${CASE_COUNT}%4")
if [[ -n "${SEMREC_DEPENDENCY:-}" ]]; then
  SBATCH_ARGS+=(--dependency="afterok:${SEMREC_DEPENDENCY}")
fi
exec sbatch "${SBATCH_ARGS[@]}" \
  04_Scripts_Experiments/scripts/semantic_reconstruction/pathological_stress_adversarial_a100.sbatch
