#!/usr/bin/env bash
# Run the 15-prompt StrongREJECT smoke test end to end.
#
# Loads .env (HF_TOKEN, WANDB_*, OLMOTRACE_*), then runs the Hydra pipeline. Pass extra Hydra
# overrides through, e.g.:
#   bash 04_Scripts_Experiments/scripts/run_smoke_test.sh sampling.n_completions=8 wandb.mode=offline
set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer the project venv if present.
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

# Default cache locations (overridable via real env).
export HF_HOME="${HF_HOME:-$REPO_ROOT/models}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
# vLLM engine subprocess must spawn (the driver initializes CUDA before later engine loads).
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

exec "$PY" -m brass.pipeline.run_experiment experiment=smoke_test "$@"
