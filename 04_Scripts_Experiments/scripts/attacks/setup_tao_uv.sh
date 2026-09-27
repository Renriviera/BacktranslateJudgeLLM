#!/usr/bin/env bash
# Install only TAO's extra Python packages into the repository's shared uv environment.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
command -v uv >/dev/null 2>&1 || { echo "uv is required on PATH." >&2; exit 2; }
command -v module >/dev/null 2>&1 && module load python3
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-/tmp/$USER}/uv-cache}"
mkdir -p "$UV_CACHE_DIR"
declare -a MISSING=()
for entry in \
  'fschat:fschat==0.2.36' \
  'ml_collections:ml-collections==1.1.0' \
  'shortuuid:shortuuid==1.0.13' \
  'datasketch:datasketch' \
  'spacy:spacy' \
  'accelerate:accelerate' \
  'rouge_score:rouge-score' \
  'openai:openai'; do
  module_name="${entry%%:*}"
  package_spec="${entry#*:}"
  if ! .venv/bin/python -c 'import importlib.util,sys; sys.exit(importlib.util.find_spec(sys.argv[1]) is None)' "$module_name"; then
    MISSING+=("$package_spec")
  fi
done
if [[ "${#MISSING[@]}" -gt 0 ]]; then
  uv pip install --python .venv/bin/python --no-deps "${MISSING[@]}"
fi
echo "TAO extra packages are available in the shared environment at $REPO_ROOT/.venv"
