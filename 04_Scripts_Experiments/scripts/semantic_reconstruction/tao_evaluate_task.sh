#!/usr/bin/env bash
# Evaluate one item inside the unified two-A100 TAO job.
set -euo pipefail
TASK_ID="${1:?task id is required}"
RUN_ID="${2:?run id is required}"
TAO_CACHE_PATH="${3:?TAO cache path is required}"
DATASET_INDEX="$((TASK_ID - 1))"
OUT_DIR="06_Results_Artifacts/new_runs/semantic_reconstruction"
PROMPT_FILE="$OUT_DIR/tao_prompt_${DATASET_INDEX}_${RUN_ID}.txt"
ATTACKED_FILE="$OUT_DIR/tao_attacked_${DATASET_INDEX}_${RUN_ID}.txt"
RESPONSE_FILE="$OUT_DIR/tao_response_${DATASET_INDEX}_${RUN_ID}.txt"

.venv/bin/python - "$DATASET_INDEX" "$TAO_CACHE_PATH" "$PROMPT_FILE" "$ATTACKED_FILE" <<'PY'
import json, sys
from pathlib import Path
from brass.data import load_dataset_prompts

idx = int(sys.argv[1])
cache_path, prompt_path, attacked_path = map(Path, sys.argv[2:])
cache = json.loads(cache_path.read_text())
behavior_id = f'strongreject:{idx}'
record = cache.get(behavior_id)
if not isinstance(record, dict) or record.get('attack') != 'tao':
    raise SystemExit(f'No TAO optimized prompt for {behavior_id}')
rows = [row for row in load_dataset_prompts('strongreject') if row.id == behavior_id]
if len(rows) != 1:
    raise SystemExit(f'Expected one original behavior for {behavior_id}; found {len(rows)}')
attacked = record.get('attacked_prompt')
if not isinstance(attacked, str) or not attacked or attacked == rows[0].prompt:
    raise SystemExit(f'Missing or unchanged TAO prompt for {behavior_id}')
prompt_path.write_text(rows[0].prompt)
attacked_path.write_text(attacked)
PY

.venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/generate_tao_response.py \
  --model allenai/Olmo-3-7B-Instruct \
  --prompt-file "$ATTACKED_FILE" \
  --output "$RESPONSE_FILE" \
  --seed "$((235711 + TASK_ID))" \
  --max-model-len 16384 \
  --max-tokens 8192

.venv/bin/python 04_Scripts_Experiments/scripts/semantic_reconstruction/pilot.py \
  --model "${SEMREC_MODEL:-Qwen/Qwen3.5-9B}" \
  --target-model allenai/Olmo-3-7B-Instruct \
  --prompt-file "$PROMPT_FILE" \
  --attacked-prompt-file "$ATTACKED_FILE" \
  --response-file "$RESPONSE_FILE" \
  --source-label "tao_strongreject_${DATASET_INDEX}_${RUN_ID}" \
  --attack-name "TAO GCG suffix optimization" \
  --target-max-tokens 8192 \
  --aux-max-tokens 8192 \
  --judge-max-tokens 8192 \
  --classifier-max-tokens 8192 \
  --max-model-len 16384 \
  --k "${SEMREC_K:-3}" \
  --seed "$((235711 + TASK_ID))" \
  --output "$OUT_DIR/tao_semantic_${DATASET_INDEX}_${RUN_ID}.json"
