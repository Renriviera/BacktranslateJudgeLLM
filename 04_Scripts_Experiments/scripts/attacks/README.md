# Adversarial prompt generation (TAO-Attack & SlotGCG)

This directory drives the two vendored white-box GCG-family optimizers against the OLMo-3.1
checkpoints and writes the adapter cache JSON consumed by
`brass.attacks.{tao,slotgcg}_adapter` (`06_Results_Artifacts/results/attacks/{tao,slotgcg}/<model_tag>.json`).

The vendored repos live under `04_Scripts_Experiments/src/brass/attacks/external/` (gitignored). All glue, dataset
converters, the OLMo chat template, and the repo patches are tracked here.

## Why a separate environment

The repos pin `transformers==4.28/4.44` + `torch==2.4`, which (a) predate `Olmo3ForCausalLM`
support and (b) lack Blackwell (sm_120) kernels. We therefore run them in an isolated
**`.venv-attacks`** built on the same modern stack as the main env (torch 2.11+cu130,
transformers 5.12) plus the attack extras (`fschat` no-deps, `ml_collections`, `rouge_score`,
`accelerate`, `openai`). One shared venv suffices because, once both repos target OLMo3/Blackwell,
their original conflicting pins are moot.

```bash
python3 -m venv .venv-attacks
.venv-attacks/bin/pip install torch==2.11.0 transformers==5.12.0 accelerate sentencepiece \
    protobuf numpy pandas scipy ml_collections rouge_score shortuuid nltk einops datasets \
    huggingface_hub openai spacy datasketch
.venv-attacks/bin/pip install --no-deps fschat==0.2.36
.venv-attacks/bin/python -m spacy download en_core_web_sm   # SlotGCG eval_utils loads it at import
# Exact pins captured in requirements-attacks.txt:
#   .venv-attacks/bin/pip install -r requirements-attacks.txt
```

`spacy`/`datasketch`/`en_core_web_sm` are pulled in transitively by SlotGCG's `eval_utils` (loaded
at import even for step 1); `openai` is imported unconditionally by TAO's `check_openai`.

## One-time setup

```bash
# 1. (re)apply patches to the gitignored vendored repos (idempotent)
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/patch_external_repos.py

# 2. convert StrongREJECT + HarmBench into each repo's input format (100 behaviors each)
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/build_attack_datasets.py --max-behaviors 100
# Paper's curated I-GCG AdvBench split (50 records, order preserved):
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/build_attack_datasets.py --datasets advbench --max-behaviors 0
```

Converted inputs land in `01_Datasets_Benchmarks/data/attacks/{tao,slotgcg}/`. HarmBench uses official optimizer
targets; StrongREJECT targets are synthesized affirmative strings ("Sure, here is ...").

## Target model (7B by default)

The runners default to the **7B** counterparts of the 32B checkpoints (≈4–5× faster gradient
iteration, ~14GB bf16 vs ~64GB):

| role     | 32B (BRASS pipeline)              | 7B (attack default)              | tag                     |
|----------|-----------------------------------|----------------------------------|-------------------------|
| instruct | `allenai/Olmo-3.1-32B-Instruct`   | `allenai/Olmo-3-7B-Instruct`     | `olmo3_7b_instruct`     |
| dpo      | `allenai/Olmo-3.1-32B-Instruct-DPO` | `allenai/Olmo-3-7B-Instruct-DPO` | `olmo3_7b_instruct_dpo` |
| base     | `allenai/Olmo-3-1125-32B`         | `allenai/Olmo-3-1025-7B`         | `olmo3_7b_base`         |

(There is no "Olmo-3.1" 7B Instruct on HF; `Olmo-3-7B-Instruct` is the corresponding 7B.) Fetch
them with `python 04_Scripts_Experiments/scripts/download_models.py --only base_7b instruct_7b dpo_7b`. Pass `--model
allenai/Olmo-3.1-32B-Instruct` to fall back to the 32B target.

## Running the optimizers

The runners load the target in `.venv-attacks`, run the optimizer, and merge results into the
model-scoped cache (keyed by both `"{dataset}:{behavior_id}"` and the raw behavior text, so the
pipeline's `prompt.id` / `prompt.prompt` lookups both hit).

```bash
# Validate end-to-end wiring (1 behavior, few steps, ~minutes):
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_tao.py      --dataset strongreject --dry-run
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_slotgcg.py  --dataset strongreject --dry-run

# TAO paper threat-model smoke tests:
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_tao.py --dataset advbench \
    --model lmsys/vicuna-7b-v1.5 --dry-run
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_tao.py --dataset advbench \
    --model meta-llama/Llama-2-7b-chat-hf --dry-run

# Real runs (LONG; launch yourself). Per-dataset, per-attack (7B default):
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_tao.py      --dataset strongreject --num-steps 500 --batch-size 256
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_slotgcg.py  --dataset strongreject --num-steps 500 --search-width 512

# Multi-GPU: one saturating worker per GPU (near-linear). On a SINGLE GPU leave --num-workers 1.
.venv/bin/python 04_Scripts_Experiments/scripts/attacks/run_slotgcg.py  --dataset strongreject --num-workers 4 --gpu 0,1,2,3
```

### TAO AdvBench-50 paper protocol

The reproducibility launcher runs the two 7B threat models sequentially on one GPU:

```bash
# Download first (Llama-2 is gated and needs an accepted license + HF_TOKEN).
HF_HOME=/mnt/data/hf_cache .venv/bin/python 04_Scripts_Experiments/scripts/download_models.py \
    --only tao_vicuna_7b_v1_5 tao_llama2_7b_chat

bash 04_Scripts_Experiments/scripts/attacks/launch_tao_advbench.sh
```

For each model, behavior 50 is optimized for 1,000 steps from the 20-token `!` suffix. Its
result initializes the other 49 behaviors, each run for 500 steps with batch/top-k 256,
`K=N=3`, `tau=1`, `alpha=beta=0.2`, and `gamma=0.5`. The default local refusal-prefix check is
stored as `local_success` and the launcher stops a behavior when that local check passes; it is
not equivalent to the paper's GPT-4 Turbo plus human ASR.
Runs are resume-safe through the native per-behavior JSONL files.

Always pass `HF_HOME=/mnt/data/hf_cache HF_HUB_CACHE=/mnt/data/hf_cache/hub` (or your cache) so the
subprocess reuses the cached checkpoints.

### Parallelism (`--num-workers`)

GCG-family optimizers are single-behavior. `--num-workers N` splits the behaviors into N disjoint
contiguous slices run as concurrent processes, assigned round-robin over `--gpu` ids. **Measured:**
at `search_width=512` one worker already saturates a single GPU's compute, so two co-located
workers each run ~2× slower and total wall time is unchanged (no speedup). Use `--num-workers > 1`
**only** with multiple GPUs (`--gpu 0,1,2,3`), where it scales ~linearly.

### Cost note / measured runtime (7B Instruct, single 96GB GPU)

Benchmarked SlotGCG at the paper config (`num_steps=500`, `search_width=512`, `num_adv_string=20`):
**~3.3–5 s/step** (avg ~4 s), i.e. **~28–42 min/behavior** (model load amortized across a worker's
slice). For **StrongREJECT = 100 behaviors**:

- **~46–70 GPU-hours (~2–3 days)** on one GPU, sequentially. Co-located workers do **not** reduce
  this (compute-bound — see above).
- **÷ number of GPUs** if you spread workers across GPUs (e.g. 4 GPUs → ~12–17 h).
- Knobs that genuinely cut single-GPU time: lower `--num-steps` (e.g. 250), smaller `--search-width`
  (e.g. 256, ~halves per-step but may lower ASR), or enable early stopping on refusal in the method
  config to terminate easy behaviors early.

`main.tex` (§Sampling) targets 100 behaviors for the main run; the *completion* sampling budget
(64–128 base / 32–64 instruct / 32–64 attacked) is configured separately in the BRASS pipeline
(`n_completions`), not here — the attack produces one (or `--num-test-cases`) adversarial prompt
per behavior.

## What was patched (see `patch_external_repos.py` for exact diffs)

**TAO-Attack**
- `attack.py`: configurable `--data_path` / `--max_behaviors`; honor `--num_steps` (was hardcoded
  1000); OpenAI judge made optional (falls back to local refusal-prefix signal); register OLMo
  FastChat template; init `ms`/`cur_neg_idx`/`cur_neg_string` (referenced but undefined upstream);
  write a unified `tao_results.jsonl` with the final suffix.
- `opt_utils.py`: load OLMo in bf16 with the fast tokenizer.
- `string_utils.py`: route OLMo (ChatML) through the incremental slicing path (the legacy
  `.system`/`char_to_token` path used a removed FastChat API).
- `attack_manager.py`: generic `get_input_embeddings()` fallback for non-Llama models (OLMo3).

**SlotGCG**
- `model_utils.py`: make `ray`/`vllm` imports optional (only needed for the vLLM completion step);
  `torch_dtype=` → `dtype=` for transformers 5.x.
- Driven via a generated method config (`run_slotgcg.py`) with explicit OLMo `target_model`,
  `attn_implementation: eager` (so the VSS attention step returns weights), and
  `use_prefix_cache: False` (avoids the legacy KV-cache tuple layout).

## Output

`06_Results_Artifacts/results/attacks/tao/<tag>.json` and `06_Results_Artifacts/results/attacks/slotgcg/<tag>.json`, e.g.
`olmo31_instruct.json`. Point the pipeline at them:

AdvBench reproduction outputs are `06_Results_Artifacts/results/attacks/tao/vicuna_7b_v1_5.json` and
`06_Results_Artifacts/results/attacks/tao/llama_2_7b_chat.json`, with adjacent `_advbench_manifest.json` files.

```bash
.venv/bin/python -m brass.pipeline.run_experiment attack=tao \
    attack.cache_path=06_Results_Artifacts/results/attacks/tao/olmo31_instruct.json
```
