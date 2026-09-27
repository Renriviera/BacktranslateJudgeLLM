# Backtranslation

Research workspace for response-to-prompt inversion, repeated prompt/response cycles,
judge validation, and benign-task quality loss. The Python namespace remains `brass` 
to preserve existing imports.

## Start here

- [Experiment guide](04_Scripts_Experiments/EXPERIMENTS.md): preparation, inversion, controls, and new runs.
- [Judges and metrics](05_Validation_Metrics/README.md): prefix, StrongREJECT, Sol, drift, and quality.
- [Results guide](06_Results_Artifacts/README.md): archived runs, coverage gaps, and current scientific status.
- [Collaboration](CONTRIBUTING.md): two-person workflow and publishing.
- [Credential handling](SECURITY.md): scan scope and commit protection.

## Layout

```text
01_Datasets_Benchmarks/   Attack inputs, audit selection, benchmark provenance
02_Preprocessing/        Fresh-study preparation and input validation
03_Models/               Model registry and download entry point; no weights
04_Scripts_Experiments/  Python library, experiment scripts, Hydra configs, protocols
05_Validation_Metrics/   Metric documentation, tests, credential and artifact checks
06_Results_Artifacts/    Historical results, source snapshots, migration inventory
```

All runtime code lives in `04_Scripts_Experiments/src/brass`. Configuration lives in
`04_Scripts_Experiments/configs`; category guides link to the authoritative implementation.
There are no links back to the original BRASS checkout and no copied Git history.

## Setup

Use Python 3.11 or newer; validation was performed with Python 3.12. Use this folder
as the repository root. On a cluster, load/select a Python 3.12 module first. `uv` can
manage the project environment and install packages entirely in your user space; no
administrator access is needed. Point its cache at writable scratch if the home
directory is read-only.

```bash
git lfs install --local
git lfs pull
export UV_CACHE_DIR="${SCRATCH:-/tmp/$USER}/uv-cache"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev,analysis,sol,benchmarks]'
uv run --python .venv/bin/python pre-commit install
uv run --python .venv/bin/python bt.py --help
uv run --python .venv/bin/python bt.py test
uv run --python .venv/bin/python bt.py verify-artifacts
uv run --python .venv/bin/python bt.py audit
```

The offline publication scanner requires only Python's standard library. The full test
suite imports analysis libraries; optional tokenizer integration checks are explicitly
skipped when their local model cache/dependencies are absent. Some analysis dependencies
(notably `sentence-transformers`) pull in PyTorch. On clusters, confirm the available
GPU/CUDA stack before installing; to avoid GPU packages for CPU-only checks, install
`-e '.[dev,sol,benchmarks]'` instead. GPU generation requires the separate `inference`
extra; install that only in a GPU environment with a compatible cluster CUDA stack.
`requirements-attacks.txt` preserves historical attack dependencies; it is not a portable
CUDA lockfile. For the TAO runner on HPC, use the targeted installer below to add its extra
packages to the shared `.venv` without replacing the cluster's PyTorch/CUDA stack. Model
downloads remain explicit:

```bash
python bt.py models --list
uv pip install --python .venv/bin/python -e '.[inference]'
uv run --python .venv/bin/python bt.py models --only instruct_7b qwen3_32b
```

For the full PAP/StrongREJECT semantic reconstruction evaluation, submit the A100 array from
the repository root:

```bash
bash 04_Scripts_Experiments/scripts/semantic_reconstruction/submit_one_prompt.sh
```

The 313-item run used Qwen3-32B for PAP mutation/validity, OLMo-3-7B-Instruct as target, and
Qwen3.5-9B for response classification and semantic reconstruction. It produced 301 valid
PAP prompts and 12 mutation refusals; all 301 target responses were classified as refusals.
The detailed results and partial TAO follow-up are in
[`semantic_reconstruction_attack_results.md`](04_Scripts_Experiments/docs/semantic_reconstruction_attack_results.md).
The batch reads `HF_TOKEN` from `.env` and writes raw artifacts and logs under the ignored
`06_Results_Artifacts/new_runs/semantic_reconstruction/`. Set `HF_HOME` or `SCRATCH` to a
writable persistent cache location on the cluster.

TAO's targeted HPC dependencies can be added to the shared `.venv` with `uv` (no sudo):

```bash
bash 04_Scripts_Experiments/scripts/attacks/setup_tao_uv.sh
```

Credentials come from your environment or an untracked local `.env` copied from
`.env.example`. Set `HF_HOME`/`HF_HUB_CACHE` for your own cache. No weights, account
sessions, real `.env`, virtual environments, or original Git objects are included.
The full filesystem scan deliberately fails if a populated `.env` is present; use a
clean export for that check. The commit hook scans staged content and LFS objects.

## Run a new study

The launcher works from any current directory; relative arguments are interpreted
from this repository root. It never starts inference merely by being imported.

```bash
python bt.py prepare orbit --out 06_Results_Artifacts/new_runs/orbit-replication
python bt.py script orbits/preflight.py \
  --output 06_Results_Artifacts/new_runs/orbit-replication/preflight.json
python bt.py orbits \
  --run-dir 06_Results_Artifacts/new_runs/orbit-replication \
  --manifest 06_Results_Artifacts/new_runs/orbit-replication/manifest.json \
  --config 06_Results_Artifacts/new_runs/orbit-replication/config.json \
  --output 06_Results_Artifacts/new_runs/orbit-replication/core
```

This reuses the archived 491-item manifest and generation settings as a replication.
Preflight needs locally downloaded models. Review the config and compute budget before
generation. Judge-study preparation is independent:

```bash
python bt.py prepare judge --out 06_Results_Artifacts/new_runs/judge-replication
python bt.py judge run --run-dir 06_Results_Artifacts/new_runs/judge-replication
```

`python bt.py script <path> ...` exposes the remaining scripts, for example
`orbits/score_inverse_surprisal.py` or `attacks/pap_funnel.py`. Each keeps its own help.

## Evidence included

PAP (authority, logic, misrepresentation) and PAIR have archived evaluations on OLMo,
Vicuna, and Llama. SlotGCG has a full OLMo evaluation and a Llama smoke artifact; a
complete SlotGCG Vicuna/Llama comparison is not present. Additional TAO artifacts are
retained as historical context.

The archive contains the benign-adjacent minimal pairs, perturbations, recurrence study,
7B repeated-inversion study, matched 7B/32B comparison, and fixed-response judge study.
The active Sol reference scored 6,299 of 6,376 selected records; 77 remain unresolved.
Sol is an automated reference, not human gold. Historical failures and missing labels
are preserved. See the results guide before drawing conclusions.

Drift and original-task correctness are implemented. A validated affect/sentiment metric
and mechanistic activation interventions are not implemented; proposed extensions are
listed in the experiment guide. Refactoring did not run new attacks or paid inference.

## GitHub upload

Publish **this folder alone**, using a new Git repository. Large result JSON, JSONL,
and embedding NPZ files are configured for Git LFS. Use a Git/LFS push rather than
dragging this multi-gigabyte directory into the web uploader. See `CONTRIBUTING.md`.
An empty local Git repository and Git LFS are initialized. No remote, commit, or
publication is created by the migration.
