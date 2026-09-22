# Prompt–response orbit experiment

The approved protocol is in `04_Scripts_Experiments/docs/prompt_response_orbits_plan.md`. This pipeline writes only new
artifacts under `06_Results_Artifacts/results/orbits/`; it does not modify archived BRASS runs. Real inference requires
GPU visibility outside the Codex filesystem sandbox on this host.

The active study directory is `06_Results_Artifacts/results/orbits/2026-09-18-pilot`. Its `preflight.json` pins cached
model revisions. Public datasets and evaluator code have separate source hashes. Credentials
are read from the existing `.env` only by commands that need the model cache; never write them
to experiment artifacts.

Run commands from the repository root with `.venv/bin/python`:

1. `04_Scripts_Experiments/scripts/orbits/audit.py --output 06_Results_Artifacts/results/orbits/2026-09-18-pilot`
2. `04_Scripts_Experiments/scripts/orbits/preflight.py --output 06_Results_Artifacts/results/orbits/2026-09-18-pilot/preflight.json`
3. `04_Scripts_Experiments/scripts/orbits/rescore_archive.py --run-dir 06_Results_Artifacts/results/orbits/2026-09-18-pilot`
4. `04_Scripts_Experiments/scripts/orbits/fetch_benign.py --output 06_Results_Artifacts/results/orbits/2026-09-18-pilot`
5. `04_Scripts_Experiments/scripts/orbits/build_manifest.py --run-dir 06_Results_Artifacts/results/orbits/2026-09-18-pilot --stage shortlist`
6. `04_Scripts_Experiments/scripts/orbits/rescore_archive.py --run-dir 06_Results_Artifacts/results/orbits/2026-09-18-pilot --judge harmbench --manifest 06_Results_Artifacts/results/orbits/2026-09-18-pilot/screening_candidates.json`
7. `04_Scripts_Experiments/scripts/orbits/build_manifest.py --run-dir 06_Results_Artifacts/results/orbits/2026-09-18-pilot --stage pilot`

The pilot inference command is:

```bash
.venv/bin/python 04_Scripts_Experiments/scripts/orbits/run.py \
  --run-dir 06_Results_Artifacts/results/orbits/2026-09-18-pilot \
  --manifest 06_Results_Artifacts/results/orbits/2026-09-18-pilot/pilot_manifest.json \
  --config 06_Results_Artifacts/results/orbits/2026-09-18-pilot/pilot_config.json \
  --output 06_Results_Artifacts/results/orbits/2026-09-18-pilot/pilot_v1
```

For the length-calibration run, use `pilot_config_v2.json` and `pilot_v2` instead. Never reuse
an output directory with changed generation settings. The seed schedule and 96 prompt IDs
are identical between these two pilot configurations. Keep pilot source groups out of main
evaluation splits. Do not build or launch the main study until the pilot validity review passes.

The runner stores every observed generation, including malformed inverse output and truncated
responses. Those outcomes terminate that trajectory and are not carried forward as converged
states. A repeated command resumes completed requests. Token reservations cover worst-case
batch output before inference; interrupted reservations remain conservatively charged.

After generation, `summarize.py`, `score_benign.py`, `score_harmful.py`, and `embed_orbits.py`
accept `--manifest` and `--events-dir` (the latter three also take `--run-dir`). Benign code
scoring requires the immutable ID of the image built from `evaluator.Dockerfile`. The Docker
container has no network, host mounts, or writable root filesystem. Attack responses are never
executed. Missing rubric/human judgments stay unknown rather than being inferred from embeddings.

`progress.json` is live progress; `completion.json` means generation ended, not that the scientific
experiment passed. `validity_report.json` records numeric gates and pending qualitative review.
Review packets do not constitute completed human annotation. Truncation and inverse failure are
reported independently of harmfulness, correctness, and source-task retention.

CPU checks: `.venv/bin/pytest 05_Validation_Metrics/tests/test_orbits_*.py`.

The third pilot uses `pilot_config_v3.json` / `pilot_v3`, a 4,096-token forward cap and the
`explicit_json` inverse. It reuses identical initial responses from v2. Template changes
address formatting only; they do not make the inverse uniquely identifiable or establish
task faithfulness. The separate qualitative screening records observed task changes,
short-answer ambiguity and refusal-to-safe-alternative reconstruction. It is not human labeling.

After a complete pilot passes the recorded gates, `calibrated_pilot.json` identifies its
directory and config. `build_manifest.py --stage main` then constructs the held-out pool;
`freeze_execution.py --run-dir ... --code-image sha256:...` freezes main and diagnostic
manifests, config files and source hashes before main outcomes are examined. Start the queue
with `execute_study.py --queue 06_Results_Artifacts/results/orbits/2026-09-18-pilot/execution_queue.json`.

`study_progress.json` identifies the active stage, controller PID and worker PID. Per-batch
progress is in each job's `progress.json`; full stage logs are in `execution_logs/`. Sending
SIGTERM to the controller stops its current worker process group. Run the same queue again
to resume; uncompleted batches may retain conservative budget reservations. Do not edit frozen
manifests/configs or source files while the queue runs. A necessary code repair requires a
documented amendment and a new explicit source freeze; the controller refuses silent changes.

The frozen queue collects core, fixed-prompt, temperature, alternate-inverse, deep-extension,
equal-call and historical-success-start diagnostics. The latter have unverified historical
stop metadata and are not fresh replay. Optional reference/context/perturbation diagnostics,
human adjudication and confirmatory detection analysis remain separate outstanding stages.
`analyze.py` writes descriptive curves with task-group intervals and explicit missingness;
it does not declare a validated jailbreak detector. Exact new compute cost comes from the
shared ledger, not the sum of event token lengths in directories that reuse earlier events.
