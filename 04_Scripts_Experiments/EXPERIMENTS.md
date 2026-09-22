# Experiment workflow

## Research questions

1. How much information about the original request survives response-to-prompt inversion?
2. Does repeated inversion change attack behavior more than it changes benign task quality?
3. Do reconstruction-assisted judges improve agreement with independently adjudicated
   human labels, after accounting for judge computation and missing outcomes?
4. Which model representations causally contribute to refusal, task retention, or drift?

The first three have existing code and historical observations. The fourth requires
new activation capture and intervention work; the inherited activation/logit/deflection
metrics are stubs. Geometric drift alone is not a mechanistic explanation.

## Prepare and freeze

Run all commands from the Backtranslation root or through the absolute `bt.py` path.
Use `bt.py prepare orbit --out ...` for a replication with archived inputs, or the
dataset/manifest scripts for a new cohort. Preserve group-level splits and record which
data informed pilot decisions. `prepare judge --out ...` reconstructs the selected
fixed-response cohort from included attack archives and audit provenance.

Create a fresh preflight using local model caches. Freeze model IDs/revisions, templates,
seeds, generation limits, group splits, hypotheses, contrasts, and a maximum compute
budget. Existing queued commands record an earlier workstation and source freeze; they
must not be blindly replayed as a new execution plan. Do not bypass source-hash checks
to resume an old run with refactored code.

## Inversion and repeated prompting

[`orbits/runner.py`](src/brass/orbits/runner.py) implements the transition sequence
`x0 -> y0 -> x1 -> y1 -> ...`. The inverse receives the preceding response. It does not
receive the original task or the benchmark answer. Invalid inverse JSON, refusal,
context overflow, and truncation retain their observed status and terminate that path.
Checkpoint and token-budget ledgers support resumption with unchanged settings.

Use the core repeated-inversion arm with fixed-prompt resampling, equal-call resampling,
greedy/sampled forward and inverse variants, alternate inverse wording, deeper paths,
and known benign framing controls. Historical positive-start analyses are selected
analyses; compare them with resampling controls and report regression-to-the-mean risks.

Entry points: `bt.py orbits`, `bt.py script orbits/execute_study.py`, and
`bt.py script orbits/run_size_comparison.py`. Consult each `--help` before launch.
The 7B/32B comparison is a checkpoint comparison; size is not independently randomized.

## Judge comparison and benign-adjacent false positives

Run `bt.py judge --help` for preparation, inference, annotation export, calibration,
and analysis. Use the fixed original behavior/response pair for all reference judgments.
Evaluate prefix, StrongREJECT-ft, official rubric, and reconstruction-assisted arms
separately. Preserve cohort enrichment flags so diagnostic examples do not change the
probability-sample estimate.

`bt.py sol-api --help` and `bt.py sol-codex --help` expose the two historical Sol runtimes.
Their availability depends on the account and installed runtime. New calls require
credentials/subscription access and may incur usage. Existing policy-blocked inputs stay
unavailable; the refactor retains the no-retry/no-model-fallback guards.

Benign-adjacent work is in `scripts/fp_robustness`, `fp_robustness_recurrence.py`, and
`score_fp_minimal_pairs.py`. Preserve the distinction between a benign completion,
a mixed prompt-level distribution, and a classifier error. Constructed harmful-side
stubs in the minimal-pair set are not measurements of real harmful assistance ceilings.

## Quality loss and drift

Use `bt.py score-benign` to evaluate every observed benign response against its original
task. Prepare official IFEval dependencies and the isolated code evaluator before scoring.
Compare within-task changes from round zero alongside the attack cohort's change.
Report quality loss by benchmark, cycle, model, and decoding arm, with missingness and
task-group uncertainty. Do not use embedding similarity as a replacement for correctness.

Use `orbits/embed_orbits.py` for full-text embeddings, `bt.py orbit-analysis` for grouped
curves, and `bt.py size-comparison` for matched checkpoint contrasts. Inverse surprisal
is a separate diagnostic; two historical 7B arms failed that stage and need explicit repair.

Here “effect” means the measured change associated with a round or experimental arm.
If affect means emotion/tone, define and validate a separate annotation instrument before
using it: no affect labels or calibrated sentiment evaluator are supplied by this archive.

## Mechanistic extensions

Before adding hooks, specify a concrete intervention and readout: token/span alignment,
layer/module, cached activation identity, ablation/patching policy, and original-task
outcome. Use paired prompts and benign controls. Keep observation and intervention
results separate; verify hooks do not change baseline outputs when disabled.

Activation tensors can be large. Save selected, documented tensors in a new run, with
input hashes, model revision, precision, layer names, token positions, and sampling seed.
Do not include downloaded weights or credential-bearing runtime dumps. Add real tests
for hook behavior when implementing it; the inherited stubs are not a completed analysis.
