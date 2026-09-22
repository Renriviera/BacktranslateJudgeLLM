# Matched two-cycle OLMo 7B/32B experiment

This follow-up changes the model checkpoint while preserving the previous main-core prompts,
trajectory count, inverse wording and decoding settings. It asks whether the larger checkpoint
has less short-horizon drift and preserves more of the original task and framing.

The primary comparison uses all **491 original prompts**: 131 archived-success attack prompts
(PAP 21, PAIR 50, SlotGCG 60), 240 benign prompts across eight benchmarks, and 120 controls.
Each has **eight sampled trajectories and one greedy trajectory**, with **two round trips**:
`x0 → y0 → x1 → y1 → x2 → y2`. Existing 7B states through round two are reused without
regeneration. The new main arm uses 32B for both forward generation and inverse reconstruction.
The inverse receives only the preceding response, never the original prompt or benchmark answer.

## Checkpoints and controls

The reference is `allenai/Olmo-3-7B-Instruct` at
`6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`. The new checkpoint is
`allenai/Olmo-3.1-32B-Instruct` at `ac0587e4a7744a551c059d8cd17ba220bc940dae`.
The official 32B Instruct release uses the 3.1 name; see the
[Ai2 announcement](https://allenai.org/blog/olmo3) and
[model card](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct).

Their architecture, training and parameter counts differ. Consequently, this estimates a
**checkpoint difference associated with scale**, not the causal effect of parameter count alone.
It also does not assume that larger necessarily means more capable for every task.

Both arms use BF16, an 8,192-token context, forward temperature 1.0, inverse temperature 0.7,
top-p 1.0, forward limit 4,096 tokens, inverse limit 512 tokens and root seed 235711.
Per-call seeds are keyed by the same prompt, trajectory and stage identifiers. Matching seeds
does not make the models' random draws equivalent.

The native 32B system message and stop-token defaults differ from 7B. To remove that additional
confound, the new arm explicitly uses the pinned **7B chat template** and stop tokens
`[100265, 100257]`. The preparer checks rendered token IDs for all 491 original prompts and
an inverse-message example. This is a controlled comparison under the old experiment's context,
not an evaluation of the 32B checkpoint's untouched default chat settings.

## Reconstruction diagnostic

After the main arm, the same loaded 32B model reconstructs requests from the **identical saved
7B responses** used by the first two 7B inversions. The original inverse instructions, temperatures,
trajectory IDs and seeds are retained. There is no recursive extension beyond two cycles.
This distinguishes changes in inversion on a fixed response distribution from changes in the
responses produced by the larger forward model. It is not a full crossed-model orbit design.

## Outcomes fixed before 32B generation

The seven primary comparisons at cycle two, on sampled trajectories, are:

- Response drift from the initial response, separately for attacks and benign prompts.
- Prompt drift from the original prompt, separately for attacks and benign prompts.
- Similarity of reconstructed attack prompts to their original unwrapped behavior.
- Framing word retention and framing similarity on the 40 known benign framing controls.

For benign prompts, original-prompt similarity already measures the content reference; it is not
counted as an additional independent primary test. Exact parent tasks provide content references
for the framing controls. Framing is the exact text outside the known parent task, and framing
word recall excludes words already in that task. No automatic decomposition of actual attack
scaffolds is presented as ground truth. Thus evidence about these known wrappers is a controlled
proxy for the broader claim about jailbreak framing.

The same pinned, full-text chunk-pooled MiniLM encoder measures geometry for both models.
Lexical recall and cosine similarity are complementary proxies: the former penalizes paraphrases,
while the latter can reflect topic overlap without preserving a constraint. Neither is entropy,
semantic entailment, or proof of successful harmful behavior.

Trajectory IDs are paired first. Values are averaged within prompts, then within underlying
`analysis_group_id`, so related attack prompts and repeated paths do not inflate the number
of independent observations. Report paired task-bootstrap 95% intervals using 10,000 resamples
and paired t-tests with Holm adjustment across the seven primary tests. The fixed-response
first-inversion diagnostic has a separate four-test family: content similarity for attacks and
benign prompts, plus framing recall and similarity for known wrappers. Its second inversion,
greedy paths, benchmark/family breakdowns and other retention measures are descriptive.

Invalid, truncated and missing paths remain failures or missing measurements. They are never
filled forward or treated as convergence. Each comparison pairs valid states for that metric;
all-planned-path coverage is reported alongside it. A valid reconstruction can still be evaluated
when its subsequent forward response fails. Original-task benchmark scores and initial attack
success are recorded as diagnostics. All prompts remain in the study regardless of 32B attack
transfer success; success is not a post hoc selection criterion.

## Execution and outputs

Run root: `06_Results_Artifacts/results/orbits/2026-09-19-size-32b/`.
The frozen protocol and input/source hashes are in `frozen/`. A small two-task, two-cycle smoke
check must pass before the complete manifest runs. Generation is checkpointed after each batch.
The main arm and fixed-response diagnostic share one model load. GPU embedding, isolated benign
code evaluation, existing StrongREJECT/HarmBench scoring, and paired analysis follow in sequence.

The new study has a **64-million generated-token ceiling**. The worst-case main arm is 58,825,728
tokens; the fixed-response diagnostic is at most 4,525,056, leaving room for the smoke check.
These are maximums from generation caps, not an expected workload. This is a separate ledger
from the completed 7B experiment; 7B reuse costs no generation tokens. Judge calls retain the
existing 200,000-call hard ceiling. No paid API is used.

The controller writes `study_progress.json` and stage logs, stopping on a failed stage rather
than silently changing model, precision or prompts. The output includes `comparison_report.md`,
`comparison_summary.json`, `paired_features.json`, `fixed_inverse_features.json`,
`subgroup_comparisons.json`, `paired_curves.json`, `coverage.json`, and `response_drift.png`.
HarmBench overlength labels remain missing; StrongREJECT's prefix coverage remains a limitation.
Two cycles can test short-horizon retention and drift, not asymptotic stability or stationarity.

Launch or resume the frozen queue from the repository:

```bash
.venv/bin/python 04_Scripts_Experiments/scripts/orbits/run_size_comparison.py --run-dir 06_Results_Artifacts/results/orbits/2026-09-19-size-32b
```
