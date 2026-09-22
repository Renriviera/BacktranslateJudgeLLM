# Orbit experiment: recovered main-core results

The completed core run provides evidence of extra accumulated semantic drift for the selected
attack prompts. It does not yet establish a general property that jailbreak orbits are less
stable, or that stability is a useful jailbreak detector. Task type and attack family matter
substantially.

The run covered 491 prompts, eight sampled paths plus one greedy path per prompt, and six
round trips. It recorded 56,719 generation states and 32,120,872 generated target/inverse
tokens in about 6 hours 40 minutes. Technical validity remained high: 83 generations hit the
token cap and 20 inversions had invalid JSON. Failures stop a path and are not counted as
convergence or carried forward.

## Evidence from sampled trajectories

Distances use the pinned MiniLM encoder with full-text chunk pooling. Zero means identical
embeddings; larger cosine distance indicates more change. This is a geometric proxy, not
an information-theoretic quantity or a test of harmful task fulfillment.

We first average the eight paths within each prompt, then average relatives within their
underlying task group. The main comparisons use 100 attack behavior groups and up to 240
benign groups; 239 benign groups have a usable round-six endpoint. The 56,719 states are
not treated as independent statistical observations.

- **Additional response drift after the first cycle:** attacks add 0.126 distance units
  between rounds one and six, versus 0.101 for benign tasks. Difference **+0.0254**, pointwise
  95% task-bootstrap interval **[+0.0106, +0.0403]**, Holm-adjusted **p = 0.00565**.
  This is the clearest response-side signal beyond one-step backtranslation.
- **Total response drift at round six:** 0.342 for attacks versus 0.308 for benign tasks.
  Difference +0.0339, interval [+0.0016, +0.0669]. The unadjusted p value is 0.0427;
  after correcting across the six exploratory metrics, p = 0.171. This is weaker evidence.
- **Prompt drift at round six:** 0.556 versus 0.420. Difference **+0.1357**, interval
  **[+0.0986, +0.1722]**, adjusted p = 2.91e-11. Losing attack scaffolding or token noise
  can increase this metric without demonstrating loss of the underlying requested behavior.
- **Later step-to-step response movement, rounds four through six:** 0.09366 versus
  0.09304. Difference +0.00062, interval **[−0.00831, +0.00978]**, adjusted p = 0.894.
  There is no clear evidence here of persistently greater local instability.
- **Growth in variation across sampled paths:** difference +0.0130, interval
  **[−0.0199, +0.0463]**, adjusted p = 0.887. The overall comparison is inconclusive.
- **Mean step-to-step response distance across all six transitions:** difference +0.00639,
  interval [−0.00487, +0.01772], adjusted p = 0.816.

The intervals use 10,000 task-group bootstrap resamples. The p values use Welch tests on
independent task-group means, with Holm correction across the six listed metrics. These
specific comparisons were implemented after collection for this update. They are exploratory,
not a completed preregistered confirmatory analysis. Their intervals are pointwise rather
than simultaneous.

## Heterogeneity and controls

PAP's mean response drift is **0.154**, compared with **0.384 for PAIR** and **0.375 for
SlotGCG**. PAP contains 21 prompts representing 19 behavior groups. The proposition that
framing-heavy attacks are uniformly brittle is not supported by this pattern alone.

Benign **IFEval (0.530), ARC (0.427), and SQuAD (0.412)** drift substantially. GSM8K (0.123)
and HumanEval (0.195) are much more stable geometrically. Correct short answers may omit the
original question, and reconstruction can drop output constraints. Benchmark/source effects
are therefore central, not a minor nuisance.

For the 40 paired benign framing controls, response drift changes by **−0.0016
[−0.0236, +0.0191]**. Prompt drift rises by +0.0777 [+0.0380, +0.1163]. This suggests that
discarding a wrapper is easier to observe than an effect of that wrapper on response stability.
These standardized wrappers are not exact reproductions of actual attack scaffolding.

A sensitivity analysis matches 100 attack and benign task groups without replacement, within
a factor of two in both initial prompt and response length. The response-drift difference is
**+0.0886 [+0.0495, +0.1259]**. However, the matched benign sample includes only one SQuAD
and five ARC tasks, while including 25 GSM8K, 22 HumanEval and 20 BBH tasks. Matching changes
the benchmark mix and does not remove corpus, topic, or response-format confounding.

## Behavioral scoring is provisional

All selected attack prompts had an archived response judged positive by both scorers, but
that does not guarantee success on a fresh initial draw. Recovered StrongREJECT scoring
marked 623 of 1,048 sampled initial attack responses positive. Among positive-start paths,
the task-averaged positive fraction is approximately 47% after one cycle and 28% after six.
Those figures are **prefix-classifier diagnostics**, not validated harmful-behavior retention:
StrongREJECT sees the full response in only 12.8% of the scored attack/control states.
Conditioning on an initial positive also creates regression-to-the-mean concerns; the
fixed-prompt resampling baseline is needed to interpret the decline.

HarmBench's native context is 2,048 tokens. Full responses that do not fit are now explicitly
unscored, with null labels. They are neither truncated nor counted as negative. Human review,
complete benchmark evaluation and source-task semantic retention remain pending. The primary
geometric results above do not depend on these incomplete judgments.

## What happened operationally

Generation completed, but two implementation errors blocked later work: a JSONL reader split
valid Unicode U+2028 characters inside six records, and HarmBench was configured above its
native context limit. All 56,719 raw JSON records were intact. The reader is repaired and
regression-tested; raw generation hashes are unchanged. GPU embedding recovery agrees with
CPU embeddings to within 4.5e-7 in checked pairwise cosine similarities. Forty-two focused
tests pass after the repairs.

The alternate inverse, temperature-factor, deeper-orbit and additional resampling arms had
not run when this analysis was made. A documented recovery queue preserves the original
manifests, temperatures, seeds, model revisions and 80-million-token ceiling. The original
queue and its errors remain available for audit. Live recovery progress is recorded separately
from this snapshot; no robustness result is implied merely by queuing it.

The current interpretation is **extra drift with substantial family and task dependence**.
We still need the robustness arms, original-task retention labels, and held-out detection
comparisons to determine whether repeated mappings provide a useful jailbreak signal.

Artifacts:

- [Full numeric comparisons and source hashes](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/main_core/stability_update.json)
- [Per-task features](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/main_core/stability_prompt_features.json)
- [Main technical validity](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/main_core/validity_report.json)
- [Recovery amendment](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/analysis_recovery_amendment.json)
- [Recovery queue](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/execution_queue_recovery_v1.json)
- [Live progress](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/study_progress.json)
