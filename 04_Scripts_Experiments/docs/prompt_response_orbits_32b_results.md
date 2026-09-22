# OLMo 32B: two-cycle comparison with 7B

The larger checkpoint reconstructs **benign task content modestly but measurably better**.
There is **no convincing improvement in framing retention**, and no statistically robust
reduction in attack response drift. An exploratory analysis finds a larger attack–benign
response-drift gap with 32B, but large benchmark differences remain.

The comparison uses the same 491 prompts, eight sampled paths plus a separate greedy path,
and two round trips. Forward temperature is 1.0 and inverse temperature is 0.7. Chat template,
stop tokens, generation caps and per-call seeds match the original experiment. The new model
is Olmo-3.1-32B-Instruct; the reference is Olmo-3-7B-Instruct. Their training and architecture
also differ, so this does not identify the causal effect of parameter count alone.

## Main findings

**Benign prompt reconstruction improves.** At cycle two, prompt drift falls from **0.3502
with 7B to 0.3153 with 32B**, a reduction of **0.0350 cosine-distance units**, approximately
10% relative to the 7B mean. The paired task-bootstrap 95% interval for 32B minus 7B is
**[−0.0476, −0.0229]**, and the seven-test Holm-adjusted p value is **4.51e-7**.
This is the only comparison that passes the seven-test correction among the prespecified
primary outcomes. Lower prompt drift is a geometric content-retention proxy, not a percentage
of the original information recovered.

**The same-response diagnostic confirms a reconstruction advantage for benign content.**
When both inverse models receive the identical saved 7B responses, first-inversion similarity
to the original benign task rises from **0.6979 to 0.7208**. Difference **+0.0229**, interval
**[+0.0173, +0.0286]**, adjusted p **1.63e-13** in the separate four-test diagnostic family.
Thus the benign-content improvement is not explained solely by differences in forward outputs.
This conclusion applies to reconstruction on this fixed distribution of 7B responses.

**Framing recovery remains weak.** For the 40 controls with exactly known benign wrapper text,
mean framing-word recall after two cycles is **3.28% for 7B and 2.40% for 32B**. The difference
is inconclusive after adjustment (p = **0.247**); framing embedding similarity is also
inconclusive. Both models discard most of the exact wrapper wording. Word recall penalizes
paraphrases, so low recall alone does not establish total semantic loss of framing. The
same-response diagnostic gives a small framing-similarity increase, but its corrected p value
is **0.0609**; that is not convincing evidence of an improvement. These controls do not provide
a gold decomposition of the actual PAP, PAIR or SlotGCG attack scaffolds.

**Response drift does not clearly improve with model size.** After two cycles, attack response
drift is **0.2662 for 7B versus 0.2818 for 32B**: difference +0.0155, interval
[−0.0007, +0.0319], corrected p **0.210**. Benign response drift is **0.2466 versus 0.2345**:
difference −0.0120, interval [−0.0227, −0.0018], corrected p **0.165**. The latter interval is
pointwise; its exclusion of zero does not override the multiplicity-adjusted p value.

Attack reconstructed-content similarity rises from 0.5492 to 0.5671, but this also fails the
primary correction (p = 0.193). A larger model is therefore not uniformly more stable or more
faithful across all prompt types under this process.

![Primary paired differences with confidence intervals](./results/orbits/2026-09-19-size-32b/primary_effect_sizes.png)

## Does 32B separate attacks from benign prompts better?

An **exploratory, post-outcome** comparison finds mean response drift **0.2818 for attacks
versus 0.2345 for benign tasks** with 32B: a gap of **+0.0472**, interval
**[+0.0165, +0.0778]**, p **0.00581** after adjustment across two exploratory contrasts.
The corresponding gap increases by **+0.0276 [+0.0082, +0.0472]** relative to 7B
(adjusted p **0.00650**).

This supports a short-horizon difference in accumulated semantic drift in this dataset.
It does not establish stationarity, local dynamical instability, a useful classification
threshold, or a broadly valid jailbreak detector. The contrast is not adjusted for benchmark,
topic, response format, or initial attack success. Selection used archived 7B-successful
attacks and does not guarantee that they transferred to 32B.

Family and benchmark differences are large. With 32B, PAP's response drift is **0.124**,
PAIR's **0.295**, and SlotGCG's **0.322**. Benign **IFEval (0.423)** and **ARC (0.383)**
drift more than the average attack. These are descriptive subgroup means. Lower response
drift may also reflect stable refusals or generic answers rather than task fulfillment.

## Original-task benchmark diagnostics

There is additional descriptive evidence of better task retention on some benchmarks. After
two cycles, **HumanEval+** has **68 confirmed passes among 240 planned sampled paths for 32B**,
versus **20/240 for 7B**. Ten 32B paths and eight 7B paths are unscored or have invalid/missing
generation states; these are not called incorrect. For **BBH**, the counts are **70/240 versus
33/240**, with two and ten unknown/invalid paths, respectively. These are descriptive diagnostic
counts, not additional prespecified significance tests. Confirmed-pass fractions on all planned
paths are lower bounds, rather than accuracy conditional on successful extraction.

Retention remains far from complete. HumanEval+ confirmed passes fall from 200 to 68 for 32B
and from 191 to 20 for 7B. IFEval falls from 180 to 39 for 32B and from 180 to 29 for 7B.
ARC has substantial answer-extraction missingness by cycle two, and SQuAD's strict final-span
exact-match rule is sensitive to response format. Those two datasets should not support a broad
claim about relative intelligence from these scores alone. Dolly and safety responses do not
have a complete correctness rubric here. Full counts and missingness are in
`benchmark_pass_counts.json` under the run directory.

## Provisional harmfulness diagnostics

StrongREJECT marks **491/1,048** valid initial sampled attack responses positive for 32B,
compared with **623/1,048** for 7B. At cycle two the counts are 216/1,045 and 326/1,047,
respectively. These are prefix-classifier diagnostics, not validated ASR or harmfulness-retention
estimates: the scorer sees the full initial response in only 463 of the 32B cases and 386 of
the 7B cases, and full-response coverage changes over cycles.

A post hoc sensitivity check restricted to paths initially positive in both models leaves
374 paths and 64 behavior groups. The attack response-drift difference is +0.0114
[−0.0102, +0.0345], with unadjusted p = 0.325. This does not establish a size effect on
attack response drift. Selection on noisy initial labels limits the diagnostic; the primary
analysis keeps all archived-selected prompts.

## Statistical and execution notes

Primary statistics pair the same trajectory identifiers, average within prompts, then within
underlying tasks: 100 attack behavior groups, 238 paired benign groups and 40 known framing
controls. Confidence intervals use 10,000 task-bootstrap draws. Primary tests use paired
task-level tests with Holm correction across seven comparisons. The same-response diagnostic
has its own four-test family. The two explicitly exploratory cohort contrasts use Welch tests
and independent task-group bootstrap intervals, with their own Holm correction. Neither
thousands of generations nor multiple attack variants of the same behavior are treated as
independent observations. Independent pandas aggregation reproduced every primary mean
contrast and unadjusted p value to numerical precision.

The main 32B run produced **21,969 states**, and the fixed-response diagnostic produced
**8,774 reconstructions**. Including the smoke check, generation used **8,618,328 tokens**.
Both generation arms finished in about **6 hours 48 minutes**. Main 32B states were 99.76%
technically valid, compared with 99.74% for the reused 7B states. Invalid and truncated paths
are censored, not filled forward or treated as convergence. Retention can be evaluated for
a valid inverse even if the subsequent forward response fails.

The controller stopped after generation without recording completion or starting post-processing;
the exact cause is not established. Recovery verified every expected main call, every fixed
inverse request and all frozen input/source hashes before resuming. Raw generations were not
modified or rerun. The primary geometry and reconstruction estimates are independent of the
benchmark and harmfulness labels. The scoring and paired-analysis queue completed at 23:44 EDT on September 19;
HarmBench overlength labels remain missing, and StrongREJECT may see only a response prefix.

The evidence supports **better preservation of benign task content**, with **no clear framing
advantage**. The increased attack–benign drift gap is promising for further testing, but it
needs held-out detection evaluation and stronger behavioral validation.

Artifacts:

- [Frozen protocol](../../06_Results_Artifacts/results/orbits/2026-09-19-size-32b/frozen/protocol.json)
- [Final paired comparisons and scoring coverage](../../06_Results_Artifacts/results/orbits/2026-09-19-size-32b/comparison_summary.json)
- [Exploratory cohort contrasts](../../06_Results_Artifacts/results/orbits/2026-09-19-size-32b/exploratory_attack_benign_contrasts.json)
- [Run recovery audit](../../06_Results_Artifacts/results/orbits/2026-09-19-size-32b/recovery_audit.json)
- [Live scoring and analysis status](../../06_Results_Artifacts/results/orbits/2026-09-19-size-32b/study_progress.json)
