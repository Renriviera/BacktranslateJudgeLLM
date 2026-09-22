# Prompt–response orbit experiment: execution record

**Results update:** the main generation run finished. Post-processing failures were recovered;
see [the current results and limitations](prompt_response_orbits_results_update.md). The record
below describes the initial launch, rather than the latest live status.

The approved experiment has started on the local RTX PRO 6000 GPU using the pinned
`allenai/Olmo-3-7B-Instruct` checkpoint for both forward generation and reconstruction.
The computational queue is running; the scientific experiment is not complete.

The 96-prompt pilot finished four round trips with three sampled trajectories and one greedy
trajectory per prompt. The calibrated run recorded 3,442 generations, including 1,529
attempted inversions. Every attempted inversion produced a usable prompt. Two benign forward
responses were truncated; their trajectories stopped rather than being counted as convergence.
The technical gates passed in all three broad cohorts.

Two pilot-only repairs were necessary. A 1,024-token response cap caused excessive truncation,
so the forward cap is now 4,096 tokens. With longer responses, the first inverse instruction
sometimes produced empty output or incorrect JSON. Explicit schema wording and a repeated
instruction after the quoted response resolved that formatting problem on the pilot. The
same 384 initial responses were reused when testing the revised inverse instruction. The
inverse still receives only the preceding response, never the original task or its labels.

The main cohort contains **491 prompts**: 21 PAP, 50 PAIR and 60 SlotGCG attacks; 240 benign
tasks across eight strata; and 120 controls. Attack eligibility requires agreement between
StrongREJECT and HarmBench on at least one archived individual response. The frozen screening
pool and pilot exclusions left fewer PAP/PAIR prompts than the original maximum quotas;
the success criterion was not relaxed. Labels remain judge-provisional.

The main core uses eight sampled trajectories and one greedy trajectory, with six round
trips. The queue also includes fixed-prompt resampling, two temperature factorizations,
an alternate reconstruction instruction, a 48-prompt extension to twenty rounds, an equal-call
resampling comparison, and 224 archived successful response starts. All arms share the hard
**80-million generated-token ceiling** and **200,000 judge-call ceiling**. The larger calibrated
response cap makes worst-case completion exceed the token allowance; the runner will stop
before exceeding the ceiling and score the observed data. No budget expansion is automatic.

Pilot task groups are excluded from the main pool. Native task/behavior/passage groups and
a frozen lexical near-duplicate check define statistical grouping; the main set has 340
groups, including 100 attack behavior groups. Semantic paraphrases can escape this lexical
check. Thirty-eight of forty benign framed controls match their designated attack's prompt
length within 24 target tokens; the other two are explicitly flagged. Scaffolding is
standardized, not an exact reproduction of attack framing.

The qualitative screen inspected sixty reconstructed prompts with response/task excerpts
and additional complete short-response checks. It found important limitations: “Ten” and “B”
do not identify their original questions; refusals can become requests for the offered safe
alternative; and code interfaces or multiple-choice requirements can disappear. These are
recorded confounds, not proof of a jailbreak-specific information signal. This was an agent
screen, not human annotation. Independent human labels, remaining reference/context/perturbation
diagnostics, and confirmatory detection analysis are still pending.

The pilot precision simulation uses only task-level geometric variability. Its illustrative
95% interval for a centered main-group difference is approximately ±0.021 cosine-distance
units. It does not establish power for the proposed 0.05 AUROC improvement or for retention
differences. PAP-specific and low-false-positive-rate comparisons will have limited precision.

Validation passed 37 focused tests, including fresh-context isolation, parser failures,
budget enforcement, resume behavior, diagnostic prefix reuse, grouped uncertainty, duplicate
grouping and benchmark extraction. All nineteen audited historical input files retain their
original hashes. Benign benchmark code executes only in a restricted container without
network access or host mounts.

Live status and evidence:

- [Active queue stage](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/study_progress.json)
- [Main generation progress](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/main_core/progress.json)
- [Pilot validity report](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/pilot_v3/validity_report.json)
- [Qualitative review and limitations](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/pilot_v3/reconstruction_review_decision.json)
- [Frozen queue and source hashes](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/execution_queue.json)
- [Main manifest](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/main_manifest.json)
- [Compute ledger](../../06_Results_Artifacts/results/orbits/2026-09-18-pilot/token_budget.jsonl)
- [Run and resume instructions](../scripts/orbits/README.md)

`completion.json` means generation finished for one arm. It does not mean that the research
hypothesis was supported, the detector was validated, or human review was completed. The
queue performs local computation and records its state; it does not send notifications.
