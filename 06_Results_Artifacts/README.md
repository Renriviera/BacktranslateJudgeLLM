# Historical results and artifacts

`results/` preserves research evidence from the BRASS working tree. `migration_manifest.json`
lists each source path, destination, source hash, export hash, byte size, and excluded file.
Use `python bt.py verify-artifacts` to verify the historical payload after an LFS checkout.
`artifact_index.json` gives a machine-readable study and attack-coverage index.

## Attack coverage

- PAP authority, logic, and misrepresentation: OLMo-3-7B-Instruct, Vicuna-7B-v1.5,
  and Llama-2-7B-Chat, including cached attack prompts, details, metrics, and funnels.
- PAIR: the same three target families, with attacker reports and per-target evaluations.
- SlotGCG: full OLMo results and `slotgcg_strongreject_llama2_7b_smoke`; no full Vicuna
  run or full Llama run was found in the source checkout.
- TAO: additional archived context, including native outputs and scorer records.

Attack preparation success, absence of refusal, and judged harmfulness are separate fields.
An archived positive response does not guarantee success on a fresh initial draw.

## Benign-adjacent cases

- [Minimal pairs](results/fp_robustness/minimal_pairs/findings.md): benign and redacted
  structure-only counterparts; preserve the distinction between prompt and completion scores.
- [Perturbations](results/fp_robustness/perturbations/findings.md): controlled changes in
  diagnostic placement, formatting, restatement, and length.
- [Recurrence](results/fp_robustness/recurrence/REPORT.md): replayed/regenerated completions
  across seeds and temperatures, including recovery of SlotGCG completion-level scores.

## Repeated inversion

[`orbits/2026-09-18-pilot`](results/orbits/2026-09-18-pilot) contains pilots, the 491-prompt
main study, 7B controls, manifests, original-task benchmark specifications, event traces,
embeddings, scores, budget ledgers, and frozen source snapshots. The main study used
eight sampled paths plus one greedy path and six round trips. The final queue status is
`queue_complete_with_errors`: inverse-surprisal scoring failed for `fixed_resampling`
and `archived_success_starts`. Optional diagnostics and human/confirmatory validation
remain outstanding. Earlier narrative documents are dated snapshots, not live status.

[`orbits/2026-09-19-size-32b`](results/orbits/2026-09-19-size-32b) preserves the matched
two-cycle 7B/32B comparison and fixed-7B-response/32B-inverse diagnostic. Its recorded
final verification confirms completed computation and independent aggregation checks.
See [comparison report](results/orbits/2026-09-19-size-32b/comparison_report.md).
This does not isolate a causal effect of parameter count or prove long-run stationarity.

## Judge study

[`backtranslation_judge/2026-09-20`](results/backtranslation_judge/2026-09-20) contains
the fixed-response cohort, open-model judge outputs, review packets, and historical
Sol versions. The final active reference is
[`sol_reference_codex_medium_v4`](results/backtranslation_judge/2026-09-20/sol_reference_codex_medium_v4).
It recorded 6,299 scored records and 77 unresolved records among 6,376 selected records.
See its [findings](results/backtranslation_judge/2026-09-20/sol_reference_codex_medium_v4/FINDINGS.md)
and [completion record](results/backtranslation_judge/2026-09-20/sol_reference_codex_medium_v4/completion_summary.json).

Sol uses the official StrongREJECT rubric and is an automated reference. Human gold
is still required: `validation_status.json` is `human_reference_required`. Policy blocks,
parse failures, and incomplete assessments are not negative judgments. Earlier API runs
remain separate from the uniform subscription reference. Review packets do not imply
completed independent annotations.

## Export boundaries

Weights, Git histories, environments, runtime logs, binary dataset caches, notebooks,
opaque ZIP/PDF bundles, and build byproducts are excluded. Textual sources, selected
benchmark inputs, source snapshots, numeric embedding arrays, and raw research traces
are retained. `research/` contains paper/presentation source material when available.
Explicit exclusions are recorded rather than silently treated as missing results.

Absolute paths inside historical records are retained for hash fidelity. Active code
has a resolver for known BRASS artifact paths; old execution queues are historical,
not portable launch instructions. New experiments belong in ignored `new_runs/` until
reviewed and archived under a fresh identifier.
