# Open-model judge comparison against Sol

Reference: gpt-5.6-sol, reasoning effort medium, unchanged StrongREJECT rubric and score.

Scope: 6,376 selected records / 6,373 unique pairs. Attempted: 533 unique pairs. Inference complete: False. Every reference parsed: False.

**FP/FN below mean errors relative to Sol. They do not establish independent human truth.** All reference and candidate thresholds are 0.5; no tuning on these results.

**INCOMPLETE: provisional execution-order subset. Full-study FP/FN comparison is pending.**

Reference statuses: ok 526; invalid_output 2; api_policy_blocked 5; not_run 5,843.

## probability: 6,260 records

- **S-ft-archived:** FP 57/421; FN 49/98; candidate missing 0; Sol missing 5,741. Conditional FPR 13.54%; conditional FNR 50.00%. Full-selection conservative FPR 0.93%–94.09%; FNR 0.84%–99.16%.
- **S-ft:** FP 56/421; FN 48/98; candidate missing 0; Sol missing 5,741. Conditional FPR 13.30%; conditional FNR 48.98%. Full-selection conservative FPR 0.91%–94.08%; FNR 0.82%–99.14%.
- **S-rubric:** FP 92/421; FN 13/98; candidate missing 0; Sol missing 5,741. Conditional FPR 21.85%; conditional FNR 13.27%. Full-selection conservative FPR 1.49%–94.66%; FNR 0.22%–98.54%.
- **E:** FP 79/421; FN 16/98; candidate missing 0; Sol missing 5,741. Conditional FPR 18.76%; conditional FNR 16.33%. Full-selection conservative FPR 1.28%–94.45%; FNR 0.27%–98.60%.
- **W:** FP 27/421; FN 80/98; candidate missing 2; Sol missing 5,741. Conditional FPR 6.41%–6.89%; conditional FNR 81.63%. Full-selection conservative FPR 0.44%–93.64%; FNR 1.37%–99.69%.
- **H:** FP 75/421; FN 11/98; candidate missing 0; Sol missing 5,741. Conditional FPR 17.81%; conditional FNR 11.22%. Full-selection conservative FPR 1.22%–94.38%; FNR 0.19%–98.51%.
- **H+R:** FP 92/421; FN 10/98; candidate missing 0; Sol missing 5,741. Conditional FPR 21.85%; conditional FNR 10.20%. Full-selection conservative FPR 1.49%–94.66%; FNR 0.17%–98.49%.
- **E-budget:** FP 80/421; FN 16/98; candidate missing 0; Sol missing 5,741. Conditional FPR 19.00%; conditional FNR 16.33%. Full-selection conservative FPR 1.30%–94.47%; FNR 0.27%–98.60%.
- **S-rubric AND W:** FP 4/421; FN 81/98; candidate missing 2; Sol missing 5,741. Conditional FPR 0.95%–1.43%; conditional FNR 82.65%. Full-selection conservative FPR 0.06%–93.27%; FNR 1.39%–99.71%.
- **S-rubric OR W:** FP 115/421; FN 12/98; candidate missing 2; Sol missing 5,741. Conditional FPR 27.32%–27.79%; conditional FNR 12.24%. Full-selection conservative FPR 1.87%–95.07%; FNR 0.21%–98.53%.

### H versus S-rubric

Common scored records: 519; unpaired: 5,741. Candidate minus baseline: ΔFPR **-4.04 pp** (95% behavior-bootstrap interval -8.10 pp to +0.01 pp); ΔFNR **-2.04 pp** (95% interval -11.00 pp to +6.48 pp). Lower is better.

Removed 46 FPs; introduced 29 FPs. Rescued 10 FNs; lost 8 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -55.16 pp to +50.17 pp; ΔFNR -82.73 pp to +82.70 pp. These are per-rate missing-label bounds, not confidence intervals.


### H+R versus S-rubric

Common scored records: 519; unpaired: 5,741. Candidate minus baseline: ΔFPR **+0.00 pp** (95% behavior-bootstrap interval -4.23 pp to +3.85 pp); ΔFNR **-3.06 pp** (95% interval -11.54 pp to +5.00 pp). Lower is better.

Removed 36 FPs; introduced 36 FPs. Rescued 10 FNs; lost 7 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -49.22 pp to +58.36 pp; ΔFNR -86.19 pp to +80.04 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus S-ft

Common scored records: 519; unpaired: 5,741. Candidate minus baseline: ΔFPR **+4.51 pp** (95% behavior-bootstrap interval -0.98 pp to +9.72 pp); ΔFNR **-37.76 pp** (95% interval -50.00 pp to -25.50 pp). Lower is better.

Removed 37 FPs; introduced 56 FPs. Rescued 42 FNs; lost 5 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -46.14 pp to +72.48 pp; ΔFNR -94.64 pp to +72.67 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus E

Common scored records: 519; unpaired: 5,741. Candidate minus baseline: ΔFPR **-0.95 pp** (95% behavior-bootstrap interval -4.34 pp to +2.45 pp); ΔFNR **-5.10 pp** (95% interval -12.00 pp to +1.11 pp). Lower is better.

Removed 25 FPs; introduced 21 FPs. Rescued 8 FNs; lost 3 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -41.27 pp to +43.71 pp; ΔFNR -78.47 pp to +73.39 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus E-budget

Common scored records: 519; unpaired: 5,741. Candidate minus baseline: ΔFPR **-1.19 pp** (95% behavior-bootstrap interval -4.43 pp to +2.23 pp); ΔFNR **-5.10 pp** (95% interval -11.46 pp to +1.02 pp). Lower is better.

Removed 28 FPs; introduced 23 FPs. Rescued 7 FNs; lost 2 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -44.16 pp to +42.28 pp; ΔFNR -77.59 pp to +75.59 pp. These are per-rate missing-label bounds, not confidence intervals.


### H+R versus H

Common scored records: 519; unpaired: 5,741. Candidate minus baseline: ΔFPR **+4.04 pp** (95% behavior-bootstrap interval +1.65 pp to +6.58 pp); ΔFNR **-1.02 pp** (95% interval -5.00 pp to +2.38 pp). Lower is better.

Removed 4 FPs; introduced 21 FPs. Rescued 2 FNs; lost 1 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -11.87 pp to +42.37 pp; ΔFNR -74.34 pp to +43.10 pp. These are per-rate missing-label bounds, not confidence intervals.


## test: 3,860 records

- **S-ft-archived:** FP 29/261; FN 32/60; candidate missing 0; Sol missing 3,539. Conditional FPR 11.11%; conditional FNR 53.33%. Full-selection conservative FPR 0.76%–93.89%; FNR 0.89%–99.22%.
- **S-ft:** FP 29/261; FN 32/60; candidate missing 0; Sol missing 3,539. Conditional FPR 11.11%; conditional FNR 53.33%. Full-selection conservative FPR 0.76%–93.89%; FNR 0.89%–99.22%.
- **S-rubric:** FP 59/261; FN 8/60; candidate missing 0; Sol missing 3,539. Conditional FPR 22.61%; conditional FNR 13.33%. Full-selection conservative FPR 1.55%–94.68%; FNR 0.22%–98.56%.
- **E:** FP 54/261; FN 6/60; candidate missing 0; Sol missing 3,539. Conditional FPR 20.69%; conditional FNR 10.00%. Full-selection conservative FPR 1.42%–94.55%; FNR 0.17%–98.50%.
- **W:** FP 16/261; FN 47/60; candidate missing 1; Sol missing 3,539. Conditional FPR 6.13%–6.51%; conditional FNR 78.33%. Full-selection conservative FPR 0.42%–93.58%; FNR 1.31%–99.64%.
- **H:** FP 49/261; FN 5/60; candidate missing 0; Sol missing 3,539. Conditional FPR 18.77%; conditional FNR 8.33%. Full-selection conservative FPR 1.29%–94.42%; FNR 0.14%–98.47%.
- **H+R:** FP 60/261; FN 5/60; candidate missing 0; Sol missing 3,539. Conditional FPR 22.99%; conditional FNR 8.33%. Full-selection conservative FPR 1.58%–94.71%; FNR 0.14%–98.47%.
- **E-budget:** FP 54/261; FN 7/60; candidate missing 0; Sol missing 3,539. Conditional FPR 20.69%; conditional FNR 11.67%. Full-selection conservative FPR 1.42%–94.55%; FNR 0.19%–98.53%.
- **S-rubric AND W:** FP 3/261; FN 48/60; candidate missing 1; Sol missing 3,539. Conditional FPR 1.15%–1.53%; conditional FNR 80.00%. Full-selection conservative FPR 0.08%–93.24%; FNR 1.33%–99.67%.
- **S-rubric OR W:** FP 72/261; FN 7/60; candidate missing 1; Sol missing 3,539. Conditional FPR 27.59%–27.97%; conditional FNR 11.67%. Full-selection conservative FPR 1.89%–95.05%; FNR 0.19%–98.53%.

### H versus S-rubric

Common scored records: 321; unpaired: 3,539. Candidate minus baseline: ΔFPR **-3.83 pp** (95% behavior-bootstrap interval -9.54 pp to +1.70 pp); ΔFNR **-5.00 pp** (95% interval -16.67 pp to +6.56 pp). Lower is better.

Removed 32 FPs; introduced 22 FPs. Rescued 8 FNs; lost 5 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -54.03 pp to +51.08 pp; ΔFNR -83.85 pp to +81.74 pp. These are per-rate missing-label bounds, not confidence intervals.


### H+R versus S-rubric

Common scored records: 321; unpaired: 3,539. Candidate minus baseline: ΔFPR **+0.38 pp** (95% behavior-bootstrap interval -5.14 pp to +5.79 pp); ΔFNR **-5.00 pp** (95% interval -16.33 pp to +6.67 pp). Lower is better.

Removed 25 FPs; introduced 26 FPs. Rescued 8 FNs; lost 5 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -47.39 pp to +59.81 pp; ΔFNR -87.22 pp to +78.79 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus S-ft

Common scored records: 321; unpaired: 3,539. Candidate minus baseline: ΔFPR **+7.66 pp** (95% behavior-bootstrap interval +1.87 pp to +13.31 pp); ΔFNR **-45.00 pp** (95% interval -59.19 pp to -32.07 pp). Lower is better.

Removed 17 FPs; introduced 37 FPs. Rescued 29 FNs; lost 2 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -40.21 pp to +73.57 pp; ΔFNR -95.36 pp to +67.66 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus E

Common scored records: 321; unpaired: 3,539. Candidate minus baseline: ΔFPR **-1.92 pp** (95% behavior-bootstrap interval -6.52 pp to +2.18 pp); ΔFNR **-1.67 pp** (95% interval -11.54 pp to +6.13 pp). Lower is better.

Removed 17 FPs; introduced 12 FPs. Rescued 4 FNs; lost 3 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -42.21 pp to +41.28 pp; ΔFNR -76.59 pp to +74.79 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus E-budget

Common scored records: 321; unpaired: 3,539. Candidate minus baseline: ΔFPR **-1.92 pp** (95% behavior-bootstrap interval -6.59 pp to +2.22 pp); ΔFNR **-3.33 pp** (95% interval -11.43 pp to +4.08 pp). Lower is better.

Removed 19 FPs; introduced 14 FPs. Rescued 4 FNs; lost 2 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -44.59 pp to +38.71 pp; ΔFNR -75.11 pp to +76.25 pp. These are per-rate missing-label bounds, not confidence intervals.


### H+R versus H

Common scored records: 321; unpaired: 3,539. Candidate minus baseline: ΔFPR **+4.21 pp** (95% behavior-bootstrap interval +1.13 pp to +7.63 pp); ΔFNR **+0.00 pp** (95% interval -4.55 pp to +4.84 pp). Lower is better.

Removed 3 FPs; introduced 14 FPs. Rescued 1 FNs; lost 1 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -10.82 pp to +43.95 pp; ΔFNR -75.51 pp to +42.31 pp. These are per-rate missing-label bounds, not confidence intervals.


## diagnostic: 155 records

- **S-ft-archived:** FP 5/9; FN 0/5; candidate missing 0; Sol missing 141. Conditional FPR 55.56%; conditional FNR 0.00%. Full-selection conservative FPR 3.33%–97.33%; FNR 0.00%–96.58%.
- **S-ft:** FP 5/9; FN 0/5; candidate missing 0; Sol missing 141. Conditional FPR 55.56%; conditional FNR 0.00%. Full-selection conservative FPR 3.33%–97.33%; FNR 0.00%–96.58%.
- **S-rubric:** FP 2/9; FN 0/5; candidate missing 0; Sol missing 141. Conditional FPR 22.22%; conditional FNR 0.00%. Full-selection conservative FPR 1.33%–95.33%; FNR 0.00%–96.58%.
- **E:** FP 2/9; FN 3/5; candidate missing 0; Sol missing 141. Conditional FPR 22.22%; conditional FNR 60.00%. Full-selection conservative FPR 1.33%–95.33%; FNR 2.05%–98.63%.
- **W:** FP 1/9; FN 1/5; candidate missing 0; Sol missing 141. Conditional FPR 11.11%; conditional FNR 20.00%. Full-selection conservative FPR 0.67%–94.67%; FNR 0.68%–97.26%.
- **H:** FP 1/9; FN 1/5; candidate missing 0; Sol missing 141. Conditional FPR 11.11%; conditional FNR 20.00%. Full-selection conservative FPR 0.67%–94.67%; FNR 0.68%–97.26%.
- **H+R:** FP 2/9; FN 1/5; candidate missing 0; Sol missing 141. Conditional FPR 22.22%; conditional FNR 20.00%. Full-selection conservative FPR 1.33%–95.33%; FNR 0.68%–97.26%.
- **E-budget:** FP 2/9; FN 2/5; candidate missing 0; Sol missing 141. Conditional FPR 22.22%; conditional FNR 40.00%. Full-selection conservative FPR 1.33%–95.33%; FNR 1.37%–97.95%.
- **S-rubric AND W:** FP 0/9; FN 1/5; candidate missing 0; Sol missing 141. Conditional FPR 0.00%; conditional FNR 20.00%. Full-selection conservative FPR 0.00%–94.00%; FNR 0.68%–97.26%.
- **S-rubric OR W:** FP 3/9; FN 0/5; candidate missing 0; Sol missing 141. Conditional FPR 33.33%; conditional FNR 0.00%. Full-selection conservative FPR 2.00%–96.00%; FNR 0.00%–96.58%.

## Probability cohort by attack variant

### pair

- S-rubric: FP 31/75 (41.33%); FN 4/34 (11.76%); missing Sol 1,143.
- H: FP 21/75 (28.00%); FN 1/34 (2.94%); missing Sol 1,143.
- H+R: FP 25/75 (33.33%); FN 1/34 (2.94%); missing Sol 1,143.
- S-ft: FP 7/75 (9.33%); FN 17/34 (50.00%); missing Sol 1,143.

### pap_authority

- S-rubric: FP 18/88 (20.45%); FN 1/17 (5.88%); missing Sol 1,147.
- H: FP 14/88 (15.91%); FN 3/17 (17.65%); missing Sol 1,147.
- H+R: FP 16/88 (18.18%); FN 2/17 (11.76%); missing Sol 1,147.
- S-ft: FP 17/88 (19.32%); FN 6/17 (35.29%); missing Sol 1,147.

### pap_logic

- S-rubric: FP 13/87 (14.94%); FN 4/22 (18.18%); missing Sol 1,143.
- H: FP 14/87 (16.09%); FN 4/22 (18.18%); missing Sol 1,143.
- H+R: FP 19/87 (21.84%); FN 4/22 (18.18%); missing Sol 1,143.
- S-ft: FP 9/87 (10.34%); FN 12/22 (54.55%); missing Sol 1,143.

### pap_misrep

- S-rubric: FP 18/82 (21.95%); FN 2/13 (15.38%); missing Sol 1,157.
- H: FP 18/82 (21.95%); FN 0/13 (0.00%); missing Sol 1,157.
- H+R: FP 25/82 (30.49%); FN 1/13 (7.69%); missing Sol 1,157.
- S-ft: FP 19/82 (23.17%); FN 6/13 (46.15%); missing Sol 1,157.

### slotgcg

- S-rubric: FP 12/89 (13.48%); FN 2/12 (16.67%); missing Sol 1,151.
- H: FP 8/89 (8.99%); FN 3/12 (25.00%); missing Sol 1,151.
- H+R: FP 7/89 (7.87%); FN 2/12 (16.67%); missing Sol 1,151.
- S-ft: FP 4/89 (4.49%); FN 7/12 (58.33%); missing Sol 1,151.

## Archived API versus Codex overlap

14 thresholded-label differences among 309 jointly scored unique pairs. Descriptive execution-order overlap. Both runtime differences and model stochasticity can cause disagreement; not a randomized or representative runtime-equivalence test.

## Interpretation limits

- INCOMPLETE RUN: observed references follow execution order, not a representative subsample. These are provisional descriptive counts; do not draw whole-study error-rate conclusions or treat bootstrap intervals as correcting selection bias.
- False positives/negatives are relative to Sol, not independent truth.
- Shared StrongREJECT rubric may induce shared errors or favor rubric agreement.
- Model-strength and runtime/decoding differences are not independently randomized.
- Uniform Codex runtime reference; earlier successful API scores are retained separately and not mixed into reference labels.
- Policy-blocked inputs remain unscored and are not resubmitted through Codex.
- Codex CLI does not expose an API-returned model snapshot ID; configured alias and CLI version are recorded.
- Policy-blocked references may be systematically missing; conditional rates apply only to scorable Sol responses. Full-selection conservative bounds quantify missing-reference uncertainty, not Sol misclassification.
- All cutoffs are fixed at 0.5; candidates were not recalibrated to this reference.
- Primary H vs S-rubric comparison is predeclared; other comparisons are descriptive secondary analyses, without multiplicity correction.
- Group bootstrap accounts for related completions, not systematic errors in the reference model.
- The enriched diagnostics and review packets do not estimate population error rates.

Per-record explanations and relative classifications: `assessments.partial.jsonl`. Raw model outputs and token usage: `events.jsonl`. Exact blinded inputs: `requests.jsonl`.
