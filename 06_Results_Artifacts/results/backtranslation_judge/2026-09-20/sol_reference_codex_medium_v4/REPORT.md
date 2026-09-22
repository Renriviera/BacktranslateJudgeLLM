# Open-model judge comparison against Sol

Reference: gpt-5.6-sol, reasoning effort medium, unchanged StrongREJECT rubric and score.

Scope: 6,376 selected records / 6,373 unique pairs. Attempted: 6,373 unique pairs. Inference complete: True. Every reference parsed: False.

**FP/FN below mean errors relative to Sol. They do not establish independent human truth.** All reference and candidate thresholds are 0.5; no tuning on these results.

Reference statuses: ok 6,299; invalid_output 34; api_policy_blocked 42; incomplete_multiple_messages 1.

## probability: 6,260 records

- **S-ft-archived:** FP 650/5,008; FN 613/1,178; candidate missing 0; Sol missing 74. Conditional FPR 12.98%; conditional FNR 52.04%. Full-selection conservative FPR 12.79%–14.25%; FNR 48.96%–54.87%.
- **S-ft:** FP 660/5,008; FN 605/1,178; candidate missing 0; Sol missing 74. Conditional FPR 13.18%; conditional FNR 51.36%. Full-selection conservative FPR 12.99%–14.44%; FNR 48.32%–54.23%.
- **S-rubric:** FP 925/5,008; FN 166/1,178; candidate missing 0; Sol missing 74. Conditional FPR 18.47%; conditional FNR 14.09%. Full-selection conservative FPR 18.20%–19.66%; FNR 13.26%–19.17%.
- **E:** FP 923/5,008; FN 239/1,178; candidate missing 4; Sol missing 74. Conditional FPR 18.43%–18.51%; conditional FNR 20.29%. Full-selection conservative FPR 18.16%–19.70%; FNR 19.09%–25.00%.
- **W:** FP 414/5,008; FN 1,001/1,178; candidate missing 32; Sol missing 74. Conditional FPR 8.27%–8.77%; conditional FNR 84.97%–85.57%. Full-selection conservative FPR 8.15%–10.09%; FNR 79.95%–86.42%.
- **H:** FP 903/5,008; FN 178/1,178; candidate missing 4; Sol missing 74. Conditional FPR 18.03%–18.07%; conditional FNR 15.11%–15.28%. Full-selection conservative FPR 17.77%–19.26%; FNR 14.22%–20.29%.
- **H+R:** FP 1,082/5,008; FN 138/1,178; candidate missing 4; Sol missing 74. Conditional FPR 21.61%–21.65%; conditional FNR 11.71%–11.88%. Full-selection conservative FPR 21.29%–22.79%; FNR 11.02%–17.09%.
- **E-budget:** FP 967/5,008; FN 231/1,178; candidate missing 5; Sol missing 74. Conditional FPR 19.31%–19.41%; conditional FNR 19.61%. Full-selection conservative FPR 19.03%–20.58%; FNR 18.45%–24.36%.
- **S-rubric AND W:** FP 76/5,008; FN 1,018/1,178; candidate missing 32; Sol missing 74. Conditional FPR 1.52%–2.02%; conditional FNR 86.42%–87.01%. Full-selection conservative FPR 1.50%–3.44%; FNR 81.31%–87.78%.
- **S-rubric OR W:** FP 1,253/5,008; FN 149/1,178; candidate missing 32; Sol missing 74. Conditional FPR 25.02%–25.52%; conditional FNR 12.65%–13.24%. Full-selection conservative FPR 24.66%–26.60%; FNR 11.90%–18.37%.

### H versus S-rubric

Common scored records: 6,182; unpaired: 78. Candidate minus baseline: ΔFPR **-0.42 pp** (95% behavior-bootstrap interval -2.02 pp to +1.15 pp); ΔFNR **+1.02 pp** (95% interval -2.31 pp to +4.49 pp). Lower is better.

Removed 402 FPs; introduced 381 FPs. Rescued 109 FNs; lost 121 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -0.60 pp to -0.28 pp; ΔFNR +0.48 pp to +1.85 pp. These are per-rate missing-label bounds, not confidence intervals.


### H+R versus S-rubric

Common scored records: 6,182; unpaired: 78. Candidate minus baseline: ΔFPR **+3.16 pp** (95% behavior-bootstrap interval +1.50 pp to +4.81 pp); ΔFNR **-2.38 pp** (95% interval -5.61 pp to +0.95 pp). Lower is better.

Removed 344 FPs; introduced 502 FPs. Rescued 126 FNs; lost 98 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR +2.98 pp to +3.31 pp; ΔFNR -2.95 pp to -1.61 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus S-ft

Common scored records: 6,182; unpaired: 78. Candidate minus baseline: ΔFPR **+4.87 pp** (95% behavior-bootstrap interval +2.55 pp to +7.11 pp); ΔFNR **-36.22 pp** (95% interval -41.73 pp to -30.68 pp). Lower is better.

Removed 385 FPs; introduced 629 FPs. Rescued 475 FNs; lost 49 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR +4.77 pp to +5.50 pp; ΔFNR -37.93 pp to -34.67 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus E

Common scored records: 6,178; unpaired: 82. Candidate minus baseline: ΔFPR **-0.42 pp** (95% behavior-bootstrap interval -1.54 pp to +0.66 pp); ΔFNR **-5.19 pp** (95% interval -7.58 pp to -2.76 pp). Lower is better.

Removed 265 FPs; introduced 244 FPs. Rescued 105 FNs; lost 44 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -0.54 pp to -0.14 pp; ΔFNR -6.06 pp to -4.51 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus E-budget

Common scored records: 6,177; unpaired: 83. Candidate minus baseline: ΔFPR **-1.30 pp** (95% behavior-bootstrap interval -2.43 pp to -0.20 pp); ΔFNR **-4.51 pp** (95% interval -6.86 pp to -2.19 pp). Lower is better.

Removed 298 FPs; introduced 233 FPs. Rescued 100 FNs; lost 47 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -1.44 pp to -1.00 pp; ΔFNR -5.38 pp to -3.87 pp. These are per-rate missing-label bounds, not confidence intervals.


### H+R versus H

Common scored records: 6,181; unpaired: 79. Candidate minus baseline: ΔFPR **+3.58 pp** (95% behavior-bootstrap interval +2.86 pp to +4.35 pp); ΔFNR **-3.40 pp** (95% interval -4.83 pp to -2.01 pp). Lower is better.

Removed 59 FPs; introduced 238 FPs. Rescued 58 FNs; lost 18 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR +3.48 pp to +3.67 pp; ΔFNR -3.81 pp to -3.04 pp. These are per-rate missing-label bounds, not confidence intervals.


## test: 3,860 records

- **S-ft-archived:** FP 357/3,108; FN 393/730; candidate missing 0; Sol missing 22. Conditional FPR 11.49%; conditional FNR 53.84%. Full-selection conservative FPR 11.41%–12.11%; FNR 52.26%–55.19%.
- **S-ft:** FP 365/3,108; FN 388/730; candidate missing 0; Sol missing 22. Conditional FPR 11.74%; conditional FNR 53.15%. Full-selection conservative FPR 11.66%–12.36%; FNR 51.60%–54.52%.
- **S-rubric:** FP 563/3,108; FN 103/730; candidate missing 0; Sol missing 22. Conditional FPR 18.11%; conditional FNR 14.11%. Full-selection conservative FPR 17.99%–18.69%; FNR 13.70%–16.62%.
- **E:** FP 586/3,108; FN 132/730; candidate missing 2; Sol missing 22. Conditional FPR 18.85%–18.92%; conditional FNR 18.08%. Full-selection conservative FPR 18.72%–19.49%; FNR 17.55%–20.48%.
- **W:** FP 250/3,108; FN 624/730; candidate missing 19; Sol missing 22. Conditional FPR 8.04%–8.49%; conditional FNR 85.48%–86.16%. Full-selection conservative FPR 7.99%–9.14%; FNR 82.98%–86.57%.
- **H:** FP 558/3,108; FN 96/730; candidate missing 4; Sol missing 22. Conditional FPR 17.95%–18.02%; conditional FNR 13.15%–13.42%. Full-selection conservative FPR 17.83%–18.59%; FNR 12.77%–15.96%.
- **H+R:** FP 689/3,108; FN 75/730; candidate missing 4; Sol missing 22. Conditional FPR 22.17%–22.23%; conditional FNR 10.27%–10.55%. Full-selection conservative FPR 22.01%–22.78%; FNR 9.97%–13.16%.
- **E-budget:** FP 618/3,108; FN 127/730; candidate missing 3; Sol missing 22. Conditional FPR 19.88%–19.98%; conditional FNR 17.40%. Full-selection conservative FPR 19.74%–20.54%; FNR 16.89%–19.81%.
- **S-rubric AND W:** FP 50/3,108; FN 633/730; candidate missing 19; Sol missing 22. Conditional FPR 1.61%–2.06%; conditional FNR 86.71%–87.40%. Full-selection conservative FPR 1.60%–2.75%; FNR 84.18%–87.77%.
- **S-rubric OR W:** FP 756/3,108; FN 94/730; candidate missing 19; Sol missing 22. Conditional FPR 24.32%–24.77%; conditional FNR 12.88%–13.56%. Full-selection conservative FPR 24.15%–25.30%; FNR 12.50%–16.09%.

### H versus S-rubric

Common scored records: 3,834; unpaired: 26. Candidate minus baseline: ΔFPR **-0.13 pp** (95% behavior-bootstrap interval -2.16 pp to +1.85 pp); ΔFNR **-0.96 pp** (95% interval -5.15 pp to +3.37 pp). Lower is better.

Removed 250 FPs; introduced 246 FPs. Rescued 74 FNs; lost 67 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -0.23 pp to -0.03 pp; ΔFNR -1.23 pp to -0.40 pp. These are per-rate missing-label bounds, not confidence intervals.


### H+R versus S-rubric

Common scored records: 3,834; unpaired: 26. Candidate minus baseline: ΔFPR **+4.09 pp** (95% behavior-bootstrap interval +1.95 pp to +6.26 pp); ΔFNR **-3.85 pp** (95% interval -7.97 pp to +0.42 pp). Lower is better.

Removed 204 FPs; introduced 331 FPs. Rescued 86 FNs; lost 58 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR +3.96 pp to +4.18 pp; ΔFNR -4.10 pp to -3.20 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus S-ft

Common scored records: 3,834; unpaired: 26. Candidate minus baseline: ΔFPR **+6.25 pp** (95% behavior-bootstrap interval +3.60 pp to +8.89 pp); ΔFNR **-39.97 pp** (95% interval -46.58 pp to -33.38 pp). Lower is better.

Removed 209 FPs; introduced 403 FPs. Rescued 307 FNs; lost 16 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR +6.14 pp to +6.42 pp; ΔFNR -40.41 pp to -38.69 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus E

Common scored records: 3,832; unpaired: 28. Candidate minus baseline: ΔFPR **-0.90 pp** (95% behavior-bootstrap interval -2.26 pp to +0.45 pp); ΔFNR **-4.95 pp** (95% interval -7.89 pp to -2.22 pp). Lower is better.

Removed 173 FPs; introduced 145 FPs. Rescued 60 FNs; lost 24 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -1.00 pp to -0.80 pp; ΔFNR -5.06 pp to -4.39 pp. These are per-rate missing-label bounds, not confidence intervals.


### H versus E-budget

Common scored records: 3,831; unpaired: 29. Candidate minus baseline: ΔFPR **-1.93 pp** (95% behavior-bootstrap interval -3.21 pp to -0.68 pp); ΔFNR **-4.26 pp** (95% interval -7.05 pp to -1.62 pp). Lower is better.

Removed 190 FPs; introduced 130 FPs. Rescued 57 FNs; lost 26 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR -2.06 pp to -1.79 pp; ΔFNR -4.51 pp to -3.73 pp. These are per-rate missing-label bounds, not confidence intervals.


### H+R versus H

Common scored records: 3,833; unpaired: 27. Candidate minus baseline: ΔFPR **+4.22 pp** (95% behavior-bootstrap interval +3.29 pp to +5.22 pp); ΔFNR **-2.89 pp** (95% interval -4.68 pp to -1.11 pp). Lower is better.

Removed 31 FPs; introduced 162 FPs. Rescued 34 FNs; lost 13 TPs (new FNs).

Allowing arbitrary labels for every missing reference and candidate: ΔFPR +4.12 pp to +4.28 pp; ΔFNR -3.15 pp to -2.53 pp. These are per-rate missing-label bounds, not confidence intervals.


## diagnostic: 155 records

- **S-ft-archived:** FP 51/100; FN 10/52; candidate missing 0; Sol missing 3. Conditional FPR 51.00%; conditional FNR 19.23%. Full-selection conservative FPR 49.51%–52.43%; FNR 18.18%–23.64%.
- **S-ft:** FP 51/100; FN 10/52; candidate missing 0; Sol missing 3. Conditional FPR 51.00%; conditional FNR 19.23%. Full-selection conservative FPR 49.51%–52.43%; FNR 18.18%–23.64%.
- **S-rubric:** FP 23/100; FN 2/52; candidate missing 0; Sol missing 3. Conditional FPR 23.00%; conditional FNR 3.85%. Full-selection conservative FPR 22.33%–25.24%; FNR 3.64%–9.09%.
- **E:** FP 6/100; FN 15/52; candidate missing 0; Sol missing 3. Conditional FPR 6.00%; conditional FNR 28.85%. Full-selection conservative FPR 5.83%–8.74%; FNR 27.27%–32.73%.
- **W:** FP 7/100; FN 44/52; candidate missing 0; Sol missing 3. Conditional FPR 7.00%; conditional FNR 84.62%. Full-selection conservative FPR 6.80%–9.71%; FNR 80.00%–85.45%.
- **H:** FP 9/100; FN 5/52; candidate missing 0; Sol missing 3. Conditional FPR 9.00%; conditional FNR 9.62%. Full-selection conservative FPR 8.74%–11.65%; FNR 9.09%–14.55%.
- **H+R:** FP 12/100; FN 5/52; candidate missing 0; Sol missing 3. Conditional FPR 12.00%; conditional FNR 9.62%. Full-selection conservative FPR 11.65%–14.56%; FNR 9.09%–14.55%.
- **E-budget:** FP 9/100; FN 11/52; candidate missing 0; Sol missing 3. Conditional FPR 9.00%; conditional FNR 21.15%. Full-selection conservative FPR 8.74%–11.65%; FNR 20.00%–25.45%.
- **S-rubric AND W:** FP 1/100; FN 44/52; candidate missing 0; Sol missing 3. Conditional FPR 1.00%; conditional FNR 84.62%. Full-selection conservative FPR 0.97%–3.88%; FNR 80.00%–85.45%.
- **S-rubric OR W:** FP 29/100; FN 2/52; candidate missing 0; Sol missing 3. Conditional FPR 29.00%; conditional FNR 3.85%. Full-selection conservative FPR 28.16%–31.07%; FNR 3.64%–9.09%.

## Probability cohort by attack variant

### pair

- S-rubric: FP 295/918 (32.14%); FN 34/319 (10.66%); missing Sol 15.
- H: FP 260/918 (28.32%–28.43%); FN 20/319 (6.27%); missing Sol 15.
- H+R: FP 312/918 (33.99%–34.10%); FN 13/319 (4.08%); missing Sol 15.
- S-ft: FP 90/918 (9.80%); FN 162/319 (50.78%); missing Sol 15.

### pap_authority

- S-rubric: FP 174/976 (17.83%); FN 41/257 (15.95%); missing Sol 19.
- H: FP 151/976 (15.47%); FN 57/257 (22.18%); missing Sol 19.
- H+R: FP 179/976 (18.34%); FN 47/257 (18.29%); missing Sol 19.
- S-ft: FP 162/976 (16.60%); FN 137/257 (53.31%); missing Sol 19.

### pap_logic

- S-rubric: FP 144/947 (15.21%); FN 37/287 (12.89%); missing Sol 18.
- H: FP 170/947 (17.95%); FN 47/287 (16.38%–16.72%); missing Sol 18.
- H+R: FP 200/947 (21.12%); FN 37/287 (12.89%); missing Sol 18.
- S-ft: FP 143/947 (15.10%); FN 146/287 (50.87%); missing Sol 18.

### pap_misrep

- S-rubric: FP 145/1,073 (13.51%); FN 37/163 (22.70%); missing Sol 16.
- H: FP 188/1,073 (17.52%–17.61%); FN 32/163 (19.63%); missing Sol 16.
- H+R: FP 256/1,073 (23.86%–23.95%); FN 22/163 (13.50%–14.11%); missing Sol 16.
- S-ft: FP 229/1,073 (21.34%); FN 90/163 (55.21%); missing Sol 16.

### slotgcg

- S-rubric: FP 167/1,094 (15.27%); FN 17/152 (11.18%); missing Sol 6.
- H: FP 134/1,094 (12.25%); FN 22/152 (14.47%–15.13%); missing Sol 6.
- H+R: FP 135/1,094 (12.34%); FN 19/152 (12.50%–13.16%); missing Sol 6.
- S-ft: FP 36/1,094 (3.29%); FN 70/152 (46.05%); missing Sol 6.

## Archived API versus Codex overlap

14 thresholded-label differences among 309 jointly scored unique pairs. Descriptive execution-order overlap. Both runtime differences and model stochasticity can cause disagreement; not a randomized or representative runtime-equivalence test.

## Interpretation limits

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

Per-record explanations and relative classifications: `assessments.jsonl`. Raw model outputs and token usage: `events.jsonl`. Exact blinded inputs: `requests.jsonl`.
