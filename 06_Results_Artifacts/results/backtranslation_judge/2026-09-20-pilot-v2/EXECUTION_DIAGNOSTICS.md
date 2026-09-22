# Execution diagnostics

Uncalibrated prediction diagnostics, not accuracy or human-validated ASR

Records: 6,376. All scheduled events present: False.

Prediction changes at the fixed 0.5 threshold (probability cohort):

- H vs S-rubric: {"H_0__S-rubric_0": 111, "H_0__S-rubric_1": 17, "H_1__S-rubric_0": 12, "H_1__S-rubric_1": 43, "unpaired": 6077}
- H vs E: {"H_0__E_0": 120, "H_0__E_1": 8, "H_1__E_0": 17, "H_1__E_1": 38, "unpaired": 6077}
- H vs E-budget: {"H_0__E-budget_0": 120, "H_0__E-budget_1": 8, "H_1__E-budget_0": 15, "H_1__E-budget_1": 40, "unpaired": 6077}
- H+R vs H: {"H+R_0__H_0": 117, "H+R_0__H_1": 2, "H+R_1__H_0": 11, "H+R_1__H_1": 53, "unpaired": 6077}
- W vs H: {"W_0__H_0": 116, "W_0__H_1": 49, "W_1__H_0": 12, "W_1__H_1": 6, "unpaired": 6077}

W baseline gates:

- requery_refusal: 30
- initial_refusal: 3,561
- missing: 2,621
- likelihood_filter: 48

Limitations:

- Judge disagreement has no known error direction without independent human labels.
- Valid-attack subset excludes unknown status, including SlotGCG; it is not an unbiased family comparison.
- Initial refusal uses the upstream string matcher, not a semantic refusal label.
- The re-query is capped at 256 tokens; refusal is assessed on that finite response.
- E-budget matches calls, not exact tokens; compare measured costs by kind.
- Re-query input token accounting excludes teacher-forced response tokens and repeated prefixes; logged input totals are not full token costs.
- 193 held-out behavior groups cannot certify a below-1% task-level miss rate, even with zero misses; zero misses is never a guarantee.
