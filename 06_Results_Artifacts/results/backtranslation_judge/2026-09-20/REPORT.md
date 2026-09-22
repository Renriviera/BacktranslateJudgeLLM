# Backtranslation judge execution report

Judge disagreements are not false positives or false negatives without adjudicated human labels.

Inference events: 56,437; generated tokens: 5,744,066.
Thresholds: uncalibrated 0.5. Human gold supplied: False.

Probability-sample predictions (missing outputs remain missing):

- S-ft-archived: 1231 positive, 5029 negative, 0 missing, out of 6260 selected.
- S-ft: 1249 positive, 5011 negative, 0 missing, out of 6260 selected.
- S-rubric: 1985 positive, 4275 negative, 0 missing, out of 6260 selected.
- E: 1900 positive, 4356 negative, 4 missing, out of 6260 selected.
- W: 589 positive, 5639 negative, 32 missing, out of 6260 selected.
- H: 1947 positive, 4309 negative, 4 missing, out of 6260 selected.
- H+R: 2169 positive, 4087 negative, 4 missing, out of 6260 selected.
- E-budget: 1952 positive, 4303 negative, 5 missing, out of 6260 selected.
- S-rubric AND W: 231 positive, 5997 negative, 32 missing, out of 6260 selected.
- S-rubric OR W: 2326 positive, 3902 negative, 32 missing, out of 6260 selected.

These are predicted success counts, not validated attack successes. Do not interpret lowered predicted ASR as improved FPR.

Files: analysis.json, predictions.jsonl, review_queue.json, events/, run_spec.json, annotations/.
