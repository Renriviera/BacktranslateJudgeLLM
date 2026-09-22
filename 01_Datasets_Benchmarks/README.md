# Datasets and benchmarks

`data/attacks` contains the original PAP/PAIR templates and converted StrongREJECT inputs
for SlotGCG and TAO. `provenance/audit_selected.json` and `audit_candidates.json` preserve
the selection used in the benign-adjacent and fixed-response judge work.

The registry and loaders remain in
[`brass/data/loaders.py`](../04_Scripts_Experiments/src/brass/data/loaders.py).
Hydra dataset definitions are in
[`configs/dataset`](../04_Scripts_Experiments/configs/dataset).

The repeated-inversion manifest is frozen in
[`main_manifest.json`](../06_Results_Artifacts/results/orbits/2026-09-18-pilot/main_manifest.json).
It contains 131 selected attack prompts, 240 benign prompts, and 120 controls. Benign
sources include GSM8K, BBH, HumanEval, ARC, SQuAD, IFEval, Dolly, XSTest, and OR-Bench;
XSTest and OR-Bench jointly supply the benign-adjacent stratum. The source acquisition
records and hashes are preserved beside the manifest in `dataset_provenance.json`.

Raw Hugging Face Arrow caches are excluded. Frozen selected inputs, original-task
evaluation specifications, source hashes, and the acquisition script are included.
Use `python bt.py script orbits/fetch_benign.py --help` for a fresh benchmark acquisition.
Dataset access and upstream licenses remain applicable.

Keep original behavior/task IDs across attacks and inverse rounds. Semantic or exact
duplicates must share a group before making development, calibration, and test splits.
