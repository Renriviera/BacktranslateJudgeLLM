# Preprocessing

[`prepare.py`](prepare.py) creates a new orbit input directory or prepares the fixed-response
judge cohort. It rejects existing destinations and destinations in the historical archive.
It records source hashes and performs no inference.

```bash
python bt.py prepare orbit --out 06_Results_Artifacts/new_runs/my-orbit-study
python bt.py prepare judge --out 06_Results_Artifacts/new_runs/my-judge-study
```

Other authoritative preprocessing entry points:

- [`attacks/build_attack_datasets.py`](../04_Scripts_Experiments/scripts/attacks/build_attack_datasets.py): benchmark-to-attack input conversion.
- [`orbits/fetch_benign.py`](../04_Scripts_Experiments/scripts/orbits/fetch_benign.py): benchmark acquisition and evaluator setup.
- [`orbits/build_manifest.py`](../04_Scripts_Experiments/scripts/orbits/build_manifest.py): group-aware pilot/main selection.
- [`orbits/freeze_execution.py`](../04_Scripts_Experiments/scripts/orbits/freeze_execution.py): freeze a reviewed protocol and execution queue.
- [`fp_robustness/build_variants.py`](../04_Scripts_Experiments/scripts/fp_robustness/build_variants.py): reproduce benign-adjacent perturbations.
- [`backtranslation_judge_near_duplicates.py`](../04_Scripts_Experiments/scripts/backtranslation_judge_near_duplicates.py): response-independent near-duplicate screening.

The original scripts keep their paths beneath the experiment directory for compatibility.
Invoke them through `bt.py script <path> --help`. Historical source snapshots intentionally
retain their original code; they are evidence, not the active implementation.
