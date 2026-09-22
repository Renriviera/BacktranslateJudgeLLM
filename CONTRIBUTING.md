# Working together

Both collaborators should clone the same Backtranslation repository and install the local
commit hook with `pre-commit install`. Keep credentials in environment variables or each
person's ignored `.env`; share variable names and setup instructions, never values.

Use one branch per question and a pull request reviewed by the other person. Suggested
branch names are `experiment/<question>` and `fix/<issue>`. Record hypotheses, dataset
IDs, behavior-group splits, model revisions, judge versions, decoding, token limits,
and the intended analysis before launching an expensive run.

## Results and reproducibility

- Historical records under `06_Results_Artifacts/results` are evidence. Preserve their
  bytes, missing values, raw traces, source snapshots, and original run identifiers.
- Work in `06_Results_Artifacts/new_runs/<date>-<question>-<initials>`; it is ignored
  while jobs run. Never use an old completion marker to claim a new experiment finished.
- Reused manifests are replications, not new held-out data. Split by original task group
  before thresholds or model choices are fitted. Attack variants and trajectories from
  one task must remain in the same split.
- Commit reviewed results by copying the completed run into a new archive subdirectory,
  updating the results index, and adding a checksum inventory for the new run. Preserve
  the original migration manifest as the baseline, rather than rewriting its hashes to
  hide changes. Run the credential scan on the actual release contents.
- Use task-group intervals and explicit denominators. Report generation failures, missing
  judgments, and unknown human labels separately from negative outcomes.

## Local checks

```bash
python bt.py test
python bt.py verify-artifacts
python bt.py audit
```

The whole-tree scan is for a clean release checkout; it rejects a real `.env` even if
Git ignores it. The installed hook scans the Git index, including staged LFS objects,
so unstaged edits cannot conceal an earlier staged credential. CI scans the fetched
artifact payloads and runs the CPU suite. Inference and paid API calls are never CI steps.

## First publication

Run these commands from `Backtranslation`, after creating an empty remote repository
under your chosen GitHub account:

```bash
git init -b main
git lfs install --local
python bt.py audit
python bt.py verify-artifacts
git add .
python 05_Validation_Metrics/credential_scan.py --staged
git lfs status
git commit -m "Extract backtranslation research workspace and archived evidence"
# Add your own remote URL, then push main.
```

Do not copy the parent BRASS `.git`: it contains unrelated objects and is outside the
new export's credential audit. LFS stores the large traces separately; both collaborators
need access to those objects. Do not disable the LFS filter to work around a failed push.

Review upstream license notices before selecting a repository-wide license. The source
project declared MIT metadata, but no top-level license text was supplied. This export
does not grant a new blanket license over third-party code or benchmark datasets.
