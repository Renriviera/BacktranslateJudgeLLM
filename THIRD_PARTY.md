# Source and license provenance

The extracted first-party project declared MIT in `pyproject.toml`, but its working tree
did not contain a top-level license file. This export preserves source provenance and
does not invent a blanket license for the combined code, datasets, or outputs.

- **TAO-Attack**: `https://github.com/ZevineXu/TAO-Attack`, checkout
  `bd4238051e53830cb1429466dea0c3d2f2ece335`. The supplied `LICENSE` is retained under
  `04_Scripts_Experiments/src/brass/attacks/external/TAO-Attack`.
- **SlotGCG**: `https://github.com/youai058/SlotGCG`, checkout
  `d76c3c9b6bc1c2cfffd2906301d3945c9f9d4b56`. No top-level license file was found in
  this checkout. Embedded dependencies retain the notices present in their source files.
- **StrongREJECT rubric** and **backtranslation defense**: original license files and
  rubric/templates are retained in `04_Scripts_Experiments/src/brass/backtranslation_judge/vendor`.
- **Benchmark/evaluator sources**: source snapshots and provenance records are retained
  under the historical orbit studies. Follow each upstream dataset and evaluator license.

External Git repositories were exported as files, without their `.git` histories, model
weights, notebooks, or runtime output. The included external code contains the existing
BRASS patches; checkout revisions alone do not describe those modifications. The migration
manifest supplies hashes of the exact exported files.
