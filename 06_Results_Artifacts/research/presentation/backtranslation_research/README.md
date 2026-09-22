# Backtranslation research presentation

The editable 25-slide Beamer deck starts at main.tex. Compile with XeLaTeX.

Each numbered research section has one or two slides. Ten complete example slides follow, with five H false positives and five H false negatives relative to Sol, and two examples per attack variant. Three appendix slides contain the actual component prompts. The final slide records conclusions and references.

## Build

Run ./build.sh from this directory.

From the repository root, python3 scripts/build_backtranslation_presentation.py regenerates saved-data tables, plots, prompts and verified example exports. That step needs the experiment repository. All generated TeX files and figure PDFs are already included, so compiling the deck does not need the experiment repository or any model/API calls.

Dependencies: TeX Live with XeLaTeX, latexmk, Beamer, fontspec, PGFPlots, TikZ, standalone, adjustbox and xurl. Fonts: Noto Sans, Noto Sans CJK SC, Noto Sans Devanagari, Noto Sans Arabic and DejaVu Sans Mono.

The page is 360 × 202.5 mm (16:9). Full-text example slides are intentionally dense because each complete example occupies one slide. Use PDF zoom for detailed reading. The accompanying complete_examples.md and complete_examples.jsonl retain exact original text and newlines. Slide text reflows whitespace but does not summarize, repair or complete the recorded outputs.

## Scientific interpretation

- The recorded 7B experiment shows continued drift after cycle one, not stationarity. Later local movement is smaller and similar across cohorts.
- The 7B/32B comparison changes training and architecture as well as parameter count.
- H and H+R are offline ASR judges of the original attacked completion. R means re-query, not rubric. No deployed output gate was tested.
- The two pipeline figures distinguish retaining the original answer from returning a reconstructed answer. The benchmark evidence comes from repeated replacement in the orbit experiments, not a randomized direct evaluation of both pipeline deployments.
- All false-positive and false-negative labels are relative to GPT-5.6 Sol with the unchanged StrongREJECT rubric at medium reasoning. Sol can misjudge safe alternatives. Labels remain unchanged.
- The ten examples are selected illustrations, not a random sample. Missing scores remain unclassified. Recorded re-query truncation is explicit.
- The original attack archives and re-query outputs are copied from recorded artifacts. No new attack or judgment generation was performed for this deck.

## Evidence

provenance.json maps source artifacts to SHA-256 hashes and records verified example identities. exact_prompts.json preserves the templates. The supplied table screenshot appears unchanged as assets/wang_table2.png. Both pipeline illustrations have editable TikZ source in assets/.

Primary external references:

- Wang, Shi, Bai and Hsieh (2024), Defending LLMs against Jailbreaking Attacks via Backtranslation, Findings of ACL. https://aclanthology.org/2024.findings-acl.948/
- Robey, Wong, Hassani and Pappas (2023), SmoothLLM, arXiv:2310.03684.
- Souly et al. (2024), A StrongREJECT for Empty Jailbreaks, NeurIPS Datasets and Benchmarks, arXiv:2402.10260.
