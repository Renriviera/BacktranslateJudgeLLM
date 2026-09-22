#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
(
  cd assets
  xelatex -interaction=nonstopmode -halt-on-error pipeline_original.tex
  xelatex -interaction=nonstopmode -halt-on-error pipeline_replacement.tex
)
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
