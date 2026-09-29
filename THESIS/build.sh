#!/bin/sh
# Build the thesis: references, the Arabic pages (LuaLaTeX), then pdfLaTeX with BibTeX.
set -e
cd "$(dirname "$0")"
python3 ../JAIS_PAPERS/refs/make_bib.py . references.bib
(cd arabic && lualatex -interaction=nonstopmode arabic_summary.tex >/dev/null && rm -f arabic_summary.aux arabic_summary.log)
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
