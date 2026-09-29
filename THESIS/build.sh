#!/bin/sh
# Build the thesis: references, then LuaLaTeX with BibTeX.
set -e
cd "$(dirname "$0")"
python3 ../JAIS_PAPERS/refs/make_bib.py . references.bib
lualatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
lualatex -interaction=nonstopmode main.tex >/dev/null
lualatex -interaction=nonstopmode main.tex >/dev/null
