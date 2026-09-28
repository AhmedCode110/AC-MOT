# Submission checklist — Paper 1 (Journal of Aerospace Information Systems, full-length paper)

| Item | Requirement | Status |
|---|---|---|
| Template | AIAA `new-aiaa.cls` with `[journal]`: 10 pt, one column, double spaced, letter paper | done (`manuscript.tex`) |
| Compiles | `pdflatex → bibtex → pdflatex ×2`, no errors, no overfull boxes | done (24 pages) |
| Title | ≤ 12 words, no abbreviations | 10 words: "Scene-Adaptive Detector Operating-Point Control for Unmanned Aerial Vehicle Multi-Object Tracking" |
| Abstract | one paragraph, 100–200 words, no references, no undefined abbreviations | 179 words, abbreviations spelled out |
| Nomenclature | symbols with definitions | done |
| Abbreviations | defined at first use in the text (SCI, NMS, MOTA, IDF1, HOTA, FPS, IDS) | done |
| Headings | Roman-numbered sections, lettered subsections (class default) | done |
| Figures | vector PDF, numbered, cited in order, captions self-contained | Figs. 1–6 done; Fig. 7 (scene examples) appears automatically once `figures/fig7_scene_examples.pdf` exists |
| Tables | numbered, cited, captions above | Tables 1–7 done |
| References | numbered in citation order (`new-aiaa.bst`), DOIs verified, journal versions preferred | 38 references, all from `refs/verified.json` |
| Conclusions | no new material | checked |
| Data/claims | every number traceable | `result_provenance.md` |
| Related manuscript | disclosed in cover letter | done |
| Author block | name and affiliation | affiliation line must be completed by the author |
| AIAA membership footnote, funding, conflict-of-interest statement | journal forms | to be completed by the author at submission |
| Word count | full-length paper | about 9,900 words including tables and references (pdftotext count) |

Build: `python JAIS_PAPERS/refs/make_bib.py manuscript.tex references.bib` (only if citations change), then
`pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript`.
Regenerate figures/tables: `python scripts/make_figures.py`, `python scripts/make_tables.py` (from any directory).
