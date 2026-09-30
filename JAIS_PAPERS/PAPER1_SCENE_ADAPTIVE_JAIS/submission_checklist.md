# Submission checklist — Paper 1 (Journal of Aerospace Information Systems, full-length paper)

| Item | Requirement | Status |
|---|---|---|
| Template | AIAA `new-aiaa.cls` with `[journal]`: 10 pt, one column, double spaced, letter paper | done (`manuscript.tex`) |
| Compiles | `pdflatex → bibtex → pdflatex ×2`, no errors, no overfull boxes | done (24 pages) |
| Title | ≤ 12 words, no abbreviations | 12 words: "Calibration Versus Scene Switching of Detector Operating Points for Aerial Multi-Object Tracking" |
| Abstract | one paragraph, 100–200 words, no references, no undefined abbreviations | 199 words, abbreviations spelled out |
| Nomenclature | symbols with definitions | done |
| Abbreviations | defined at first use in the text (SCI, NMS, MOTA, IDF1, HOTA, FPS, IDS) | done |
| Headings | Roman-numbered sections, lettered subsections (class default) | done |
| Figures | vector PDF, numbered, cited in order, captions self-contained | Figs. 1–6 done; Fig. 7 (scene examples) appears automatically once `figures/fig7_scene_examples.pdf` exists |
| Tables | numbered, cited, captions above | Tables 1–7 done |
| References | numbered in citation order (`new-aiaa.bst`), DOIs verified, journal versions preferred | 38 references, all from `refs/verified.json` |
| Conclusions | no new material | checked |
| Data/claims | every number traceable | `result_provenance.md` |
| Related manuscript | disclosed in cover letter | done |
| Author block | names and affiliations | Ahmed Gouda Ismail, Mohamed S. Mohamed (Military Technical College), Tarek Ahmed Mahmoud (Egypt University of Informatics), as in the authors' ICMISI 2026 paper; cities to be confirmed by the authors |
| AIAA membership footnote, funding, conflict-of-interest statement | journal forms | to be completed by the authors at submission |
| Length | AIAA guideline (Rev. Aug. 2024, `JAIS_PAPERS/refs/aiaa_rules/limits_2024.txt`): regular article 7–10 published pages, 10,000–12,000 words; = 20–26 double-spaced 10-pt serif manuscript pages, all pages counted | 26 manuscript pages (within 20–26) |
| Fees | AIAA: "The only potential fee associated with publication of journal articles is that required if publishing with an Open Access status" (voluntary APC $2,700; `refs/aiaa_rules/open_access.txt`) | no page or overlength charge; do not select Open Access to avoid any fee |

Build: `python JAIS_PAPERS/refs/make_bib.py manuscript.tex references.bib` (only if citations change), then
`pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript`.
Regenerate figures/tables: `python scripts/make_figures.py`, `python scripts/make_tables.py` (from any directory).
