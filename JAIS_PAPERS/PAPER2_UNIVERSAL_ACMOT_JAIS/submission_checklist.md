# Submission checklist — Paper 2 (Journal of Aerospace Information Systems, full-length paper)

| Item | Requirement | Status |
|---|---|---|
| Template | AIAA `new-aiaa.cls` `[journal]`: 10 pt, one column, double spaced | done |
| Compiles | pdflatex → bibtex → pdflatex ×2, no errors, no overfull boxes | done (20 pages) |
| Length | AIAA guideline (Rev. Aug. 2024, `JAIS_PAPERS/refs/aiaa_rules/limits_2024.txt`): 20–26 double-spaced 10-pt manuscript pages for a regular article | 20 pages |
| Fees | AIAA: the only potential publication fee is the voluntary Open Access charge (`refs/aiaa_rules/open_access.txt`) | none if Open Access is not selected |
| Title | ≤ 12 words, no abbreviations | 9 words |
| Abstract | one paragraph, 100–200 words, no references, no undefined abbreviations | 193 words |
| Nomenclature | symbols defined | done |
| Figures | 7 (architecture, prior design vs frozen layer, forest plot of every cell, per-sequence heatmap, confidence shift, regimes and failure, runtime) | done |
| Tables | 7 (host contracts, data and roles, MOT17 development, KITTI, external, ablation, runtime) | done |
| References | numbered, verified DOIs / arXiv | 43, all from `refs/verified.json` |
| Development vs external separated | Table 2, Development and Freeze Protocol, Fig. 3 groups | done |
| Negative external result shown | C-TWiX KITTI car −1.52 [−4.05, −0.13] in abstract, Table 5, Fig. 3, Failure Cases | done |
| Cold-start protection | future work only | done |
| Related manuscript | disclosed in cover letter; mentioned in text without shared results | done |
| Author block, AIAA forms | affiliation, membership, funding, conflict of interest | to be completed by the author |

Build: `python JAIS_PAPERS/refs/make_bib.py manuscript.tex references.bib`; `pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript`.
Regenerate: `python scripts/make_figures.py`, `python scripts/make_tables.py`.
