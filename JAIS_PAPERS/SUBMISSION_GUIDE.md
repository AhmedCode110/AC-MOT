# Submission guide (no fees)

| Paper | Journal | Folder | Fee |
|---|---|---|---|
| Scene-Adaptive Detector Operating-Point Control for Unmanned Aerial Vehicle Multi-Object Tracking | Journal of Aerospace Information Systems (AIAA) | `PAPER1_SCENE_ADAPTIVE_JAIS/` | none if Open Access is not selected (AIAA: the only fee is the voluntary Open Access charge) |
| Training-Free Self-Calibrating Control Layer for Heterogeneous Multi-Object Trackers | Signal, Image and Video Processing (Springer) | `PAPER2_SIVP_SPRINGER/` | none on the subscription route; the APC applies only if Open Access is chosen |

`PAPER2_UNIVERSAL_ACMOT_JAIS/` is the longer AIAA-format version of Paper 2; it is not submitted anywhere.

## Before submitting (both)
1. All three authors read the final PDF and agree to the author order, affiliations, e-mails, and the declarations.
2. Confirm the affiliations and cities ("Military Technical College, Cairo, Egypt"; "Egypt University of Informatics, Cairo, Egypt").
3. Confirm the Springer declarations (funding: none; competing interests: none; author contributions as written) or correct them.
4. Proofread the English.
5. Make the GitHub repository private until the papers are accepted. The repository contains the manuscripts themselves; similarity checkers index public web pages, and a public copy of the manuscript can produce a high similarity score or be read as prior publication.
6. Use of AI tools: both publishers require authors to disclose the use of AI tools in preparing a manuscript (Springer: document the use in the Methods or an equivalent section; AI-assisted copy editing alone is exempt). Decide with your supervisors and state it in the submission if tools were used. Undisclosed use discovered later can lead to retraction; disclosed use is accepted by both publishers. A neutral statement is: "Generative AI tools were used to assist with code, language editing, and manuscript formatting; the authors designed the study, verified all results, and take full responsibility for the content."

## Paper 1 — Journal of Aerospace Information Systems
1. Open https://arc.aiaa.org/journal/jais → "Submit a Manuscript" (AIAA ScholarOne Manuscripts), create an account for the corresponding author.
2. Article type: Full-Length Paper (regular article).
3. Upload: `manuscript.pdf`; the LaTeX source (`manuscript.tex`, `references.bib`, `new-aiaa.cls`, `new-aiaa.bst`, `figures/`, `tables/`) if requested; paste `cover_letter.md` into the cover-letter field.
4. Enter all three authors with e-mails; each receives a ScholarOne e-mail and must confirm, otherwise the submission is not complete.
5. Funding: enter "none" (or the grant, if any). Open Access: No.
6. Check the PDF that ScholarOne builds before the final "Submit".

## Paper 2 — Signal, Image and Video Processing
1. Open https://link.springer.com/journal/11760 → "Submit manuscript".
2. Article type: Original Paper (research article).
3. Upload `manuscript.tex`, `references.bib` (or the generated `manuscript.bbl`), `sn-jnl.cls`, `sn-mathphys-num.bst`, `figures/*.pdf`, `tables/*.tex`, and `manuscript.pdf`; paste `cover_letter.md`.
4. Journal limits checked: at most 10 pages in the final two-column format with only references on page 10 (the manuscript is compiled in that format: 10 pages, page 10 references only); abstract 150–250 words (249); Springer Nature LaTeX template (used).
5. Publishing model: choose the subscription (non-Open-Access) route.
6. Declarations are in the manuscript; the submission form repeats them.

## Checks already performed
- Every number traced to a result file (`result_provenance.md` in each folder).
- Every reference verified against Crossref or arXiv; one wrong DOI (ApproxDet) was found and removed; the reference builder now refuses any DOI whose title does not match.
- No figure contains Type 3 (bitmap) fonts; no LaTeX errors, overfull boxes, or undefined references.
- No text is copied from other publications; the two manuscripts share no results, tables, figures, or paragraphs.
