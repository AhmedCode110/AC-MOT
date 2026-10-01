# Final consistency audit — Paper 1 (cross-pipeline integration)

Manuscript state audited: commit 6a85d1a; re-checked after the approved trims at 4e747a6 (manuscript.tex, tables/tab8_crosspipeline.tex).
Sources: `research/transfer_legacy/rescore/` (commit 97853c6, unchanged since; reproduction gates all true),
`research/paper_split/evidence/legacy/MATCHED_STATIC_A0_TESTDEV.json`, and the Drive files listed in
`result_provenance.md` and `research/transfer_legacy/U2MOT_SPARSETRACK_EVIDENCE_AUDIT.md`.
The numeric checks in §1–2 are made by `scripts/check_crosspipe_numbers.py`, which rebuilds every quoted string
from the JSON files and requires it to occur verbatim in the manuscript or Table 8.

## 1. Every number against its source JSON

| Item | Manuscript | Source field | Check |
|---|---|---|---|
| U2MOT HOTA | 54.94 vs 55.00 | `u2mot_testdev_rescore.json` `systems.*.trackeval_overall.HOTA` (54.940, 55.001) | OK |
| U2MOT ΔHOTA, CI, W/T/L, LOO | −0.06 [−0.31, +0.22], 9/0/8, −0.14 to +0.04 | `paired_controller_vs_baseline.metrics.HOTA` | OK |
| U2MOT motmetrics MOTA / IDF1 | 53.77 vs 53.87; 69.85 vs 69.78 | `systems.*.motmetrics_overall` (53.767/53.873; 69.849/69.775) | OK |
| SparseTrack HOTA | 69.10 vs 68.96; default 0.70 69.17 | `sparsetrack_val_half_rescore.json` `systems.*.trackeval_overall.HOTA` (69.099, 68.964, 69.167; 0.80: 68.916) | OK |
| SparseTrack vs 0.75 | +0.13 [+0.02, +0.24], 6/0/1, +0.05 to +0.18 | `pairs.adaptive_vs_static075.metrics.HOTA` (delta 0.1348) | OK |
| SparseTrack vs 0.70 | −0.07 [−0.54, +0.19], 4/0/3 | `pairs.adaptive_vs_static070.metrics.HOTA` | OK |
| SparseTrack motmetrics MOTA / IDF1 | 76.93 vs 76.89; 81.53 vs 81.49 | `systems.{adaptive,static075}.motmetrics_overall` | OK |
| Table 8 row 1 | −0.09 [−0.25, +0.06]; conf 0.35, NMS 0.35, 960 px | `MATCHED_STATIC_A0_TESTDEV.json` `results_V1_minus_A0_pp.HOTA`, `A0_system` | OK |
| Table 8 notes b, c (settings) | see table | Drive files named in `make_tables.py` comment and `result_provenance.md` | transcribed; checked against the audit file |

Abstract: 200 words (limit 200). No number from the new section appears in the abstract.

## 2. Confidence intervals

All cross-pipeline intervals are 95 % percentile intervals of a paired sequence bootstrap: rows 2–3 with 10,000
resamples, seed 0 (`bootstrap_resamples`, `seed` in both JSON files); row 1 with 5,000 resamples, seed 42
(`MATCHED_STATIC_A0_TESTDEV.json`). The table caption and Limitations item 7 state both. "Did not change
measurably" / "did not differ measurably" is used only where the interval contains 0 (U2MOT; SparseTrack vs 0.70;
row 1); "gain" only where it excludes 0 (SparseTrack vs 0.75).

## 3. Split labels

U2MOT: VisDrone2019 test-dev, 17 sequences (held-out). SparseTrack: MOT17 validation half, 7 sequences
(in-sample validation). Primary study labels unchanged. Matches the evidence audit.

## 4. Evaluator names

HOTA: TrackEval commit 12c8791 (U2MOT: TrackEval metric classes on the U2MOT-filtered boxes; SparseTrack: standard
MOT17 pedestrian preprocessing). MOTA/IDF1 in Sec. XIII: motmetrics. TrackEval MOTA/IDF1 are not quoted anywhere, so
the two evaluators are never mixed within a comparison. U2MOT evaluation stated as the U2MOT repository protocol,
not the official VisDrone toolkit (Sec. XIII, Limitations 2).

## 5. Held-out vs validation terminology

"Held-out" is used for the primary test-dev/UAVDT results and for U2MOT; SparseTrack is "in-sample validation"
everywhere (abstract "in-sample", Sec. XIII heading and text, Table 8 status, Limitations 1, Conclusions).
"The frozen rule has no held-out score" (MOT17 test unscored, MOT20 not run).

## 6. Frozen vs recalibrated policy wording

U2MOT controller: "separately recalibrated member of the same design family, not Q transferred unchanged";
note b gives the weights (Q's), transforms (Eq. heur), label-free terciles and hand-set operating points.
"The pipelines share an outcome, not one transferred controller." No sentence claims the same frozen controller
across pipelines (searched: "unchanged", "same controller", "transferred", "three pipelines", "across all").

## 7. Speed claims

No throughput is compared for the cross-pipeline rows ("No throughput is compared, because neither pipeline has a
matched timing on the evaluated split"). The U2MOT validation speed-up (+25.8 %) is not mentioned. Existing speed
claims of the primary study are unchanged.

## 8. Generalization claims

Generalization is claimed only for the primary controller (UAVDT zero-tuning, unchanged). For SparseTrack: "does not
establish held-out generalization". For U2MOT: no generalization claim beyond "did not change measurably at this
sample size".

## 9. Matched static vs author-calibrated static

Row 1: "matched static anchor". U2MOT: "author-calibrated static operating point ... not a matched static anchor"
(Sec. XIII, Table 8 row 2, Limitations 1). SparseTrack: "static NMS 0.75 (named in the freeze record)" and "the
default threshold of 0.70 in the tracker's configuration". Erratum recorded in the evidence audit §8: the U2MOT
author-calibrated point is not identical to the controller's hard tier (NMS 0.70 vs 0.60); the manuscript does not
claim it is.

## 10. Page count and JAIS format

Official build (workflow `manuscripts_build.yml`, run 36845999624, commit 6a85d1a, pdfTeX 1.40.25, TeX Live 2023;
report `JAIS_PAPERS/BUILD_REPORT.md`, PDF committed in 73ff93a): exit 0, 0 LaTeX errors, 0 undefined references or
citations, 0 overfull boxes, **28 pages** (26 before the integration). Template `new-aiaa.cls` `[journal]`, 10 pt,
one column, double spaced; Tables 1–8 numbered and cited; abstract 200 words; 42 references in citation order.
The AIAA guideline recommends 20–26 double-spaced pages for a regular article; it is a recommendation, not a hard
limit (`JAIS_PAPERS/refs/aiaa_rules/limits_2024.txt`), so 28 pages is 2 over the recommendation.

Trim candidates (not applied; require the authors' approval), estimated in double-spaced pages:
1. Introduction (844 words): drop the roadmap paragraph and merge the "last point shapes the reading" paragraph into
   the contributions (~0.6 page).
2. "Do the scene cues predict where switching helps?" (479 words): keep the two findings, move cue-by-cue numbers to
   the provenance file (~0.4 page).
3. Limitations (482 words): merge items 5 and 7 (sample size), shorten item 8 (provenance) (~0.3 page).
4. Table 8 notes b and c: move the operating-point lists to the text of Sec. XIII or to supplementary material
   (~0.3 page).
5. Move Table 6 (temporal grid) or Fig. 6 (per-sequence) to supplementary material (~0.5–0.7 page).
Items 1–4 together reach about 26 pages.

## Other checks

- References: 42 entries, all generated from `refs/verified.json`; the verify_refs run 36845358786 (commit 1e7c594)
  only reordered verified.json; regenerating references.bib gives the same 42 entries (order differs, which
  `new-aiaa.bst` ignores because it numbers by citation order).
- Cover letter: the SparseTrack/MOT17 overlap with the SIVP manuscript is disclosed (different methods and runs;
  baseline HOTA 69.17 here vs 68.88 there).
- `research/transfer_legacy/rescore/` byte-identical to commit 97853c6; sanitized copies in
  `research/transfer_legacy/rescore_public/` with a sha256 manifest.

## 11. After the approved trims 1–4 (commit 4e747a6)

Applied: Introduction roadmap removed and the reading paragraph condensed; cue-audit subsection tightened with every
number kept; Limitations "small validation set" and "statistical resolution" merged into "sample size", provenance
item shortened; Table 8 notes b and c condensed with every setting kept. No number, interval or claim changed;
limitations on the post hoc anchor, the custom VisDrone protocol, U2MOT recalibration and SparseTrack in-sample status
are unchanged. `scripts/check_crosspipe_numbers.py`: 17/17 OK. Abstract 200 words.

Official build run 36847293222 (commit 4e747a6, PDF in 0c78be4): exit 0, 0 errors, 0 undefined references, 0 overfull
boxes, **28 pages** (last page now ~340 words instead of ~380; about 27.8 filled pages).

Supplementary moves tested locally and not applied: removing Table 6 leaves 28 pages; removing Fig. 6 leaves 28 pages;
removing both leaves 28 pages (last page ~84 words); reducing all figure widths from 0.95 to 0.80 of the text width
leaves 28 pages. Because none of these reaches 26 pages, Table 6 and Fig. 6 were kept in the paper.
