# Submission checklist — Paper 2 (Journal of Aerospace Information Systems, full-length paper)

## Title

Chosen (10 words, no abbreviations): **Training-Free Host-Preserving Control of Detector-Score Operating Points for Multi-Object Tracking**

Candidates considered:
1. Training-Free Host-Preserving Control of Detector-Score Operating Points for Multi-Object Tracking — *chosen*: names the three defining properties that the evidence supports (training-free, host preservation, score operating points) without implying universality.
2. Causal Self-Calibrating Control of Detector-Score Operating Points in Heterogeneous Multi-Object Tracking (12 words) — accurate; "heterogeneous" could be read as a generality claim.
3. Adaptation with Restraint: Causal Self-Calibrating Score Control for Multi-Object Tracking (10 words) — memorable; the slogan is better placed in the text.
4. Knowing When Not to Intervene: Training-Free Control of Detection Thresholds for Multi-Object Tracking (13 words) — exceeds the 12-word guidance.
5. A Frozen Causal Controller for Detector-Score Operating Points Across Heterogeneous Trackers (11 words) — emphasizes the protocol more than the method.

"Universal" is not used in the title or abstract.

## Format and content

| Item | Requirement | Status |
|---|---|---|
| Template | AIAA `new-aiaa.cls`, `[journal]` (10 pt, one column, double spaced) | done |
| Compiles | `tectonic manuscript.tex` (XeTeX; runs BibTeX internally); equivalently `pdflatex → bibtex → pdflatex ×2` | done: no errors, no undefined references or citations; one overfull box of 0.19 pt |
| Abstract | one paragraph, 100–200 words, no references | 192 words; HOTA defined; no citations |
| Nomenclature | symbols with definitions | done |
| Abbreviations | defined at first use (HOTA, IoU, ReID, COCO, MOTA, IDF1, DetA, AssA, IDS, FP, FN, W/T/L, CPU, GBI) | done |
| Figures | vector PDF (matplotlib, TikZ), numbered in order, captions self-contained | 10 figures: 1 fragility, 2 architecture (TikZ), 3 partition, 4 causal timeline (TikZ), 5 V6-TF vs V7f, 6 development, 7 external, 8 calibration shift, 9 runtime, 10 C-TWiX failure |
| Tables | numbered, captions above | 9 tables: 1 related work, 2 ablation, 3 systems, 4 development, 5 external, 6 KITTI, 7 calibration, 8 runtime, 9 reproduction audit |
| References | numbered in citation order, DOI or arXiv identifier verified | 46 entries (`scripts/make_bib.py`; see `source_audit.md`); one software reference; one companion-manuscript reference |
| Every number traceable | `result_provenance.md` | done |
| Hostile-review pass | `REVIEWER_DEFENSE.md` (20 required criticisms + 5 anticipated) | done |
| Related manuscript | disclosed in the cover letter; cited in Sec. V | done |
| Author block | name and affiliation | **affiliation line must be completed by the author** |
| AIAA membership footnote, funding, conflict-of-interest statement | journal forms | **to be completed by the author** |
| Data availability | statement at the end | done (repository tag and result commits) |

## Length — action needed

AIAA guideline (Rev. Aug. 2024, `JAIS_PAPERS/refs/aiaa_rules/limits_2024.txt`): regular article 7–10 published pages
(10,000–12,000 words or equivalent), i.e. **20–26** double-spaced 10-pt manuscript pages. The manuscript has
**29 pages** (about 7,500 words of running text plus 10 figures and 9 tables). The guideline is advisory ("a journal
editor … may request that a manuscript be shortened"), and AIAA has no page or overlength charge
(`refs/aiaa_rules/open_access.txt`). If the editor asks for 26 pages, the least damaging cuts, in order:
1. Move Fig. 9 (runtime bars; the same numbers are in Table 8) to supplementary material (≈ 0.4 page).
2. Move Table 2 (mechanism ablation) to supplementary material and keep one sentence in Sec. V (≈ 0.7 page).
3. Shorten Table 9 to the external systems only; the development-host reproductions stay in Table 4 (≈ 0.4 page).
4. Move Fig. 3 (nested partition illustration) to supplementary material (≈ 0.6 page).
Do not remove the failure analysis, the calibration-shift table, the reproduction classes, or the external table.

## Scope — risk to note

The evidence is on ground-level public benchmarks (MOT17, KITTI, DanceTrack). No aerial-platform result of the frozen
V7f exists (the post-freeze aerial run did not complete). The manuscript and the cover letter state this, and the
aerospace relevance is argued from the integration problem of separately qualified detectors and trackers. If the
editor judges the scope insufficient, the manuscript is written so that only the introduction's motivation paragraph
and the cover letter would change for a computer-vision or robotics journal.

## Paper 1 consistency

Paper 1's cover letter (`JAIS_PAPERS/PAPER1_SCENE_ADAPTIVE_JAIS/cover_letter.md`) refers to this manuscript by an earlier
tentative title ("Self-Calibrating Control for Heterogeneous Multi-Object Tracking in Aerospace Vision"). **Update that
line to the final title before submitting either manuscript.** Paper 2 now cites Paper 1 as a submitted companion
manuscript, as Paper 1's cover letter promises.

## Build and regeneration

```
cd JAIS_PAPERS/PAPER2_UNIVERSAL_ACMOT_JAIS
python scripts/make_tables.py        # tables/*.tex, tables/numbers.tex, tables/text_numbers.json
python scripts/make_figures.py       # figures/*.pdf (Fig. 3 (fig03_partition) needs --cache <outputs/det_cache_val_native>)
python ../refs/verify_refs_paper2.py # only if references change (network)
python scripts/make_bib.py           # references.bib (cited, verified entries only)
tectonic manuscript.tex
```

## Evidence produced for this manuscript (after the freeze, analysis only)

- Paired-bootstrap intervals for the 14 calibration-shift conditions: GitHub Actions run 36587453420,
  `research/final/paper2_calib_boot/` (lock 10/10; every re-run identical to the recorded pooled values;
  significance rule declared in commit 938aaff before the run). Result: 7 of 14 significant recoveries,
  0 significant degradations — this corrects the "8 of 14" in earlier summaries.
- `tests/test_v7f_paper2_claims.py`: 25 tests of the causality, reset, host-anchoring, and invariance
  statements on the frozen V7f configuration (all passed; no locked file changed).
