# Cue audit — final provenance status (2026-10-01)

Scope: every scientific statement in Paper 1 (`JAIS_PAPERS/PAPER1_SCENE_ADAPTIVE_JAIS`), Paper 2 / SIVP
(`JAIS_PAPERS/PAPER2_SIVP_SPRINGER`) and the thesis (`THESIS/`) that depends on E24, E25, E26, the S2 cue-utility
analysis (E33), or the cue selections of the learned-controller attempts (E34, E35).

Evidence files (all in `research/cue_audit/`; sha256 in `MANIFEST.md`):
- `stats_summary_E24_E26.json` — read-only extraction of the 336 per-sequence result pickles of E24 (224) and E26
  (112) on the development Mac (`outputs/opt_{audit,cues}/stats/`); `stats_files_E24_E26.json` lists every pickle
  with its sha256, stored fields (float64 as stored, integer counts, 19-alpha HOTA arrays) and per-sequence scalars;
  script `extract_stats_summary.py`. Pooled values use the project's `tools/seqstats.combine`; nothing was re-run and
  no interval was computed. The pickles themselves were not committed (development Mac only).
- `cue_benefit_audit.json` (E25), `s2_cue_utility.json` (E33, truncated), `v5_s2_cues_stdout.txt` (E33, stdout of
  the same run), `v5_s3/selection.json` + `v5_s3/v5_s3_full_stdout.txt` (E34), `v5_s3b/selection.json` +
  `v5_s3b/v5_s3b_full_stdout.txt` (E35), `opt_audit/grid.json`, `opt_cues/grid.json`.

Status classes: **A** directly artifact-verified; **P** protocol-record supported only; **PV** partially verified;
**U** unsupported.

| # | Claim (where) | Source artifact | Status | Remaining gap / action |
|---|---|---|---|---|
| 1 | Hand-designed index's resolution rule "no better than a uniform 640/736 mix at equal pixel cost" (Paper 1 and thesis, cue-audit paragraph; before this revision) | `stats_summary_E24_E26.json` → `E24.checks`: SCI pixel cost 0.722/0.724 (YOLOv8n/RT-DETR-L, fraction of 832²) between fixed 640 (0.592) and fixed 736 (0.783); SCI pooled HOTA 32.00/39.07 between fixed 640 (30.85/38.87) and fixed 736 (32.55/39.37) | PV | No mixture configuration exists in the grid (`mixture_configuration_in_grid: false`); equality with a mixture at equal cost is an interpolation recorded in Amendment 4, not a measurement. **Reworded** in Paper 1 and thesis to the measured facts. |
| 2 | Fixed 736 pixels scored above the SCI rule (new wording; registry E24) | same, `fixed_736_ge_sci_HOTA_both: true` (+0.55/+0.31 HOTA; fixed 736 uses 8% more pixels) | A | — |
| 3 | SCI sensitivity mapping "equals a constant 0.4" (registry E24, Amendment 4; not in the manuscripts) | same, `constant_sensitivity_0p4_vs_sci_delta_HOTA`: −0.07 (YOLOv8n), −0.15 (RT-DETR-L); a constant 0.5 scored +0.97/+0.24 | PV | "Equals" overstates; the stored results show a constant 0.4 within 0.15 HOTA. Registry left unchanged (historical record). |
| 4 | Per-frame audit: no causal cue reached \|ρ\| = 0.3 within sequences; signs change between sequences and detectors (Paper 1, thesis) | `stats_summary_E24_E26.json` → `E25`: maximum 0.279 (RT-DETR-L, uav0000268_05773_v, tiny-object ratio); all 8 cues have both signs over the 14 sequence-detector cells | A | Some cells hold NaN for the tiny-object ratio (no variance); stated in the file, not in the manuscripts. |
| 5 | No single cue nor the hand-designed index beat random allocation of the higher resolution on both detectors (Paper 1, thesis) | `E26.checks.cues_beating_random_HOTA_on_both_detectors: []`; random pooled HOTA 33.06 (YOLOv8n) / 39.71 (RT-DETR-L) | A | — |
| 6 | "At matched compute" (Paper 1, thesis; before this revision) | `E26.checks`: random pixel cost 0.803; cue rules 0.675–0.968; none within 0.01 of random. The rules were *designed* for about half of the steps (running-median split, `universal_policy_pipeline.py` at fc003bf) | U (as realized cost) | **Removed/reworded**: the manuscripts now state that the rules did not match the pixel budget and name the exceptions. |
| 7 | Exceptions on one detector: tiny-object ratio +1.26 HOTA on YOLOv8n (7/7 sequences) with a 19% larger pixel budget, −0.45 on RT-DETR-L; hand-designed index +0.04 on YOLOv8n (new wording) | `E26.versus_reference` (`c_tiny`, `c_legacy_sci`) and `mean_pixel_cost` | A | — |
| 8 | Registry E26 text "random 33.06/39.71 HOTA ≥ every cue incl. legacy SCI" and Amendment 4 "selection frequency 0/2 for every cue" | same | U (for YOLOv8n) | Random's values are exact, but on YOLOv8n two rules exceed it (#7). The definition of "selection frequency" was not recorded and no selection stage was run for this grid. Not used in the manuscripts; the historical records are left unchanged and this note supersedes them. |
| 9 | Cue utility: signal above chance on both detectors only for detector-output, tracker-state, and image-motion cues; none for image-appearance cues (Paper 1, thesis) | `v5_s2_cues_stdout.txt`: SIGNAL for det_count, det_gap, trk_survival, trk_match, img_motion, img_motion_resp; none for img_edges, img_brightness, img_blur in any of the four targets (resolution, sensitivity, gate τ, association offset) | A (stdout log) | The machine-readable JSON is truncated (#14); values available only at the log's 3-decimal precision. |
| 10 | Thesis: "the cues that carried signal described the detector output rather than the image" (ch. 4) and "the image cues did not identify … whereas cues describing the detector output did carry signal" (ch. 5) | same | PV → A after rewording | Image-motion cues also carried signal. **Reworded** to "detector output, the tracker state, and image motion rather than the appearance of the image" and "image-appearance cues (edge strength, brightness, blur)". |
| 11 | Individual training folds occasionally selected an image cue (Paper 1, thesis) | `v5_s3/v5_s3_full_stdout.txt` (gate τ: img_blur, img_brightness in two folds of attempt 1); `v5_s3b/selection.json` (img_blur for sensitivity in the fold holding out uav0000086_00000_v, attempt 2) | A | — |
| 12 | Learned controllers scored lower than fixed global parameters on held-out sequences, significantly only for YOLOv8n; interval values (Paper 1 qualitative, thesis with values) | attempt 2: `v5_s3b/v5_s3b_full_stdout.txt` (Δ of ½(HOTA+IDF1): −2.26 [−4.77, −0.38] YOLOv8n, 1/7 sequences ≥ 0; −0.60 [−2.15, +1.10] RT-DETR-L, 4/7) | PV | Attempt 1 (−2.42 [−4.86, −0.61]; −2.64 [−8.30, +0.57]) is recorded only in `research/OPTIMIZATION_PROTOCOL.md` Amendment 5b; its per-sequence outer-fold pickles exist on the Mac (`outputs/v5/s3/full/outer/`) but no stored aggregate. Paper 1 **reworded** "lost to" → "scored lower than … (significantly only for YOLOv8n in both attempts)"; thesis wording aligned. |
| 13 | Thesis: "the learned rules beat the global values on the training sequences in every fold and lost on the held-out ones" | `research/OPTIMIZATION_PROTOCOL.md` Amendment 5c ("training folds 7/7 yet lose on held-out sequences") | P | Not re-derivable from the stored selection files (some folds selected no cue). Unchanged; the thesis already attributes the intervals to the protocol record. |
| 14 | The adaptive-resolution gain of the hand-designed controller came mainly from its larger pixel budget (Paper 1 and thesis, audit paragraph and conclusions) | #1–#2 (pattern) + Paper 1 Table 5 (mean 722 vs 640 pixels, legacy evidence) | PV | The audit used a different controller; **reworded** from "added accuracy mainly by …; a uniform allocation … did as well" / "indicates" to "suggests … although the audit used a different controller". |
| 15 | "In this data, the scene cues do not identify the frames in which a different operating point pays off" (Paper 1, thesis) | #4 (A), #5 (A), #7 | PV | One cue exceeded random allocation on one detector at a larger pixel budget (#7), now stated next to it; wording kept. |
| 16 | SIVP: scene-driven controllers are tied to their detector and cannot observe a change of the detector's score scale (Related work) | none needed (structural argument) | not dependent on the cue audit | — |
| 17 | Thesis ch. 1: the scene controller is "structurally unable to see a shift of the detector's confidence scale" | none needed (structural argument) | not dependent | — |

## `s2_cue_utility.json` — search for a complete copy
- The file (225 bytes, sha256 `0dca4c12…a30`, modified 2026-09-27 08:03:28 +03:00) is truncated: the run's `json.dump`
  raised `TypeError: Object of type int64 is not JSON serializable`; the traceback is at the end of the stdout log
  `outputs/analysis/v5_s2_cues.txt` (same modification second). The run used a version of `tools/v5_s2_cues.py` without
  `default=float`; the committed version (62111e8) has the fix but was not re-run.
- Searched: the repository and its three worktrees (`Universal-ACMOT`, `-y0zkeh`, `-paper2`), all git refs (the file enters history
  only in 1a94cfb, as the same truncated copy), `outputs/`, `~/Desktop`, `~/Documents`, `~/Downloads`, `/private/tmp`, and the AC-MOT /
  Universal-ACMOT folders on the four local Google Drive mounts (`find` by name `*s2_cue*`, `*cue_utility*`, `v5_s2*`).
- **No complete copy of `s2_cue_utility.json` exists.** The only complete record of the run's values is its stdout log,
  committed unchanged as `v5_s2_cues_stdout.txt`; it is the same run, not a substitute experiment, but it carries only
  3-decimal values and is not the JSON artifact.

## Manuscript changes in this revision
Paper 1 `manuscript.tex` (cue-audit subsection and conclusions), thesis `ch4_scene_adaptive.tex` (cue-audit section
and conclusions) and `ch5_self_calibrating.tex` (motivation paragraph), as listed in #1, #6, #7, #10, #12, #14.
SIVP: no sentence depends on these experiments; unchanged. Paper 1 `result_provenance.md` rows for this section now
point to the files above.
