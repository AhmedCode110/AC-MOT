# OATrack-comparison AC-MOT experiment: final report (VisDrone)

## 1. What was executed

1. Selected and froze a publicly available pretrained YOLO11m detector
   (no in-house training) as the common detector for all four systems.
2. Generated the full detection cache (3 resolutions x 3 NMS IoU values,
   one forward pass per frame/resolution) for the 7-sequence VisDrone-val
   (final benchmark) and 8-sequence calibration split.
3. Ran a full, predeclared, calibration-only development process for a
   detector-side AC-MOT controller: 5 independent automated search stages
   (resolution mapping, NMS mapping, cue-subset selection, joint
   threshold+action search) plus two oracle/headroom diagnostics and a
   segment-level diagnostic -- all converged on the same null result
   under a predeclared robustness gate.
4. Found and corrected a bug in an early verification script that had
   wrongly excluded confidence floor from the search space; the
   corrected, broadly-distributed finding (static confidence floor 0.40
   beats 0.01) became part of the frozen policy. A diagnosis-motivated
   adaptive-confidence follow-up was tested and also refuted.
5. Froze AC-MOT as a documented no-op (a constant, non-adaptive detector
   operating point) and ran the untouched final 7-sequence VisDrone-val
   exactly once for systems 3 and 4.
6. Paired bootstrap (5000 resamples) on the resulting deltas.

Every step, including every negative result, is committed to git with
its own provenance file under `research/acmot_paper_v2/`.

## 2. Frozen detector

`dronefreak/visdrone-yolo11m` (`best.pt`), Hugging Face, SHA256
`c6f98247e7c731a567aaf37ca7b808beff588b687623e028b25a2e1d846e96e7`.
VisDrone2019-**DET**-trained (not MOT), imgsz=640 native. **Not** the
official OATrack detector (not publicly released). See
`research/acmot_paper_v2/frozen_detector/DETECTOR_PROVENANCE.json`.

## 3. Calibration data

8 VisDrone2019-MOT-train sequences (`DETECTOR_SPLIT.json` ->
`calibration`), 3282 frames. Never the final 7-sequence VisDrone-val.

## 4. Frozen AC-MOT policy

**AC-MOT is frozen as a documented no-op**: a constant detector operating
point (resolution=1088, NMS IoU=0.45, confidence floor>=0.40), not a
scene-adaptive policy. See
`research/acmot_paper_v2/controller_search/FREEZE_MANIFEST.json` and
`DEVELOPMENT_DECISION_REPORT.md` for the complete record of every
formulation tested (5 search stages + 3 diagnostics, all null under the
predeclared robustness gate).

## 5. Final four-system table (VisDrone-val, 7 sequences, 71,830 GT boxes)

| System | Operating point | HOTA | MOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| 1. YOLO11m + ByteTrack | 1536/0.70, no conf floor | 41.59 | 9.88 | 48.57 | 1341 | 39274 | 24121 | 54.85 | 66.42 |
| 2. YOLO11m + OATrack (min_conf=0.40) | 1536/0.70, no conf floor | 46.77 | 28.17 | 57.11 | 773 | 25394 | 25432 | 64.63 | 64.59 |
| 3. AC-MOT (frozen no-op) + YOLO11m + ByteTrack | **1088/0.45, conf>=0.40** | 44.53 | 29.42 | 54.01 | 931 | 23354 | 26413 | 66.04 | 63.23 |
| 4. AC-MOT (frozen no-op) + YOLO11m + OATrack | **1088/0.45, conf>=0.40** | 48.27 | 34.77 | 59.96 | 499 | 20724 | 25629 | 69.03 | 64.32 |

**Important framing**: systems 1/2 use an independently predeclared
default operating point (1536/0.70, no confidence pre-filter --
documented before any detector selection or calibration work, in
`notebooks/YOLO11m_VisDrone_train_and_cache.ipynb`'s original design).
Systems 3/4 use the calibration-selected matched-static point. **Because
AC-MOT is frozen as a constant policy, systems 3/4 are, by construction,
identical to "YOLO11m at its calibration-best fixed operating point" +
{ByteTrack, OATrack}.** There is no online, per-frame, scene-conditioned
adaptation happening in systems 3/4.

## 6. Deltas (paired bootstrap, 5000 resamples, seed 42)

**System 3 - System 1 (ByteTrack):**

| metric | Δ | 95% CI |
|---|---|---|
| HOTA | +2.94 | [+2.03, +3.44] |
| MOTA | +19.54 | [+14.04, +25.82] |
| IDF1 | +5.45 | [+4.45, +6.29] |
| IDS | -410 | [-652, -178] |
| FP | -15920 | [-20889, -11212] |
| FN | +2292 | [+351, +4813] |
| Precision | +11.19 | [+10.06, +12.83] |
| Recall | -3.19 | [-6.20, -0.61] |

**System 4 - System 2 (OATrack):**

| metric | Δ | 95% CI |
|---|---|---|
| HOTA | +1.50 | [+0.37, +2.44] |
| MOTA | +6.61 | [+4.18, +8.68] |
| IDF1 | +2.86 | [+1.29, +3.97] |
| IDS | -274 | [-515, -65] |
| FP | -4670 | [-7140, -2805] |
| FN | +197 | [-1531, +2314] (not significant) |
| Precision | +4.41 | [+3.26, +6.22] |
| Recall | -0.27 | [-3.03, +2.28] (not significant) |

**These deltas are driven entirely by the better fixed operating point
found during calibration (lower resolution, tighter NMS, a confidence
pre-filter), not by any AC-MOT adaptation mechanism** -- AC-MOT was
frozen as a no-op before this comparison was run. The correct scientific
reading is: "for this detector, a calibration-selected static operating
point beats the originally-predeclared default operating point" -- a
useful, legitimate, statistically supported finding about detector
tuning, but **not** evidence for or against adaptive, scene-conditioned
AC-MOT control, which showed no benefit over the best static point in
every one of the 5 development-stage searches.

## 7. Failure analysis (per-sequence HOTA delta)

| sequence | ByteTrack (3-1) | OATrack (4-2) |
|---|---|---|
| uav0000086_00000_v | +3.41 | +2.74 |
| uav0000117_02622_v | +3.36 | +3.20 |
| uav0000137_00458_v | +2.59 | +0.29 |
| uav0000182_00000_v | +3.60 | +2.20 |
| uav0000268_05773_v | +0.06 | +0.60 |
| uav0000305_00000_v | +2.67 | +1.56 |
| uav0000339_00001_v | +3.66 | **-1.49** |

Broadly distributed (6/7 sequences improve for both comparisons), not
driven by a single outlier -- consistent with the operating-point finding
being a genuine, broadly-applicable improvement rather than an artifact.
One regression: `uav0000339_00001_v` under OATrack (-1.49 HOTA) --
OATrack's internal min_conf=0.40 combined with the external conf>=0.40
pre-filter may interact unfavorably on this specific sequence; not
investigated further (would require touching held-out val to diagnose,
which is out of scope post-freeze).

## 8. Statistically supported claims

- The calibration-selected static operating point (1088/0.45/conf>=0.40)
  significantly outperforms the originally-predeclared default
  (1536/0.70/no floor) on HOTA, MOTA, IDF1, IDS, FP, and Precision for
  both ByteTrack and OATrack hosts (95% CI excludes 0 for all of these).
- OATrack continues to show its established qualitative signature
  (fewer IDS, fewer FP, higher IDF1) relative to ByteTrack at the same
  operating point, consistent with `OATRACK_REPRO.md`'s COCO-detector
  findings and the paper's own reported pattern.
- No detector-side adaptive (scene-conditioned) AC-MOT policy -- across
  resolution, NMS, cue subset, SCI thresholds, and confidence floor,
  individually and jointly, over 5 independent automated search stages
  -- robustly beat the best fixed operating point under the predeclared
  CV robustness gate.

## 9. NOT supported

- **AC-MOT (the adaptive mechanism) does not have demonstrated benefit
  for this detector** -- the development process found no robust
  evidence of headroom, let alone of a working adaptive policy. This is
  a null result, not a disproof of AC-MOT in general (the repository
  independently records two other null results against matched-static
  references for different detector/tracker pairs -- `GAP_AUDIT.md`
  2.3).
- The FN/Recall increases in the system3-vs-1 delta are statistically
  significant in the worse direction -- the better operating point
  trades some recall for a much larger precision gain; this is not "pure
  improvement," and should not be reported as such.
- The system4-vs-2 FN/Recall changes are not statistically significant
  either way.
- Nothing here constitutes a comparison with the official OATrack paper
  or detector (different detector, different training data, different
  protocol throughout -- `DETECTOR_PROVENANCE.json`).
- No comparison with the historical/old AC-MOT formulation was performed
  (scoped out: different detector, different controller architecture --
  `DEVELOPMENT_DECISION_REPORT.md` Phase 6).

## 9b. Real end-to-end T4 runtime (200 frames, real detector forward pass, not cache playback)

| operating point | tracker | mean latency | P95 latency | FPS | detector time | tracker time | peak GPU reserved |
|---|---|---|---|---|---|---|---|
| systems 1/2 default (1536/0.70) | ByteTrack | 139.9 ms | 171.7 ms | 7.15 | 129.3 ms | 10.6 ms | 0.69 GB |
| systems 1/2 default (1536/0.70) | OATrack | 128.9 ms | 162.4 ms | 7.76 | 126.6 ms | 2.2 ms | 0.69 GB |
| systems 3/4 frozen AC-MOT (1088/0.45/conf>=0.40) | ByteTrack | 56.2 ms | 62.7 ms | 17.79 | 48.8 ms | 7.4 ms | 0.69 GB |
| systems 3/4 frozen AC-MOT (1088/0.45/conf>=0.40) | OATrack | 50.0 ms | 53.5 ms | 20.01 | 48.3 ms | 1.7 ms | 0.69 GB |

Detector time dominates end-to-end latency in every configuration.
Systems 3/4's operating point is also ~2.5x faster than systems 1/2's
(lower resolution dominates the speedup) -- a real-time benefit on top
of the quality benefit, though again attributable to the fixed
operating-point choice, not to any online AC-MOT adaptation (no
adaptation is active). Peak GPU memory is essentially flat across
configurations (single-image inference, T4 15.36GB headroom).

## 10. Remaining work before this is paper-ready

1. **UAVDT transfer** -- the required raw data exists
   (`UAV-benchmark-M` / `UAV-benchmark-MOTD_v1.0`) but sits inside a
   different, already-frozen experiment's protected directory
   (`UAVDT_EXTERNAL_GENERALIZATION`, dated 2026-09-12). The project's own
   protocol (`OATRACK_DESIGN.md`, `GAP_AUDIT.md` P4) requires the owner's
   explicit authorization before any new use of it. Not touched.
2. Old-AC-MOT apples-to-apples comparison (scoped out, see above) --
   would need a separate protocol decision.
3. **Genuinely adaptive AC-MOT development (owner research-direction
   correction, 2026-10-04)**: the owner has directed that the primary
   paper objective is a genuinely adaptive controller evaluated against
   the predeclared host baseline `r1536_n70` (not the static
   `r1088_n45_conf0.40` point, which is preserved as secondary control
   evidence only). A corrected predeclaration
   (`controller_search/PREDECLARATION_V2_ADAPTIVE.json`) is drafted:
   measurable "genuinely adaptive" definition (>=2 actions, each >=10%
   of frames, with within-sequence switching), objective redefined vs
   `r1536_n70`, and a same-distribution shuffled-control acceptance test.
   **Paused before execution**: a second `claude` process was found
   running against this same repository root; execution is held until
   the owner confirms whether that is a separate concurrent session
   (single-writer requirement).
