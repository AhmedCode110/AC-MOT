# Paper split — master audit (written before the manuscripts)

Date: 2026-09-28. Repository branch `universal-adapters-v1-y0zkeh`. This audit
classifies every experiment line found in the repository, its branches and the
owner's Google Drive research folders, and assigns it to Paper 1, Paper 2, or
neither. Only files listed here may supply numbers to the manuscripts; each
manuscript has its own `result_provenance.md` that maps every number to one of
these files.

Legend for "Class": P1 = Paper 1 evidence; P2-DEV = Paper 2 development
evidence; P2-EXT = Paper 2 post-freeze external evidence; P2-RAT = Paper 2 design
rationale (prior method, cited as prior work, never as a new result); STALE =
superseded by a later final result of the same experiment; UNUSED = outside both
stories; UNVERIFIED = no canonical file/commit, cannot be used.

## 1. Boundaries

| Paper | Scientific boundary | Frozen reference |
|---|---|---|
| Paper 1 — scene-adaptive AC-MOT | the legacy line: hand-crafted scene cues → Scene Complexity Index (SCI) → detector operating point (confidence, NMS IoU, input resolution) with YOLOv8n + ByteTrack; final state = the 2026-09-12 scientific freeze (validation-selected Trial 24 profile "V1", multi-objective Trial 22 profile "V2", VisDrone held-out test and zero-tuning UAVDT) plus the post-hoc matched-static test of 2026-09-18 | tag `v1.0.0-acmot-frozen` → a6c1fa4; branch `freeze/final-after-uavdt-2026-09-12` → b591122; freeze record `research/paper_split/evidence/legacy/ACMOT_FINAL_SCIENTIFIC_FREEZE_2026-09-12.md` |
| Paper 2 — universal causal self-calibrating AC-MOT | V7f, frozen at `universal-acmot-v7-freeze` → 488df9a; V6-TF only as the prior design whose failure motivates V7 | tag `universal-acmot-v7-freeze`; lock `research/V7_POLICY_LOCK.json` |

The universal intermediate versions V1/V3/V4/V5-TF (tags `universal-acmot-v1/v3/v4-freeze`)
belong to neither story: they are neither the scene-adaptive method of Paper 1
(V3/V4 replace the SCI by calibration/compute-budget rules) nor the method of
Paper 2. Name clash warning: the legacy freeze calls its two frozen profiles
"V1" and "V2"; they are unrelated to the universal tags `universal-acmot-v1-freeze`.
Paper 1 therefore names them the quality profile (Q, Trial 24) and the balanced
profile (B, Trial 22).

## 2. Experiment classification

### 2.1 Legacy scene-adaptive line (Paper 1)
| ID | Experiment | Data | Class | Authoritative file(s) | Notes |
|---|---|---|---|---|---|
| L-OP | operating-point sweeps: resolution 512–960 (step 32), confidence 0.05–0.50, NMS IoU 0.30–0.80; YOLOv8n + tuned ByteTrack; T4 FPS | VisDrone2019-MOT-val (7 seq) | P1 | `evidence/legacy/OPERATING_{RESOLUTION,CONFIDENCE,NMS}_SWEEP.csv` (Drive ids 1liYJ…, 1b4-c…, 1iOyf…), `OPERATING_ABLATION_REPORT.json` (Drive 13QtyF…) | validation only |
| L-TEMP | temporal ablation: smoothing window {1,3,5,7,9} × analysis stride {1,5,10,15,20} with the heuristic controller | VisDrone val | P1 | `evidence/legacy/TEMPORAL_ABLATION_FULL.csv` (Drive 1h5E_x…) | selected W=7, S=10 |
| L-OLDABL | heuristic controller component ablation (A0 default, A1 tuned tracker, A2 adaptive conf/NMS, A2R adaptive resolution, A3 full) | VisDrone val | P1 | `evidence/legacy/OLD_ACMOT_COMPONENT_ABLATION.csv` (git: paper-release-2026-09-11) | the Colab cell that wrote it was not recovered (freeze record §22) — outputs preserved |
| L-OPT | joint TPE search (50 trials) of SCI weights, conf/NMS endpoints and SCI thresholds; FPS gate; selection = highest MOTA with IDS ≤ heuristic A3 | VisDrone val | P1 | `evidence/legacy/FROZEN_DEFENSIBLE_ACMOT_CONFIG.json`, `EMPIRICAL_PARAMETER_IMPORTANCE_MOTA.json` (Drive 1mbZq8…) | Trial 24 selected |
| L-NEWABL | component ablation of the optimized controller: A0 static anchor (960/0.35/0.35), A1 + adaptive conf, A2 + adaptive NMS, A3 full | VisDrone val | P1 | `evidence/legacy/NEW_ACMOT_COMPONENT_ABLATION.csv` (+ Drive report 1e8t4n…) | A0 ≈ A3 on validation (see §4) |
| L-TEST | locked three-system held-out test: static default (conf 0.25, NMS 0.45, 640, default ByteTrack), heuristic AC-MOT, optimized AC-MOT (Trial 24); T4 processing FPS | VisDrone2019-MOT-test-dev (17 seq, 6,635 frames) | P1 main | `evidence/legacy/FINAL_TEST_*` (git: paper-release-2026-09-11), `V1_PER_SEQUENCE_METRICS.csv` (Drive 1kbOQ…), `V1_PAIRED_BOOTSTRAP_95CI.csv` (Drive 12p6s…; 5,000 resamples, seed 42) | custom class-agnostic protocol, not the official VisDrone evaluation |
| L-MATCH | post-hoc matched static anchor (960/0.35/0.35, tuned ByteTrack, no adaptation) on test-dev, paired with Trial 24 | VisDrone test-dev | P1 (attribution) | `evidence/legacy/MATCHED_STATIC_A0_TESTDEV.json` (Drive 1S-9U…, 1J1LT…) | run 2026-09-18 after the freeze; FPS from a separate T4 session |
| L-V2 | multi-objective TPE (maximize MOTA, minimize IDS; 49 complete trials); predeclared balanced selection Trial 22; test-dev and UAVDT | VisDrone val / test-dev / UAVDT | P1 | freeze record §10–14, 19; `UAVDT_*` files | test-dev attempt 1 interrupted before any result persisted; exact technical rerun (freeze record §13) |
| L-UAVDT | zero-tuning external test: static default, Trial 24, Trial 22; YOLOv8n classes car/bus/truck | UAVDT test (20 seq, 16,592 frames) | P1 | `evidence/legacy/UAVDT_FINAL_COMPARISON.json` (Drive 1gH7S…), `UAVDT_PER_SEQUENCE.csv` (Drive 1WY5o…, 1kiF3…, 1GZ65…); bootstrap in freeze record §19 (5,000 resamples, seed 42) | adapter self-test passed |
| L-RT | realtime engineering v12–v15: strict throughput gate, FP16, per-stage profiler, decoded-frame processing FPS | VisDrone (T4) | P1 (runtime context) | `versions/v12..v15/README.md` | decoded-frame processing (v15) and JPEG-decode-inclusive (v12–v14) definitions differ; only reported with their own definitions |
| L-SEM | 4-way ablation 12 sequences (seminar, June 2026) | VisDrone subset | STALE | `docs/ablation_4way_legacy_12seq.csv` | superseded by L-OLDABL/L-TEST; not used |
| L-V16/17 | paper-eval harness v16/v17 candidates | — | STALE | `versions/v16,v17/README.md` | v17 mapping = the heuristic controller used as "Old AC-MOT" in L-TEST |
| L-EXPL | early detector/tracker comparisons, noise/blur/compression/frame-drop robustness | VisDrone | UNVERIFIED | freeze record §5, §22 ("not every early exploratory run has one canonical frozen path/commit") | no noise/compression/frame-skipping result is used in Paper 1 |

### 2.2 Universal line before V7 (neither paper, except V6 as rationale)
| ID | Experiment | Class | Source |
|---|---|---|---|
| U1–U4 | universal V1/V3/V4 selections, V4 held-out test-dev (E31) | UNUSED | `research/RESULTS_TESTDEV_V4.md`, context files |
| U5 | V5 learned controller / V5-TF E36–E41 | UNUSED (superseded; E41 rejected by audit) | `research/context/EXPERIMENT_REGISTRY.md` |
| U6-D | V6-TF development (val-7, dev-40) | UNUSED | `research/final/EXPERIMENT_LEDGER.md` |
| U6-F | V6-TF post-freeze: VisDrone confirmation-16, test-dev, Faster R-CNN, BoT-SORT, UAVDT | P2-RAT (prior design; aggregate facts only) | `research/final/FINAL_RESULTS.md` |
| U6-X | V6-TF on SparseTrack / BoostTrack (MOT17 val-half), 10,000-resample bootstrap | P2-RAT (the transfer failure that motivates V7) | `research/final/EXTERNAL_PAPER_TRANSFER.md` §12–14 |

### 2.3 V7 line (Paper 2)
| ID | Experiment | Class | Authoritative file(s) |
|---|---|---|---|
| D1–D8 | diagnostics (duplicate pairs, regime statistic ρ, band precision proxies) | P2-DEV (design rationale) | `research/final/V7_EXPERIMENT_LEDGER.md` |
| E0–E20 | V7a–V7f mechanism loop (VisDrone val-7/dev-40 label-free and Mac labelled; MOT17 hosts; KITTI) | P2-DEV | ledger, `V7_DEV_RESULTS.json`, `V7_ABLATION.md` |
| DEV-MOT17 | ByteTrack (official, ultralytics), OC-SORT, BoostTrack, SparseTrack hosts; floors 0.01/0.1 | P2-DEV | `V7_STATISTICS.md`, `V7_DEV_RESULTS.json`, `V7_MAIN_RESULTS.md`, `sparsetrack_v7f/` |
| DEV-KITTI | ByteTrack / BoT-SORT / OC-SORT × YOLOv8n / RT-DETR-L on KITTI tracking training (21 seq) | P2-DEV (detector transfer) | `V7_STATISTICS.md` |
| STRESS-L | score-calibration shift (pow3, scale05, temp2, temp05) and emission-floor shift | P2-DEV (robustness) | `V7_DEV_RESULTS.json`, `V7_MAIN_RESULTS.md` Table 3 |
| TESTS | causality, reset, pass-through, name-independence, affine invariance, lock integrity (61 tests) | P2 (method verification) | `tests/test_v7_adaptive_layer.py` |
| RT | controller runtime 2.95 ms mean / 4.39 ms P95; end-to-end 57.85 → 60.44 ms | P2 | `V7_REALTIME.md`, `V7_REALTIME_yolov8n.json` |
| X-PD | PD-SORT (IEEE TCE 2025), predeclared | P2-EXT | `V7_EXTERNAL_TRANSFER.md`, `V7_EXTERNAL_RESULTS.json` |
| X-HS | Hybrid-SORT (AAAI 2024), predeclared | P2-EXT | same |
| X-CT | C-TWiX (Pattern Recognition 2025): MOT17 val-half, KITTIMOT val, DanceTrack val | P2-EXT | `V7_RECENT_EXTERNAL_RESULTS.json`, `research/final/recent/ctwix/` |
| X-TT | TOPICTrack (IEEE TIP 2025), MOT17 val-half | P2-EXT — RUNNING | `research/final/recent/topictrack/` when committed |
| X-TK | TrackTrack (CVPR 2025), DanceTrack val | P2-EXT — RUNNING | `research/final/recent/tracktrack/` when committed |
| X-AER | V7f on VisDrone val-7 (development split, never labelled for V7f), confirmation-16 (reserved for one post-freeze V7 check), test-dev (post-hoc) | P2-EXT (aerial) — RUNNING | `research/final/aerial_v7f/` when committed |
| X-SP | SparseTrack + V7f | P2-DEV (development host measured after the freeze) | `research/final/sparsetrack_v7f/` |

## 3. Conflicts and resolutions (provenance audit)
| Item | Older value | Final value | Resolution |
|---|---|---|---|
| C-TWiX MOT17 baseline HOTA | 77.535 (CI run 36482783364) | 77.550 (CI run 36484421601) | Both runs are recorded; the committed `recent/ctwix/table_MOT17.json` is from the later run and is the one used. The 0.015 difference is CPU float16 autocast on two CPU models; each Δ is paired within one run. |
| Legacy 12-sequence ablation vs final held-out | seminar table | L-TEST | seminar table not used |
| Legacy bootstrap resamples | 5,000 (freeze record) | — | kept as recorded; Paper 1 states 5,000. Paper 2 comparisons use 10,000. |
| "Old AC-MOT" in legacy files vs heuristic A3 | same controller (core_v17 PresentationController, W=7, S=10) | — | named "heuristic scene-adaptive controller" in Paper 1 |
| V4 held-out test-dev numbers in `RESULTS_CANONICAL.md` | V4 universal line | — | not part of either paper |

## 4. Findings that constrain the claims
1. Paper 1: on validation, the optimized controller (A3) equals its own static
   anchor (A0: 960 px, conf 0.35, NMS 0.35) within 0.04 HOTA; on the held-out
   test, the matched static anchor gives HOTA 33.93 vs 33.84 for the adaptive
   profile (Δ −0.09 [−0.25, +0.06], 5,000 resamples). The held-out gains over
   the static default (+5.41 HOTA) and over the heuristic controller (+1.14)
   are therefore attributable to the validation-calibrated operating point and
   tracker profile, not to frame-to-frame switching. Paper 1 must say this.
2. Paper 1: the heuristic controller's adaptive resolution does add accuracy
   over its fixed-640 counterpart on validation (A2R/A3 vs A1: +1.58/+2.20
   HOTA) at an FPS cost; this is the evidence for scene-driven resolution
   selection when the operating range is wide.
3. Paper 2: V7f preserves strong two-stage hosts (ByteTrack, BoostTrack,
   SparseTrack, Hybrid-SORT; C-TWiX MOT17), improves single-stage hosts
   (OC-SORT, PD-SORT) with CIs above zero, and has one significant negative
   external cell (C-TWiX KITTIMOT car, −1.52 HOTA) caused by false noisy-regime
   calls in a short sequence. No universal-improvement claim is allowed.
4. Neither paper has a flight test, embedded-board measurement or GPU
   measurement of V7f. Paper 1 timing is Tesla T4; Paper 2 timing is a 4-vCPU
   Xeon.

## 5. Shared material and roles
| Asset | Paper 1 role | Paper 2 role |
|---|---|---|
| VisDrone2019-MOT test-dev | main held-out test of the scene-adaptive profiles | post-hoc aerial check of V7f (if the running job completes); not a main external claim |
| VisDrone val | selection/validation for the SCI controller | development split of V7 (label-free and labelled) |
| YOLOv8n | the only detector | one of three detector families |
| ByteTrack | the only tracker | one of ten host trackers |
| UAVDT | zero-tuning cross-dataset test | not used (no V7f result) |
| TrackEval 12c8791 | evaluator | evaluator |
