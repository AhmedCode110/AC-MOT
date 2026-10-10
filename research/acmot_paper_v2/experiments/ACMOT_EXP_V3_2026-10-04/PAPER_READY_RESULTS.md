# ACMOT_EXP_V3_2026-10-04 -- paper-ready results

Generated 2026-10-04T21:18:35+00:00 from existing result files only. Every cell names its source.

## Table 1. VisDrone2019-MOT-val (7 sequences, 71,830 GT boxes). Systems 3/4 = SECOND VAL READ

| System | HOTA | MOTA | IDF1 | IDS | FP | FN | Precision | Recall | source |
|---|---|---|---|---|---|---|---|---|---|
| 1. YOLO11m + ByteTrack, r1536_n70 | 41.59 | 9.88 | 48.57 | 1341 | 39274 | 24121 | 54.85 | 66.42 | `research/acmot_paper_v2/BASELINE_SYSTEMS_RESULT.json → YOLO11m+ByteTrack.aggregate` |
| 2. YOLO11m + OATrack, r1536_n70 | 46.77 | 28.17 | 57.11 | 773 | 25394 | 25432 | 64.63 | 64.59 | `research/acmot_paper_v2/BASELINE_SYSTEMS_RESULT.json → YOLO11m+OATrack.aggregate` |
| 3. AC-MOT v3 + YOLO11m + ByteTrack | 44.52 | 27.87 | 53.93 | 1016 | 24892 | 25902 | 64.85 | 63.94 | `research/acmot_paper_v2/SECOND_VAL_READ_V3_RESULT.json → AC-MOT-v3+YOLO11m+ByteTrack.aggregate` |
| 4. AC-MOT v3 + YOLO11m + OATrack | 48.10 | 34.48 | 59.36 | 604 | 21400 | 25057 | 68.61 | 65.12 | `research/acmot_paper_v2/SECOND_VAL_READ_V3_RESULT.json → AC-MOT-v3+YOLO11m+OATrack.aggregate` |

## Table 2a. System 3 − System 1 (ByteTrack): paired bootstrap (5000 resamples, seed 42), SECOND VAL READ

| metric | Δ | 95% CI | source |
|---|---|---|---|
| HOTA | +2.94 | [+2.36, +3.37] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → bytetrack_v3_vs_system1.HOTA` |
| MOTA | +18.00 | [+13.54, +23.28] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → bytetrack_v3_vs_system1.MOTA` |
| IDF1 | +5.36 | [+4.49, +6.18] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → bytetrack_v3_vs_system1.IDF1` |
| IDS | -325.00 | [-498.00, -164.00] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → bytetrack_v3_vs_system1.IDS` |
| FP | -14382.00 | [-19176.00, -9620.00] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → bytetrack_v3_vs_system1.FP` |
| FN | +1781.00 | [+288.00, +3594.07] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → bytetrack_v3_vs_system1.FN` |
| Precision | +10.00 | [+8.71, +11.89] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → bytetrack_v3_vs_system1.Precision` |
| Recall | -2.48 | [-4.51, -0.47] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → bytetrack_v3_vs_system1.Recall` |

## Table 2b. System 4 − System 2 (OATrack): paired bootstrap (5000 resamples, seed 42), SECOND VAL READ

| metric | Δ | 95% CI | source |
|---|---|---|---|
| HOTA | +1.33 | [+0.20, +2.26] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → oatrack_v3_vs_system2.HOTA` |
| MOTA | +6.32 | [+4.74, +7.88] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → oatrack_v3_vs_system2.MOTA` |
| IDF1 | +2.26 | [+0.55, +3.47] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → oatrack_v3_vs_system2.IDF1` |
| IDS | -169.00 | [-292.00, -42.00] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → oatrack_v3_vs_system2.IDS` |
| FP | -3994.00 | [-6190.00, -2494.90] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → oatrack_v3_vs_system2.FP` |
| FN | -375.00 | [-1462.57, +670.00] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → oatrack_v3_vs_system2.FN` |
| Precision | +3.98 | [+2.96, +5.51] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → oatrack_v3_vs_system2.Precision` |
| Recall | +0.52 | [-0.87, +2.11] | `research/acmot_paper_v2/BOOTSTRAP_V3_RESULT.json → oatrack_v3_vs_system2.Recall` |

## Table 3. Controller behaviour on VisDrone-val (SECOND VAL READ)

| host | LOW | MEDIUM | HIGH | source |
|---|---|---|---|---|
| AC-MOT-v3+YOLO11m+ByteTrack | 0.074 | 0.301 | 0.625 | `research/acmot_paper_v2/SECOND_VAL_READ_V3_RESULT.json → AC-MOT-v3+YOLO11m+ByteTrack.level_frac` |
| AC-MOT-v3+YOLO11m+OATrack | 0.074 | 0.333 | 0.593 | `research/acmot_paper_v2/SECOND_VAL_READ_V3_RESULT.json → AC-MOT-v3+YOLO11m+OATrack.level_frac` |

## Table 4. Calibration (development) evidence

| item | value | source |
|---|---|---|
| CV-mean ΔHOTA vs r1536_n70 (trial 7) | +2.447 | `research/acmot_paper_v2/controller_search/FREEZE_MANIFEST_V3_ADAPTIVE.json → development_evidence` |
| per-fold ΔHOTA | +0.59, +4.16, +2.03, +3.00 | same |
| shuffled control CV-mean (seed 44) | +2.617 (causal +2.447; causal_beats_shuffled=False) | `research/acmot_paper_v2/controller_search/SHUFFLED_CONTROL_RESULT.json` |

## Table 5. End-to-end runtime, Tesla T4 (200 frames)

| system | host | FPS | mean latency ms | detector ms | tracker ms | controller ms | source |
|---|---|---|---|---|---|---|---|
| baseline r1536_n70 | bytetrack | 7.15 | 139.9 | 129.3 | 10.6 | – | `research/acmot_paper_v2/RUNTIME_BENCHMARK_RESULT.json` |
| baseline r1536_n70 | oatrack | 7.76 | 128.9 | 126.6 | 2.2 | – | `research/acmot_paper_v2/RUNTIME_BENCHMARK_RESULT.json` |
| AC-MOT v3 | bytetrack | 13.45 | 74.3 | 54.6 | 8.0 | 4.3 | `research/acmot_paper_v2/V3_RUNTIME_BENCHMARK_RESULT.json` |
| AC-MOT v3 | oatrack | 15.27 | 65.5 | 52.6 | 1.8 | 4.2 | `research/acmot_paper_v2/V3_RUNTIME_BENCHMARK_RESULT.json` |

## Table S1 (supplementary control). Static r1088_n45_conf0.40, FIRST VAL READ (commit bfa5caf)

| system | HOTA | MOTA | IDF1 | IDS | ΔHOTA vs r1536_n70 [95% CI] | source |
|---|---|---|---|---|---|---|
| static + ByteTrack | 44.53 | 29.42 | 54.01 | 931 | +2.94 [+2.03, +3.44] | `research/acmot_paper_v2/FINAL_ACMOT_SYSTEMS_RESULT.json`, `research/acmot_paper_v2/BOOTSTRAP_RESULT.json` |
| static + OATrack | 48.27 | 34.77 | 59.96 | 499 | +1.50 [+0.37, +2.44] | `research/acmot_paper_v2/FINAL_ACMOT_SYSTEMS_RESULT.json`, `research/acmot_paper_v2/BOOTSTRAP_RESULT.json` |

## Table 6. UAVDT transfer (frozen policy, no retuning; single run, not a search -- no CV/bootstrap applicable)

20 sequences, 16592 frames, 49776 detector forward passes (= 20 x 16592 x 3 action pairs, exact). Scored with the pre-existing UAVDT adapter (`tools/build_uavdt_view.py`) and its class-agnostic scorer (`tools/seqstats.py`); single placeholder vehicle class, pedestrian dropped (not present in UAVDT).

| System | HOTA | MOTA | IDF1 | IDS | FP | FN | Precision | Recall | source |
|---|---|---|---|---|---|---|---|---|---|
| baseline r1536_n70 + ByteTrack | 45.92 | -1.57 | 58.40 | 601 | 279616 | 66050 | 49.57 | 80.63 | `research/acmot_paper_v2/UAVDT_TRANSFER_RESULT.json → baseline_r1536_n70+bytetrack.aggregate` |
| **v3 adaptive + ByteTrack** | 47.00 | 16.03 | 61.77 | 486 | 212077 | 73702 | 55.75 | 78.38 | `research/acmot_paper_v2/UAVDT_TRANSFER_RESULT.json → v3_adaptive+bytetrack.aggregate` |
| baseline r1536_n70 + OATrack | 47.73 | 19.61 | 62.10 | 517 | 197937 | 75612 | 57.27 | 77.82 | `research/acmot_paper_v2/UAVDT_TRANSFER_RESULT.json → baseline_r1536_n70+oatrack.aggregate` |
| **v3 adaptive + OATrack** | 47.64 | 22.87 | 62.93 | 486 | 183669 | 78774 | 58.80 | 76.89 | `research/acmot_paper_v2/UAVDT_TRANSFER_RESULT.json → v3_adaptive+oatrack.aggregate` |

### Table 6 deltas (v3 adaptive − r1536_n70 baseline)

| host | HOTA | MOTA | IDF1 | IDS | FP | FN | Precision | Recall | source |
|---|---|---|---|---|---|---|---|---|---|
| bytetrack | +1.08 | +17.60 | +3.37 | -115 | -67539 | +7652 | +6.18 | -2.24 | `research/acmot_paper_v2/UAVDT_TRANSFER_RESULT.json → v3_adaptive+bytetrack.aggregate − baseline_r1536_n70+bytetrack.aggregate` |
| oatrack | -0.09 | +3.27 | +0.83 | -31 | -14268 | +3162 | +1.53 | -0.93 | `research/acmot_paper_v2/UAVDT_TRANSFER_RESULT.json → v3_adaptive+oatrack.aggregate − baseline_r1536_n70+oatrack.aggregate` |

## Disclosures that must accompany these numbers

1. VisDrone-val was read twice (static control first, commit bfa5caf; v3 second, commit 0af0066). Table 1 rows 3/4 are a SECOND VAL READ.
2. Shuffled control: random timing of the same action mix scores at least as well as the causal schedule on calibration; the gain is not shown to require scene-conditioned timing.
3. Selection-rule deviation: trial 7 is not the argmax of the predeclared selection rule (see FREEZE_MANIFEST.json → selection_audit).
4. Detector: third-party dronefreak/visdrone-yolo11m, trained on VisDrone2019-DET at 640 px, not VisDrone-MOT and not the OATrack detector.
5. Bootstrap used 5000 resamples (tools/v7/bootstrap.py default is 10000); values archived as computed.
6. Per-sequence ΔHOTA in FINAL_REPORT.md §7 is derived; the script that produced that table is not archived as a separate file (NEEDS VERIFICATION).
7. UAVDT transfer (Table 6) is a single frozen-policy run per host, not a search; no statistical test was applied. The ByteTrack improvement is large and consistent with the VisDrone direction; the OATrack improvement is small and mixed (HOTA essentially flat, -0.09).
