# Result provenance — Paper 2 (frozen V7f)

Every number in `manuscript.tex` maps to one committed file below. Paths are relative
to the repository root. Frozen policy: tag `universal-acmot-v7-freeze` → commit 488df9a
(`configs/universal_acmot_policy_v7.json`, lock `research/V7_POLICY_LOCK.json`).
Tables and figures are generated only by `scripts/make_tables.py`, `scripts/make_figures.py`
and `scripts/make_bib.py`, which read `scripts/data.py`; the values that exist only in
committed Markdown result files are transcribed in `scripts/data.py` with their source row.

Build: `python scripts/make_tables.py && python scripts/make_figures.py && python scripts/make_bib.py && tectonic manuscript.tex`
(run from `JAIS_PAPERS/PAPER2_UNIVERSAL_ACMOT_JAIS/`; Fig. 4 needs the development cache
`outputs/det_cache_val_native`, path passed with `--cache`).

## 1. Source files

| ID | File | Content | Produced by |
|---|---|---|---|
| S-DEV | `research/final/V7_DEV_RESULTS.json` | pooled metrics of every development cell (MOT17 hosts, BoostTrack, KITTI, stress transforms) | cloud C1 runs, committed before and at the freeze |
| S-STAT | `research/final/V7_STATISTICS.md` | bootstrap CIs of the MOT17 and KITTI development cells | `tools/v7/mot17_eval_v7.py --boot`, `tools/v7/kitti/kitti_eval.py --boot` |
| S-MAIN | `research/final/V7_MAIN_RESULTS.json` / `.md` | paper references, SparseTrack record, per-sequence MOT17 cells | commit 692cb72 / 23e12ef |
| S-SP | `research/final/sparsetrack_v7f/results.json`, `summary.txt`, `lock_check.txt` | SparseTrack + V7f (CI run 36475110213) | GitHub Actions |
| S-EXT | `research/final/V7_EXTERNAL_RESULTS.json`, `V7_EXTERNAL_TRANSFER.md` | PD-SORT, Hybrid-SORT (predeclared) with bootstrap | cloud C1, post-freeze |
| S-SEL | `research/final/V7_EXTERNAL_SELECTION.md` | predeclaration of PD-SORT, Hybrid-SORT (in the freeze commit 488df9a) | freeze commit |
| S-REC | `research/final/V7_RECENT_EXTERNAL_RESULTS.json` | C-TWiX (3 datasets), TrackTrack, TOPICTrack: pooled, per sequence, bootstrap, W/T/L | GitHub Actions runs 36484421601, 36506599893, 36483670872 |
| S-PROT | `research/final/V7_RECENT_EXTERNAL_PROTOCOL.md` | preregistered protocol, classes, contracts, predictions (commit 990a157, before any run) | — |
| S-SYS | `research/final/V7_RECENT_EXTERNAL_SYSTEMS.md`, `V7_RECENT_EXTERNAL_FAILURES.md` | reference numbers, reproduction classes, TOPICTrack failure, unaudited candidates | — |
| S-AUD | `research/final/recent/ctwix/audit_{MOT17,KITTIMOT,DanceTrack}.json`, `recent/tracktrack/audit_DanceTrack.json` | per-frame layer log of the frozen runs (regime, t1, t2, rho, rho_bar, n_in, n_pass, n_rescued) | same runs |
| S-CAL | `research/final/paper2_calib_boot/calib_boot.json`, `summary.txt`, `lock_check.txt`, `environment.txt`, `tracks.tar.gz` | 14 calibration-shift conditions: re-run of both arms from the frozen code, replay identity vs S-DEV, bootstrap, W/T/L, verdict | GitHub Actions run 36587453420 (script `tools/v7/ci/calib_boot.sh`, rule declared in commit 938aaff before the run) |
| S-RT | `research/final/V7_REALTIME_yolov8n.json`, `V7_REALTIME.md` | runtime benchmark | `tools/v7/benchmark.py`, post-freeze |
| S-V6 | `research/final/EXTERNAL_PAPER_TRANSFER.md` §9, §12, §14, §15 | V6-TF on SparseTrack/BoostTrack; fates of true detections; t2 ≈ 0.78 | V6 external transfer (commit 8272014) |
| S-LED | `research/final/V7_EXPERIMENT_LEDGER.md` (E15–E20, STRESS-L), `V7_ABLATION.md`, `V7_FAILURE_EVOLUTION.md` | design history, rejected variants, D1 pair counts | — |
| S-REG | `research/context/EXPERIMENT_REGISTRY.md` (E36, E41, E45), `FAILED_EXPERIMENTS.md` (FX-18) | V5-TF / V6-TF VisDrone development counts | — |
| S-AUDIT | `JAIS_PAPERS/PAPER_SPLIT_MASTER_AUDIT.md` §3 | C-TWiX baseline 77.535 vs 77.550 across two runs | — |
| S-TEST | `tests/test_v7_adaptive_layer.py`, `tests/test_v7_bootstrap.py` (frozen suite), `tests/test_v7f_paper2_claims.py` (post-freeze, 25 tests) | verification statements | local run on the frozen code (59 passed, 2 skipped; 25 passed) |
| S-CODE | `acmot_v7.py`, `online_calibration.py`, `configs/universal_acmot_policy_v7.json`, adapters in `tools/v7/external/`, `tools/v7/recent/` | method description, host contracts | freeze 488df9a |

## 2. Abstract and Introduction

| Number in text | Value | Source |
|---|---|---|
| halved scores → HOTA 0 (ByteTrack official, OC-SORT) | 0.00, 0.00 | S-DEV `BY_official_st_NATIVE_t_scale05`, `OC_st_NATIVE_t_scale05`; S-CAL |
| earlier layer lowered HOTA by 4.2 / 5.9 | −4.15, −5.89 | S-V6 §12, §14 |
| PD-SORT +0.61 [0.27, 1.65] | 0.613 [0.2677, 1.6458] | S-EXT `bootstrap_10k_seed42` |
| 7 of 14 calibration-shift conditions | summary.significant_recovery = 7 | S-CAL |
| C-TWiX KITTI cars −1.52 | −1.5210 [−4.053, −0.134] | S-REC ctwix/KITTIMOT/car |
| controller 2.95 ms | 2.952 | S-RT arms.v7.ctrl.mean_ms |
| nine trackers, four detector sources, three datasets | SparseTrack, BoostTrack, ByteTrack, OC-SORT, BoT-SORT, PD-SORT, Hybrid-SORT, C-TWiX, TrackTrack; YOLOX-X, YOLOv8n, RT-DETR-L, PermaTrack; MOT17, KITTI, DanceTrack | Table 3 of the manuscript (S-MAIN, S-STAT, S-EXT, S-REC) |
| cubing: 67.70→60.92, 66.43→58.34 | 67.698/60.921, 66.428/58.344 | S-DEV `BY_official_st_NATIVE`, `..._t_pow3`, `OC_st_NATIVE`, `OC_st_NATIVE_t_pow3` |
| ultralytics loses 2.0 HOTA under temperature 2 | 66.000 → 63.974 | S-DEV `BY_ultra_st_NATIVE`, `BY_ultra_st_NATIVE_t_temp2` |
| Figure 1 | all bars | S-DEV (keys in `make_figures.py:fig1`) |

## 3. Method (Section IV)

| Statement | Source |
|---|---|
| Eq. (2)–(5), thresholds per regime, remap 0.5+0.5u / 0.1+0.4u, thresholds 0.5 | S-CODE `acmot_v7.py` `_bands`, `step`; `online_calibration.py` `exact_otsu2`, `nested_otsu` |
| W = 10, cumulative median (rho_frames 0), bg_check, clean = upper, noisy t2/otsu, scores auto, cold host, cold_dup clean, rescue track/fg, motion noisy, hist 100, warmup 5 | `configs/universal_acmot_policy_v7.json` |
| rank reference: frame 1 and every 10th, last 20 | `acmot_v7.py` lines 473–475, `ecdf_samples` maxlen 20 |
| motion: phase correlation of 1/4-resolution gray frames / diagonal; cap 0.95 | `acmot_v7.py` `motion_cue`, `step` |
| 61 frozen tests, 59 passed / 2 skipped; 25 new tests passed | S-TEST (run on the worktree at the freeze files; lock verified by the tests) |
| Figure 3 (replay, 272 clean YOLOv8n frames, 0 RT-DETR-L; file fig03_partition.pdf) | `make_figures.py:fig4` on `outputs/det_cache_val_native/{yolov8,rtdetr}/uav0000268_05773_v.npz` (release asset `acmot_detcache_val_native.tar`); assertion that the plotted t1, t2 equal the layer's log |

## 4. Design rationale (Section V)

| Number | Value | Source |
|---|---|---|
| 17 cells MOTA < 0 (histogram three-class Otsu, F3) | 17 of 80 | S-REG E36 |
| 15 cells (exact three-class Otsu, E41), FP ≈ 4× reference | 15; "FP ≈ 4× V4" | S-REG E41; FAILED_EXPERIMENTS FX-18 |
| 19% duplicate boxes | 19% | S-REG E42; FX-18 |
| V6-TF candidate: 2 cells (reference 3) | 2 | S-REG E45 |
| V6-TF on SparseTrack −4.15 [−5.54, −1.66]; BoostTrack −5.89 [−7.76, −2.87]; +GBI −5.37 [−7.30, −2.75] | — | S-V6 §14 (`data.py` V6_TRANSFER) |
| 49,709 true detections; 5,787 demoted; 3,527 deleted; 1,403 below background | — | S-V6 §15 (`data.py` V6_FATES) |
| 2,166 distinct vs 1,775 duplicate pairs | — | `V7_FAILURE_EVOLUTION.md` item 1 (D1) |
| t2 ≈ 0.78 vs 0.6; precision ≈ 95%, ≈ 35 candidates/frame | — | S-V6 §15 |
| cold = none: −0.3 to −0.5 HOTA; sub-low rescue FP +700–1000; ECDF in clean frames −0.2/−1.6; host-relative band rejected (26.9 vs 14.9 boxes/frame) | — | S-LED E15, E17, E20; `V7_ABLATION.md` |
| Table 9 (ablation) | HOTA of `BY_official_bt_*`, `OC_bt_*`, `BT7C_*_pf` | S-DEV |

## 5. Results (Section VII)

| Number | Source |
|---|---|
| Table 4 (development MOT17): all cells | S-MAIN (SparseTrack bootstrap, per-sequence), S-STAT (CIs), S-DEV (pooled, DetA/AssA/FP/FN) |
| five of nine MOT17 cells identical | S-STAT ("identical output" rows) and S-DEV (equal pooled values; BoostTrack GBI keys `BT7C_*_pf_post_gbi`) |
| SparseTrack +0.055 [−0.011, +0.254], W/T/L 2/4/1 | S-SP, S-MAIN per_sequence (tie 0.01) |
| ByteTrack official floor 0.01: −0.014 [−0.058, +0.009] | S-STAT |
| OC-SORT +0.613 [+0.363, +1.241] / +0.464 [+0.266, +1.086]; MOTA +1.267 / +1.262; 7/7 | S-STAT; S-MAIN per_sequence |
| OC-SORT FP +895 / FN −1,570 (floor 0.01) | S-DEV `OC_st_BASELINE`, `OC_st_V7f` |
| PD-SORT: MOTA +1.128 [+0.273, +3.032]; IDF1 +0.817 [+0.430, +1.938]; DetA +0.780; AssA +0.465; FN −1,291 [−1,918, −730]; FP +681 [+307, +1,108]; IDS +2 [−9, +13]; 7/7; MOT17-11 MOTA −1.17 | S-EXT |
| Hybrid-SORT identical | S-EXT |
| TrackTrack −0.013 [−0.032, −0.001]; DetA −0.05, MOTA −0.07; 0/21/4; 25,453 of 25,508 clean | S-REC tracktrack/DanceTrack_post; S-SYS; S-AUD |
| C-TWiX MOT17 −0.055 [−0.456, +0.398]; AssA −0.374 [−1.082, −0.019]; FN −551; FP +477; 1,148 rescued; 2,651 of 2,659 clean | S-REC; S-AUD audit_MOT17.json |
| C-TWiX KITTI ped −0.103 [−2.163, +0.034] | S-REC |
| C-TWiX DanceTrack −1.005 [−2.622, +0.646]; 10/3/12 | S-REC |
| predictions (TOPICTrack pass-through; C-TWiX ≥ 0 on MOT17; no collapse \|Δ\| < 1) | S-PROT §8, scored against S-REC |
| Table 6 KITTI: all cells | S-STAT (`data.py` STAT_KITTI) |
| ByteTrack YOLOv8n ≈ 1,500 more missed cars | S-DEV kitti_train_yolov8 `NATIVE` FN_car 7,899 → `V7f` 9,401; S-LED ("FN +1500 car") |
| OC-SORT YOLOv8n IDS +79 [43, 125] | S-STAT; FN car 14,760 → 11,076 (S-DEV) |
| Table 7 calibration shift (14 rows), 7 recoveries, 0 degradations, 2 identical, 5 no change | S-CAL (`calib_boot.json`; every re-run identical to S-DEV pooled values: `all_replays_identical = true`) |
| unstressed 67.70 / 68.49; ultralytics FP 10,516 vs 5,722 | S-DEV `BY_official_st_NATIVE`, `BT7C_NATIVE_pf`, `BY_ultra_st_NATIVE_t_temp2`, `BY_ultra_st_NATIVE` |
| IDF1 −0.016, AssA −0.020 (ByteTrack temp. 0.5); OC-SORT temp. 0.5 DetA +0.68 [+0.07, +1.67], FP +889, FN −1,411 | S-CAL |
| Table 8 runtime; 57.85 / 60.44 ms; +2.59 ms (+4.5%); 17.29 / 16.55 FPS; 2,279 frames | S-RT |
| practical scale: 77.535 vs 77.550 | S-AUDIT §3 |

## 6. Failure analysis (Section VIII)

| Number | Source |
|---|---|
| KITTI car baseline 89.27 vs 89.3 (EXACT) | S-REC reproduction field |
| ΔDetA −1.75, ΔAssA −1.29, ΔMOTA −2.11, FN +161 [14, 375], FP 0 | S-REC ctwix/KITTIMOT/car delta |
| per sequence: 0014 −17.09 (car) −8.50 (ped); 0018 −2.92; 0008 −1.60; others \|Δ\| ≤ 0.35 | S-REC per_seq_dHOTA |
| 4.6 candidates/frame (all), 5.8 in 0014; 84 of 2,981 noisy; 0014: 10 switches, 45 noisy, 120 of 280 passed, ρ̄ ∈ [0.471, 0.542] from frame 22; 0008 noisy frames 3–14; 0018 54–80 | S-AUD audit_KITTIMOT.json (computed in this session; the computation is reproduced by `make_figures.py:fig9` (Fig. 10) for the timeline) |
| DanceTrack: 0 noisy of 25,508; 2,037 rescued; FN −1,691; FP +1,257; AssA −1.51; IDF1 −1.55; IDS +148 [−6, +323]; per-sequence −12.29 to +9.37 | S-AUD audit_DanceTrack.json; S-REC |
| KITTI YOLOv8n ρ ≈ 0.45 | `V7_FAILURE_EVOLUTION.md` item 6 |
| TOPICTrack 67.54 vs 69.6; FP −1,187, FN +1,660; exploratory +0.006 [−0.016, +0.025] | S-SYS failures table; S-REC topictrack |

## 7. Correction relative to earlier summaries

`V7_MAIN_RESULTS.md` Table 3 marks eight calibration-shift cells in bold and several summaries
spoke of recovery in "8 of 14" conditions. That bold marking was not based on intervals. The
paired bootstrap computed after the freeze (S-CAL) gives **7 of 14** significant recoveries: OC-SORT
under temperature 0.5 (+0.45 [−0.01, +0.97]) does not meet the predeclared rule. The manuscript
uses 7 of 14.
