# AC-MOT Final Scientific Freeze — 2026-09-12

> **STATUS: FINAL SCIENTIFIC FREEZE — DO NOT MODIFY**
>
> This file is the canonical scientific/provenance record for the AC-MOT state reached on 2026-09-12 after the frozen VisDrone and UAVDT evaluations. Any later tuning, candidate change, recalibration, benchmark, tracker change, detector-control change, or new result must be treated as a new version/experiment and must not overwrite this freeze.

---

## 0. Freeze identity

- Project: **AC-MOT — Adaptive Control for Real-Time Multi-Object Tracking**
- Repository: `AhmedCode110/AC-MOT`
- Freeze date: `2026-09-12`
- Timezone: `Africa/Cairo`

### Final Drive freeze

```text
/content/drive/MyDrive/AC-MOT-shared/[FROZEN][DO_NOT_MODIFY]_ACMOT_2026-09-12_FINAL_AFTER_UAVDT
```

Drive folder ID:

```text
1KtFtutk1ZDrLQzbdXXFo-hZhPfkrwMrV
```

### GitHub final freeze branch

```text
freeze/final-after-uavdt-2026-09-12
```

Created from:

```text
1882ece5dc0662f889c75f5ca48ac5d427489f42
```

---

## 1. Scientific lock rules

The following are frozen for all results in this document:

- V1 candidate = **Trial 24**.
- V2 candidate = **Trial 22**.
- No re-selection using VisDrone test-dev.
- No re-selection using UAVDT test data.
- No new Optuna/tuning/recalibration may be used to silently replace these final candidates.
- No post-test changes to SCI weights, confidence mapping, NMS mapping, resolution levels, temporal settings, tracker settings, evaluation filtering, UAVDT adapter logic, or candidate selection.
- Any later experiment must be explicitly named as a **new version** or **future work**.

---

## 2. Canonical Drive roots

Project root:

```text
/content/drive/MyDrive/AC-MOT-shared
```

Drive folder ID:

```text
1Ry6tnO69Fpx9FFn2nKTPkRxy2n8o9Ah1
```

Dataset root:

```text
/content/drive/MyDrive/AC-MOT-shared/AC-MOT-data
```

Validation:

```text
/content/drive/MyDrive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-val
```

Held-out historical final split:

```text
/content/drive/MyDrive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-test-dev
```

The VisDrone protocol used by the project is the custom class-agnostic AC-MOT TrackEval protocol and is **not** the official VisDrone leaderboard protocol.

---

## 3. Earlier read-only snapshot

The earlier snapshot remains immutable:

```text
/content/drive/MyDrive/AC-MOT-shared/[FROZEN][DO_NOT_MODIFY]_ACMOT_2026-09-12_CURRENT_STATE
```

Folder ID:

```text
1DiOeISUQe0KgTt0_CTuXf7xgLaZ9kX6P
```

Main structure:

```text
00_READ_ONLY_MANIFEST/
01_V1_FROZEN_SCIENTIFIC/
02_V2_PRE_RUN_CODE_STATE/
03_DATA_AND_FINAL_EVIDENCE_REFERENCES/
```

That snapshot represents the pre-V2-scientific-run state and must not be overwritten with later V2/UAVDT evidence.

---

## 4. GitHub provenance

### 4.1 V1 immutable state

```text
Repository: AhmedCode110/AC-MOT
Branch:     acmot-final-frozen-2026-09-11
Commit:     a6c1fa49fce1d402513c2df05b7d04b962a6e89e
Tree:       88ecb7519dc5241e09cec7edfb71ccfbb11739da
Tag:        v1.0.0-acmot-frozen
```

### 4.2 V2 pre-run state

```text
Branch: experiment/multiobjective-mota-ids-v2
Commit: 2b400347584512ebc09527a3e0e01dad82299329
Tree:   6c120b9a57f939e8a75a327c232733049b86612a
```

Key files:

```text
scripts/optuna_sci_v2_multiobjective_validation.py
scripts/run_v2_colab_shared.py
config/V2_COLAB_SHARED_PATHS.json
docs/V2_DEVELOPMENT_STATUS.md
```

### 4.3 Frozen V2 evaluation state

```text
5becc52a569f271ee8b73ce47c67d495de0d64a5
```

At this state the re-run-safe wrapper exists:

```text
scripts/run_v2_colab_shared_safe.py
```

### 4.4 TrackEval pin

```text
12c8791b303e0a0b50f753af204249e622d0281a
```

### 4.5 V2 Trial22 technical rerun

Runner commit:

```text
e4104ee3f2ee066a6d4e3a827e0ac9c902de528d
```

Runner:

```text
scripts/run_v2_trial22_testdev_technical_rerun.py
```

Launcher commit:

```text
57c6ef9bc98f50ae2ef73d22a233dc2b6ac3d433
```

Launcher:

```text
scripts/run_v2_trial22_testdev_colab_launcher.py
```

The launcher pins:

```text
AC-MOT:   5becc52a569f271ee8b73ce47c67d495de0d64a5
TrackEval: 12c8791b303e0a0b50f753af204249e622d0281a
Runner:    e4104ee3f2ee066a6d4e3a827e0ac9c902de528d
```

### 4.6 UAVDT frozen runner

```text
scripts/run_uavdt_external_generalization_frozen.py
commit: 17e9319036d403559ca005289750b7a62432fa1f
```

Launcher:

```text
scripts/run_uavdt_external_colab_launcher.py
commit: 1882ece5dc0662f889c75f5ca48ac5d427489f42
```

Pinned runtime:

```text
ultralytics==8.3.200
numpy==2.2.6
scipy==1.15.3
lap
opencv-python-headless
pandas
matplotlib
GPU = Tesla T4
```

---

## 5. Historical exploratory phase

Before the final defensible chain, the project included exploratory work on detector/model comparisons, YOLO timing and YOLOv8n selection, ByteTrack/BoT-SORT/OCSort/IOU/DeepSORT/StrongSORT, speed/frame-drop and noise/blur robustness, initial SCI design, adaptive confidence/NMS/resolution, and early AC-MOT ablations.

Recovered historical/library notebook names include:

```text
AC_MOT_v12_Colab.ipynb
AC_MOT_FULL17_Recorded_Replay_20260905.ipynb
AC_MOT_Optuna_Val_Test_Colab.ipynb
AC_MOT_Optuna_Val_Test_OfficialVisDrone.ipynb
AC_MOT_Optuna_Full_VisDrone_Official5.ipynb
AC_MOT_Optuna_SCI_Only_NoTraining.ipynb
AC_MOT_Optuna_SCI_Only_RUN_ALL.ipynb
AC_MOT_Optuna_SCI_Only_RUN_ALL_v2.ipynb
```

These are development artifacts unless explicitly tied to a canonical path/commit below.

---

## 6. V1 canonical scientific root and outputs

```text
/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers
```

Folder ID:

```text
1RRjzBy1Q_K4415A1x4FCD7B4xb60QYBs
```

Preserved key chain:

```text
SCIENTIFIC_SEARCH_SPACE.json
FROZEN_TEMPORAL_CONFIG.json
DETECTOR_DERIVED_CUE_CALIBRATION.json
OLD_A3_VALIDATION_W7_S10.json
EMPIRICAL_OPTUNA.db
EMPIRICAL_OPTUNA_TRIALS.csv
EMPIRICAL_STUDY_SIGNATURE.json
EMPIRICAL_PARAMETER_IMPORTANCE_MOTA.json
FROZEN_DEFENSIBLE_ACMOT_CONFIG.json
OPERATING_RESOLUTION_SWEEP.csv
OPERATING_CONFIDENCE_SWEEP.csv
OPERATING_NMS_SWEEP.csv
OPERATING_ABLATION_REPORT.json
TEMPORAL_ABLATION_FULL.csv
TEMPORAL_ABLATION_RANKED.csv
OLD_ACMOT_COMPONENT_ABLATION.csv
OLD_ACMOT_COMPONENT_ABLATION_REPORT.json
OLD_ACMOT_COMPONENT_ABLATION_DONE.json
NEW_ACMOT_COMPONENT_ABLATION.csv
NEW_ACMOT_COMPONENT_ABLATION_REPORT.json
NEW_ACMOT_COMPONENT_ABLATION_DONE.json
FINAL_TEST_3WORKER_PROTOCOL.json
FINAL_TEST_WORKER_1_BASELINE_DEFAULT.json
FINAL_TEST_WORKER_2_OLD_ACMOT_FROZEN.json
FINAL_TEST_WORKER_3_NEW_ACMOT_FROZEN.json
FINAL_TEST_COMPARISON_3WORKER.csv
FINAL_TEST_RESULTS_3WORKER.json
FINAL_TEST_DONE.json
final_test_3workers/
final_test_resume/
```

---

## 7. V1 method and Trial24 validation freeze

Method: `AC-MOT Defensible Empirical Calibration + Joint SCI Optimization`

```text
detector = fixed pretrained YOLOv8n
tracker = fixed tuned ByteTrack
smoothing_window = 7
analysis_stride = 10
resolution_levels = [512, 928, 960]
supported_confidence = [0.25,0.30,0.35,0.40,0.45]
supported_nms = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
tiny proxy = box area < 32x32
```

V1 search used validation only. Its maximum budget was 50 and its selection rule used the FPS gate and IDS<=Old-A3 before selecting highest MOTA with tie-breaking.

Old A3 reference:

```text
MOTA .18164674437597447
HOTA .33064013219122707
IDF1 .36296242459481065
IDS 271
FN 46609
FP 8232
FPS 42.08545786066066
```

Selected V1 candidate:

```text
Trial24
MOTA .23038087460093548 = 23.0381%
HOTA .3611015551055777  = 36.1102%
IDF1 .4075780119438273  = 40.7578%
IDS 270
FPS 37.16857745688161
```

Frozen V1 parameters:

```text
weight_crowd .12949277455301997
weight_tiny  .22174766876599927
weight_edge  .43371337893805056
weight_night .05355765312756694
weight_blur  .16148852461536325
conf_easy .30
conf_hard .40
nms_easy .35
nms_hard .35
threshold_mid .13534938199219218
threshold_high .28728676236279177
```

---

## 8. V1 VisDrone held-out final test

| System | MOTA | HOTA | IDF1 | IDS | FN | FP | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline_Default | 19.729% | 28.430% | 32.724% | 1235 | 155051 | 16308 | 36.528 |
| Old_ACMOT_Frozen | 23.236% | 32.698% | 39.516% | **1061** | 138967 | 25025 | 41.957 |
| New_ACMOT_Frozen / V1 Trial24 | **26.948%** | **33.835%** | **41.546%** | 1184 | **136858** | 19029 | 38.985 |

New vs Baseline: +7.2195 pp MOTA, +5.4051 pp HOTA, +8.8220 pp IDF1, 51 fewer IDS, 18193 fewer FN, 2721 more FP, +2.4562 FPS.

New vs Old: +3.7123 pp MOTA, +1.1366 pp HOTA, +2.0295 pp IDF1, **123 more IDS**, 2109 fewer FN, 5996 fewer FP, -2.9720 FPS.

Do not claim V1 had the lowest IDS in this comparison.

---

## 9. V1 post-hoc per-sequence/bootstrap evidence

Root:

```text
/content/drive/MyDrive/AC-MOT-shared/V1_POSTHOC_ANALYSIS_2026-09-12
```

Files:

```text
reconstructed_tracker_files/
V1_PER_SEQUENCE_EVAL_INPUT/
V1_PER_SEQUENCE_TRACKEVAL/
V1_PER_SEQUENCE_METRICS.csv
V1_PAIRED_BOOTSTRAP_SAMPLES.csv
V1_PAIRED_BOOTSTRAP_95CI.csv
RECONSTRUCTED_TRACKER_MANIFEST.csv
```

5000 paired resamples, seed 42.

New vs Baseline:

```text
MOTA +7.2195298911 pp CI [5.4361737286, 9.2933769298]
HOTA +5.4050796745 pp CI [4.1272586724, 6.9022341261]
IDF1 +8.8219725444 pp CI [6.9005329576, 11.0537470575]
IDS reduction +51 CI [-114, 219]
```

New vs Old:

```text
MOTA +3.7123164073 pp CI [2.3916325222, 5.1900750895]
HOTA +1.1366403975 pp CI [0.1378418489, 2.2671786440]
IDF1 +2.0294763365 pp CI [0.5870919005, 3.7717311228]
IDS reduction -123 CI [-244, -14]
```

---

## 10. V2 multi-objective optimization

Root:

```text
/content/drive/MyDrive/AC-MOT-shared/V2_MULTI_OBJECTIVE_MOTA_IDS
```

Folder ID: `15C4AxE6qX0QY-oXElKCqIxTnDN_m7j5E`

Outputs:

```text
V2_MULTI_OBJECTIVE_OPTUNA.db
V2_ALL_TRIALS.csv
V2_STUDY_SIGNATURE.json
V2_PROTOCOL.json
V2_SEARCH_SPACE.json
V2_INPUT_HASHES.json
logs/
plots/
```

Scientific formulation:

```text
maximize MOTA
minimize IDS
FPS >= 25 only
NO IDS <= 271 feasibility gate
TPESampler seed 42
validation-only optimization
```

Failed technical setup folders preserved separately:

```text
[FAILED][NUMPY_STALE_KERNEL]_V2_MULTI_OBJECTIVE_MOTA_IDS_2026-09-12
[FAILED][RECURSION_GUARD]_V2_MULTI_OBJECTIVE_MOTA_IDS_2026-09-12
```

These are not scientific candidate results.

---

## 11. V2 completed study and predeclared selection

Planned 50 trials; completed exactly Trial0..Trial48 = **49 COMPLETE**. Do not fabricate a missing trial.

Notable Pareto points:

```text
T16 MOTA .2283317 IDS308
T17 .2280793 IDS294
T20 .2275002 IDS289
T5 .2268320 IDS284
T4 .2260450 IDS262
T26 .2209667 IDS233
T42 .2066078 IDS219
T34 .1984706 IDS214
T22 .1933031 IDS168
T10 .1865914 IDS160
T3 .1748756 IDS148
T43 .1575470 IDS142
T32 .1571906 IDS136
T45 .1511619 IDS135
T28 .1507759 IDS117
T44 .1178707 IDS115
T8 .1153612 IDS114
```

Highest MOTA = T16; lowest IDS = T8.

Predeclared balanced score:

```text
MOTA_norm=(MOTA-min)/(max-min)
IDS_good=(maxIDS-IDS)/(maxIDS-minIDS)
score=.5*MOTA_norm+.5*IDS_good
```

Tie preference: higher HOTA, IDF1, FPS, then lower trial number.

Official selected V2 candidate: **Trial22**. Post-hoc switching is forbidden.

---

## 12. V2 Trial22 validation freeze

```text
MOTA .1933031405449551
HOTA .31651410689705645
IDF1 .3422606845656919
IDS 168
DetA .20064208220784727
AssA .5089701353603627
FN 49301
FP 4858
FPS 51.406277608143924
mean_imgsz 824.9051300070274
mean_conf .4
mean_nms_iou .4967211911207374
```

Frozen V2 parameters:

```text
weight_crowd .16464526567145857
weight_tiny .17462652795045444
weight_edge .5076112530333374
weight_night .12069564693725379
weight_blur .03242130640749567
conf_easy .4
conf_hard .4
nms_easy .6
nms_hard .35
threshold_mid .2927135841069045
threshold_high .6661671600900015
resolutions [512,928,960]
W=7
stride=10
```

---

## 13. V2 post-selection freeze and held-out audit

Root:

```text
/content/drive/MyDrive/AC-MOT-shared/V2_POST_SELECTION_TEST_2026-09-12
```

Folder ID: `1SKSnx3vYtvSC9WtKklA_1qKxYJB_P-JY`

Files:

```text
V2_TRIAL22_FINAL_FREEZE.json
V2_POST_SELECTION_TEST_PROTOCOL.json
V2_TESTDEV_ACCESS_STARTED_ATTEMPT1_INTERRUPTED.json
V2_TRIAL22_TECHNICAL_RERUN_AUDIT.json
V2_TESTDEV_ACCESS_STARTED.json
V2_TRIAL22_TESTDEV/
V2_TRIAL22_TESTDEV_DONE.json
```

Attempt1 DID access test-dev, was interrupted, and did not persist a final result. No parameters, candidate, or tuning changed before the exact technical rerun.

Final folder:

```text
/content/drive/MyDrive/AC-MOT-shared/V2_POST_SELECTION_TEST_2026-09-12/V2_TRIAL22_TESTDEV
```

Folder ID: `1LP4Zetoc5UYAkfDcS74h7k1xNG-a0oZd`

Contents:

```text
V2_TRIAL22_TESTDEV_RESULT.json
configuration.json
dataset_manifest.json
V2_Trial22_Frozen/
trackeval/
```

Final result:

```text
MOTA .23791939129545053 = 23.792%
HOTA .3121843360559406 = 31.218%
IDF1 .37870176284936796 = 37.870%
DetA .2274803684086514
AssA .4367715368425862
IDS 919
FN 149307
FP 13632
FPS 46.02399278609119
mean_imgsz 868.7987942727958
mean_conf .4
mean_nms_iou .47165425466904604
```

---

## 14. Frozen VisDrone comparison and statistics

| System | MOTA | HOTA | IDF1 | IDS | FN | FP | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 19.729% | 28.430% | 32.724% | 1235 | 155051 | 16308 | 36.528 |
| V1 Trial24 | **26.948%** | **33.835%** | **41.546%** | 1184 | **136858** | 19029 | 38.985 |
| V2 Trial22 | 23.792% | 31.218% | 37.870% | **919** | 149307 | **13632** | **46.024** |

V2 vs Baseline: +4.063 pp MOTA, +2.788 pp HOTA, +5.146 pp IDF1, 316 fewer IDS, 5744 fewer FN, 2676 fewer FP, +9.4957 FPS.

V2 vs V1: -3.1565 pp MOTA, -2.6171 pp HOTA, -3.6759 pp IDF1, **265 fewer IDS**, +12449 FN, -5397 FP, +7.0395 FPS.

5000 paired bootstrap, seed 42:

```text
V2 vs Baseline
MOTA +4.063 pp CI [0.952,6.886]
HOTA +2.788 pp CI [0.747,4.660]
IDF1 +5.146 pp CI [2.028,8.080]
IDS reduction +316 CI [175,470]

V2 vs V1
MOTA -3.157 pp CI [-6.145,-1.346]
HOTA -2.617 pp CI [-4.004,-1.694]
IDF1 -3.676 pp CI [-5.882,-2.241]
IDS reduction +265 CI [82,502]
```

V1 is the quality leader; V2 has a statistically supported IDS advantage over V1 on VisDrone.

---

## 15. UAVDT external generalization

Root:

```text
/content/drive/MyDrive/AC-MOT-shared/UAVDT_EXTERNAL_GENERALIZATION
```

Folder ID: `1z1q42Rsexuk7RDnhv4vp2Gb7aNnkFlVG`

Top-level evidence:

```text
UAVDT_EXTERNAL_PROTOCOL_FREEZE.json
UAVDT_FREEZE_RECOVERY_AUDIT.json
UAVDT_ADAPTER_V1_FREEZE.json
UAVDT_ADAPTER_V1_M0101_SELFTEST.json
UAVDT_EXTERNAL_TEST_SYSTEMS_LOCK.json
UAVDT_EXTERNAL_TEST_ACCESS_STARTED.json
data/
results/
```

Purpose: **zero-tuning external benchmark**.

Frozen systems: `Baseline_Frozen`, `V1_Trial24_Frozen`, `V2_Trial22_Frozen`.

No UAVDT-specific training, Optuna, recalibration, threshold tuning, tracker tuning, or candidate selection was allowed.

---

## 16. UAVDT split and adapter

Frozen test split: 20 sequences / 16592 frames.

```text
M0203 M0205 M0208 M0209 M0403
M0601 M0602 M0606 M0701 M0801
M0802 M1001 M1004 M1007 M1009
M1101 M1301 M1302 M1303 M1401
```

Frame counts:

```text
M0203 1007; M0205 646; M0208 265; M0209 1576; M0403 514;
M0601 372; M0602 480; M0606 1374; M0701 1308; M0801 298;
M0802 1101; M1001 1859; M1004 269; M1007 659; M1009 604;
M1101 864; M1301 1182; M1302 719; M1303 445; M1401 1050.
```

Adapter SHA256:

```text
8957b4da276121408b5769c23eb3de84dd30667c88d539e437ba6b6afc28648a
```

Detector classes: `[2,5,7]` = car, bus, truck.

GT preprocessing: remove mark/score=0, remove negative frames, normalize IDs.

Prediction preprocessing: remove negative frames, remove rows after max GT frame, reject duplicate frame-ID.

IoU threshold: `0.5`.

Ignore rule: remove prediction only when strictly fully contained inside ignore region.

Ignore-pass sequences:

```text
M0203 M0205 M0208 M0403 M0601 M0602 M0606 M0701 M0802
M1001 M1004 M1007 M1009 M1101 M1301 M1302 M1303 M1401
```

M0209 and M0801 are test sequences without that ignore pass.

Adapter self-test M0101:

```text
GT 5414 raw/kept
MOTA/HOTA/DetA/AssA/IDF1 = 1.0
IDS/FN/FP = 0
SELF-TEST PASSED
TEST DATA USED = NO
TUNING = NO
```

---

## 17. UAVDT system lock and resume semantics

Baseline:

```text
YOLOv8n
classes [2,5,7]
conf .25
NMS .45
imgsz 640
default tracker
```

V1 = Trial24 frozen V1 controller/calibration/tracker.

V2 = Trial22 frozen V2 controller/calibration/tracker.

Lock rules:

```text
TUNING NO
RECALIBRATION NO
PARAMETER CHANGES NO
CANDIDATE RESELECTION NO
SELECTION ON TEST NO
```

Checkpointing is **per system**, not per sequence. Completed systems with DONE markers are skipped on rerun. A system interrupted before its DONE marker restarts from that system's beginning. Drive-to-local copy time is excluded from processing FPS.

---

## 18. UAVDT final output tree

Root:

```text
/content/drive/MyDrive/AC-MOT-shared/UAVDT_EXTERNAL_GENERALIZATION/results/FROZEN_3SYSTEM_TEST_2026-09-12
```

Folder ID: `1YKnh6ZRWh4KT01Lv1BQXQUYxOcrOxgh8`

Top-level outputs:

```text
UAVDT_TEST_DATA_PROVENANCE.json
Baseline_Frozen/
Baseline_Frozen_DONE.json
V1_Trial24_Frozen/
V1_Trial24_Frozen_DONE.json
V2_Trial22_Frozen/
V2_Trial22_Frozen_DONE.json
UAVDT_FINAL_COMPARISON.csv
UAVDT_FINAL_COMPARISON.json
UAVDT_EXTERNAL_TEST_DONE.json
```

Each system folder contains:

```text
RESULT.json
timing.json
configuration.json
dataset_manifest.json
uavdt_eval/
<system evaluation/prediction folder>/
```

Each `uavdt_eval/` contains:

```text
aggregate.json
metrics.json
per_sequence.csv
preprocess.json
```

---

## 19. UAVDT final metrics and statistics

| System | MOTA | HOTA | IDF1 | IDS | FN | FP | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 13.841% | 24.085% | 27.887% | 558 | 270189 | 22974 | **65.015** |
| V1 Trial24 | **17.399%** | **28.390%** | **34.453%** | 321 | **258376** | 22896 | 58.353 |
| V2 Trial22 | 16.118% | 26.930% | 32.014% | **308** | 266894 | **18758** | 61.462 |

Full aggregates:

```text
Baseline: HOTA .24084979675926227, DetA .14902361041769424, AssA .39587150106935903,
MOTA .1384105882560002, IDF1 .2788744515033468, IDS558, FN270189, FP22974,
FPS65.01495040047143, mean_imgsz640, mean_conf.25, mean_nms.45.

V1: HOTA .2839004895844121, DetA .176476992384633, AssA .46068317938474,
MOTA .17398637747648912, IDF1 .3445282883593379, IDS321, FN258376, FP22896,
FPS58.35347969089768, mean_imgsz954.9161041465767, mean_conf.3511353771705505, mean_nms.35.

V2: HOTA .2692986798081065, DetA .16062668604088398, AssA .454901818538945,
MOTA .1611763946659783, IDF1 .32014222599359893, IDS308, FN266894, FP18758,
FPS61.461843816976305, mean_imgsz828.8717454194792, mean_conf.4, mean_nms.4704812861612311.
```

5000 paired per-sequence bootstrap, seed 42:

```text
V1 vs Baseline
MOTA +3.558 pp CI [1.452,5.807]
HOTA +4.305 pp CI [2.740,5.709]
IDF1 +6.565 pp CI [3.808,8.822]
IDS reduction +237 CI [104,377]

V2 vs Baseline
MOTA +2.277 pp CI [0.720,4.164]
HOTA +2.845 pp CI [1.238,4.348]
IDF1 +4.127 pp CI [1.630,6.485]
IDS reduction +250 CI [110,404]

V2 vs V1
MOTA -1.281 pp CI [-2.441,-0.202]
HOTA -1.460 pp CI [-2.190,-0.908]
IDF1 -2.439 pp CI [-3.737,-1.366]
IDS reduction +13 CI [-20,50]
```

V2 has numerically fewer IDS than V1 on UAVDT (308 vs 321), but the 13-switch difference is not significant.

FP results:

```text
V2 vs Baseline: 4216 fewer FP CI [1669,6994]
V2 vs V1: 4138 fewer FP CI [2335,6295]
```

---

## 20. Final cross-dataset frozen table

| Dataset | System | MOTA | HOTA | IDF1 | IDS | FPS |
|---|---|---:|---:|---:|---:|---:|
| VisDrone | Baseline | 19.729% | 28.430% | 32.724% | 1235 | 36.528 |
| VisDrone | V1 Trial24 | **26.948%** | **33.835%** | **41.546%** | 1184 | 38.985 |
| VisDrone | V2 Trial22 | 23.792% | 31.218% | 37.870% | **919** | **46.024** |
| UAVDT | Baseline | 13.841% | 24.085% | 27.887% | 558 | **65.015** |
| UAVDT | V1 Trial24 | **17.399%** | **28.390%** | **34.453%** | 321 | 58.353 |
| UAVDT | V2 Trial22 | 16.118% | 26.930% | 32.014% | **308** | 61.462 |

---

## 21. Final scientific conclusion

**V1 Trial24 = quality-oriented frozen leader.** It has the highest frozen MOTA/HOTA/IDF1 across both VisDrone and UAVDT among the three compared systems.

**V2 Trial22 = identity / false-positive / speed trade-off.** On VisDrone it has significantly fewer IDS than V1, fewer FP, and higher FPS but lower MOTA/HOTA/IDF1. On UAVDT it has numerically fewer IDS than V1 but not significantly; it has substantially fewer FP and higher FPS than V1 while V1 remains significantly stronger in MOTA/HOTA/IDF1.

Frozen high-level claim:

> **V1 is the quality leader. V2 demonstrates a reproducible identity/false-positive/speed trade-off, and both V1 and V2 generalize above Baseline without UAVDT-specific tuning.**

Do not claim V2 globally dominates V1.

---

## 22. Known provenance limitations

1. The exact historical Colab wrapper/cell that wrote `OLD_ACMOT_COMPONENT_ABLATION.*` was not recovered; underlying evaluation chain and outputs are preserved.
2. The exact saved V1 resume-finalization cell/source was not pinned; sequence checkpoints, receipts, worker JSONs, and merged results are preserved.
3. Not every early exploratory detector/tracker/noise/frame-drop run has one canonical frozen path/commit. Treat early work as development history unless separately recovered and verified.

---

## 23. Major run → output map

### V1 operating/calibration
`OPERATING_RESOLUTION_SWEEP.csv`, `OPERATING_CONFIDENCE_SWEEP.csv`, `OPERATING_NMS_SWEEP.csv`, `OPERATING_ABLATION_REPORT.json`, `DETECTOR_DERIVED_CUE_CALIBRATION.json`, `SCIENTIFIC_SEARCH_SPACE.json`.

### V1 temporal
`TEMPORAL_ABLATION_FULL.csv`, `TEMPORAL_ABLATION_RANKED.csv`, `FROZEN_TEMPORAL_CONFIG.json`, `OLD_A3_VALIDATION_W7_S10.json`.

### V1 joint SCI optimization
`EMPIRICAL_OPTUNA.db`, `EMPIRICAL_OPTUNA_TRIALS.csv`, `EMPIRICAL_STUDY_SIGNATURE.json`, `EMPIRICAL_PARAMETER_IMPORTANCE_MOTA.json`, `FROZEN_DEFENSIBLE_ACMOT_CONFIG.json`.

### V1 component ablations
`OLD_ACMOT_COMPONENT_ABLATION.csv`, `OLD_ACMOT_COMPONENT_ABLATION_REPORT.json`, `OLD_ACMOT_COMPONENT_ABLATION_DONE.json`, `NEW_ACMOT_COMPONENT_ABLATION.csv`, `NEW_ACMOT_COMPONENT_ABLATION_REPORT.json`, `NEW_ACMOT_COMPONENT_ABLATION_DONE.json`.

### V1 held-out final
`FINAL_TEST_3WORKER_PROTOCOL.json`, worker 1/2/3 JSONs, `FINAL_TEST_COMPARISON_3WORKER.csv`, `FINAL_TEST_RESULTS_3WORKER.json`, `FINAL_TEST_DONE.json`, `final_test_3workers/`, `final_test_resume/`.

### V1 post-hoc
`V1_PER_SEQUENCE_METRICS.csv`, `V1_PAIRED_BOOTSTRAP_SAMPLES.csv`, `V1_PAIRED_BOOTSTRAP_95CI.csv`, `RECONSTRUCTED_TRACKER_MANIFEST.csv`, `V1_PER_SEQUENCE_EVAL_INPUT/`, `V1_PER_SEQUENCE_TRACKEVAL/`, `reconstructed_tracker_files/`.

### V2 study
`V2_MULTI_OBJECTIVE_OPTUNA.db`, `V2_ALL_TRIALS.csv`, `V2_STUDY_SIGNATURE.json`, `V2_PROTOCOL.json`, `V2_SEARCH_SPACE.json`, `V2_INPUT_HASHES.json`, `logs/`, `plots/`.

### V2 post-selection
`V2_TRIAL22_FINAL_FREEZE.json`, `V2_POST_SELECTION_TEST_PROTOCOL.json`, `V2_TESTDEV_ACCESS_STARTED_ATTEMPT1_INTERRUPTED.json`, `V2_TRIAL22_TECHNICAL_RERUN_AUDIT.json`, `V2_TESTDEV_ACCESS_STARTED.json`, `V2_TRIAL22_TESTDEV/`, `V2_TRIAL22_TESTDEV_DONE.json`.

### UAVDT protocol/adapter
`UAVDT_EXTERNAL_PROTOCOL_FREEZE.json`, `UAVDT_FREEZE_RECOVERY_AUDIT.json`, `UAVDT_ADAPTER_V1_FREEZE.json`, `UAVDT_ADAPTER_V1_M0101_SELFTEST.json`, `UAVDT_EXTERNAL_TEST_SYSTEMS_LOCK.json`.

### UAVDT final
`UAVDT_EXTERNAL_TEST_ACCESS_STARTED.json`, `UAVDT_TEST_DATA_PROVENANCE.json`, three system folders and DONE markers, `UAVDT_FINAL_COMPARISON.csv`, `UAVDT_FINAL_COMPARISON.json`, `UAVDT_EXTERNAL_TEST_DONE.json`.

---

## 24. Final Drive map

```text
AC-MOT-shared/
├── [FROZEN][DO_NOT_MODIFY]_ACMOT_2026-09-12_CURRENT_STATE/
├── defensible_acmot_3workers/
├── V1_POSTHOC_ANALYSIS_2026-09-12/
├── V2_MULTI_OBJECTIVE_MOTA_IDS/
├── [FAILED][NUMPY_STALE_KERNEL]_V2_MULTI_OBJECTIVE_MOTA_IDS_2026-09-12/
├── [FAILED][RECURSION_GUARD]_V2_MULTI_OBJECTIVE_MOTA_IDS_2026-09-12/
├── V2_POST_SELECTION_TEST_2026-09-12/
├── UAVDT_EXTERNAL_GENERALIZATION/
└── [FROZEN][DO_NOT_MODIFY]_ACMOT_2026-09-12_FINAL_AFTER_UAVDT/
    └── ACMOT_FINAL_SCIENTIFIC_FREEZE_2026-09-12.md
```

---

## 25. Final rule

Do not overwrite this freeze. Any later work must use a new dated/versioned experiment and new freeze, with this state preserved unchanged.

Frozen scientific story:

```text
Exploratory development
→ V1 operating/calibration studies
→ V1 temporal/component ablations
→ V1 Trial24 validation freeze
→ V1 held-out VisDrone test
→ V1 per-sequence/bootstrap evidence
→ V2 multi-objective MOTA/IDS study (49 complete trials)
→ predeclared balanced Pareto Trial22 selection
→ V2 post-selection freeze
→ interrupted test attempt preserved
→ exact frozen technical rerun
→ V2 final VisDrone result
→ UAVDT adapter/protocol/system freeze
→ zero-tuning external generalization
→ final statistical interpretation
→ V1 = quality leader
→ V2 = identity/FP/speed trade-off
```

**END OF FINAL SCIENTIFIC FREEZE**
