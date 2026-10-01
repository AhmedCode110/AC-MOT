# U2MOT and SparseTrack evidence audit (read-only)

Scope: the U2MOT (VisDrone) and SparseTrack (MOT17/MOT20) scene-adaptive
experiments run on Colab in September 2026, audited from the original Drive
files. Nothing in those folders, in Paper 1, or in any frozen artifact was
changed. Every number below is transcribed from the file named next to it;
derived values are marked "derived" and their formula is given.

Drive owners: `a7medgodaps@gmail.com` (U2MOT package, SparseTrack final test),
`miamimiamps4@gmail.com` and `a7medgouda2@gmail.com` (SparseTrack development).

---

## 1. U2MOT + scene-adaptive controller on VisDrone2019-MOT

Package: `FINAL_U2MOT_ACMOT_FREEZE_2026-09-14` (Drive folder
`1PgeAEKYEWRTsqLjQCcAuBXba4y8YdW2w`).

### 1.1 Pipeline (proven)

| Item | Value | Source |
|---|---|---|
| Tracker code | alibaba/u2mot, commit `7411211d17cb893f5fcd6be39cd4e5f91cfe1586` | BASELINE_RUN_SUMMARY.txt, FINAL_METRICS.json |
| Detector | YOLOX-X (`yolox_x_u2mot_visdrone.py`, 105.50 M params) | BASELINE_TRACKING.log (Args, Model Summary) |
| Checkpoint | `visdrone.pth.tar`, SHA256 `8cb39ca2…c594c939` (released by the U2MOT authors) | BASELINE_RUN_SUMMARY.txt, FREEZE_DECLARATION.txt |
| Development split | VisDrone2019-MOT-val, 7 sequences, 2846 frames | VALIDATION_COMPARISON_SUMMARY.txt |
| Held-out split | VisDrone2019-MOT-test-dev, 17 sequences, 6635 frames | BASELINE_RUN_SUMMARY.txt |
| Hardware | Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, NumPy 2.1.3 | BASELINE_RUN_SUMMARY.txt |

### 1.2 Static tracker/detector parameters (proven from upstream code)

The logged `Args` lines show `track_thresh=0.6, track_buffer=30,
min_box_area=100`, but these are the argparse defaults *before* the
benchmark override. Upstream `tools/track.py::parse_benchmark` (commit
7411211, lines 344–404) is called for every sequence before
`image_track` builds the tracker, and for `benchmark='VisDrone'` sets:

```
track_thresh = 0.5, low_thresh = 0.1, match_thresh = 0.8, track_buffer = 15,
aspect_ratio_thresh = 1e5, min_box_area = 1, fps = 30,
predictor.test_size = exp.test_size, predictor.confthre = max(0.001, low_thresh - 0.01) = 0.09
```

Both the baseline and the controller script
(`track_acmot_full_calibrated.py`) call `parse_benchmark`, so both run the
tracker with 0.5 / 0.1 / 0.8 / 15 / min_box_area 1. This resolves the
apparent conflict between the log Args and BASELINE_RUN_SUMMARY.txt.
Baseline detector: conf 0.09, NMS 0.70, input 1600×896, FP16, fused.
CMC: none on validation (the repository ships file-CMC for test-dev only);
file-CMC on test-dev for both systems.

### 1.3 Controller (proven from the frozen code)

`acmot_full_policy_calibrated.py` (SHA256 `985491bb…2cabc1`, listed in
05_VALIDATED_CONTROLLER_FREEZE/SHA256SUMS.txt and in FINAL_METRICS.json):

| SCI tier | Rule | conf | NMS | input (H, W) |
|---|---|---|---|---|
| easy | SCI < 0.51854530656461328 | 0.15 | 0.70 | (704, 1280) |
| medium | SCI < 0.58955478954401341 | 0.12 | 0.65 | (800, 1440) |
| hard | otherwise | 0.09 | 0.60 | (896, 1600) |

SCI: `SceneComplexityV1(analysis_interval=10, smoothing_window=7)` with the
weights of `SceneComplexityV1` in 07_CODE_SNAPSHOT/acmot_v1.py
(crowd 0.1295, tiny 0.2217, edge 0.4337, night 0.0536, blur 0.1615 — the
weights of research/paper_split/evidence/legacy/FROZEN_DEFENSIBLE_ACMOT_CONFIG.json,
trial 24), applied to fixed cue transforms (crowd = count/30, tiny = share of
boxes < 32², edge = Canny density/0.14, night = gray mean < 80, blur =
Laplacian variance < 180), unchanged per SCI_CALIBRATION.txt. The tracker is not adapted.

What was transferred and what was not:
- transferred unchanged: the cue weights listed above and the fixed cue transforms;
- recalibrated on U2MOT validation: the two SCI boundaries (unsupervised
  q33/q67 of the validation SCI with actions disabled; no GT, MOTA, IDF1 or
  IDS used — SCI_CALIBRATION.txt). The V1 boundaries were 0.1353 / 0.2873;
  the observed validation SCI range was 0.3415–0.7246, so the V1 boundaries
  would have put every frame in the hard tier;
- set for U2MOT: the three action tuples. No selection record for them was
  found; they are not the actions of Paper 1's Q profile or of V1.

The docstring of the frozen policy says the SCI boundaries were
"transferred directly from V1"; the code and SCI_CALIBRATION.txt show they
were recalibrated. The code is authoritative.

Consequence: this is **not** the Paper 1 controller (Q) and not the same
frozen controller as V1. It is a controller of the same design family,
recalibrated without labels on the U2MOT validation split.

### 1.4 Freeze before test-dev (proven)

- FREEZE_DECLARATION.txt, FROZEN_CONFIG.json and SHA256SUMS.txt:
  modified 2026-09-13 21:35 UTC.
- FINAL_TRACKING.log (test-dev controller run): first line 21:37:55 on the
  same day, i.e. after the freeze.
- FINAL_METRICS.json records `policy_sha256` = `985491bb…` and
  `tracking_script_sha256` = `3d5a5c74…`, identical to the freeze
  SHA256SUMS.txt.
- FINAL_METRICS.json: `controller_frozen_before_test: true`,
  `test_dev_used_for_tuning: false`.

### 1.5 Evaluator (evidence)

- Baseline test-dev: U2MOT repository `tools/utils/eval_visdrone.py`
  (motmetrics 1.4.0, IoU distance 0.5, classes {1,4,5,6,9} merged
  class-agnostically, ignore regions of classes 0 and 11 removed by box
  centre). OFFICIAL_EVALUATION.txt is the first attempt and ends in the
  NumPy 2 `np.asfarray` crash; OFFICIAL_EVALUATION_FINAL.txt is the rerun
  with a runtime-only alias `np.asfarray(a, dtype) = np.asarray(a,
  dtype=dtype)` (EVALUATION_COMPATIBILITY.txt; evaluator source, predictions
  and GT not modified). The alias is numerically neutral for the float
  arrays motmetrics passes to it. The script prints metrics at one decimal.
- Controller test-dev: no evaluator log or console output was archived.
  Indirect evidence that the same evaluator produced FINAL_METRICS.json:
  (a) the field set (IDF1, IDP, IDR, Precision, Recall, MOTP as a 1−IoU
  distance of 0.229) is motmetrics' `motchallenge_metrics`; (b) derived:
  the GT object count implied by the controller's MOTA and error counts,
  (40155 + 64801 + 1152) / (1 − 0.5376678605352365) = 229506, equals the
  count implied by the baseline's error counts and MOTA, so both runs used
  the same class/ignore filtering of the same GT.
- Neither evaluation is the official VisDrone MATLAB toolkit and neither
  computes HOTA. Status: **U2MOT-repository motmetrics protocol; not
  official; the controller-side call is inferred, not logged.**

### 1.6 Results

Validation (development; boundaries calibrated on the same sequences, so
not held-out) — VALIDATION_COMPARISON_SUMMARY.txt, CMC none for both:

| | MOTA | IDF1 | IDS | FP | FN | Precision | Recall | wall FPS |
|---|---|---|---|---|---|---|---|---|
| Baseline | 75.127320 | 75.557816 | 1947 | 8837 | 7091 | 87.995164 | 90.133025 | 4.922483 |
| Controller | 75.473798 | 76.577452 | 1908 | 7992 | 7726 | 88.920313 | 89.249436 | 4.584234 |
| Δ | +0.346478 | +1.019636 | −39 | −845 | +635 | | | −6.87 % |

Matched runtime on validation (MATCHED_RUNTIME_SUMMARY.txt; same dynamic
runtime implementation for both systems): fixed baseline 269.73 ms/frame
(3.707 FPS), controller 214.48 ms/frame (4.663 FPS), +25.76 %.

Held-out test-dev (FINAL_METRICS.json):

| | MOTA | IDF1 | IDS | FP | FN |
|---|---|---|---|---|---|
| Baseline (printed at 1 decimal) | 53.9 | 69.8 | 1239 | 41385 | 63241 |
| Baseline MOTA from counts (derived) | 53.8727 | — | | | |
| Controller | 53.76679 | 69.84939 | 1152 | 40155 | 64801 |
| Δ as stored in FINAL_METRICS.json | −0.1332 (vs rounded 53.9) | +0.0494 (vs rounded 69.8) | −87 | −1230 | +1560 |
| Δ exact (derived) | **−0.1059** | in (−0.001, +0.099] | −87 | −1230 | +1560 |

Derived: baseline MOTA = 1 − 105865/229506; ΔMOTA = (105865 − 106108)/229506.
The exact baseline IDF1 is not recoverable from the archived output.

Timing on test-dev: controller mean 245.00 ms/frame (4.08 FPS, same-method
timer). The baseline test-dev run has only a wall time (1307.64 s, derived
5.07 FPS including I/O), measured differently; there is **no matched
test-dev timing comparison**. Per-sequence wall/core FPS for the controller
are in FINAL_TRACKING.log; per-frame controller trace (SCI, tier, conf,
NMS, input size, cues, frame ms) in FINAL_ACMOT_FRAME_LOG.csv and
06_FINAL_TESTDEV_17SEQ/acmot_logs.

Missing: per-sequence test-dev metrics, HOTA, any bootstrap or other
statistics. Both systems' per-sequence test-dev predictions are archived
(01_A0_BASELINE/track_res, 06_FINAL_TESTDEV_17SEQ/track_res), so all of
these can be computed without rerunning tracking.

Reading: on held-out data the controller did not change accuracy measurably
(MOTA −0.11, IDF1 within ±0.1, fewer FP and IDS, more FN). The validation
gain (+0.35 MOTA, +1.02 IDF1) did not transfer. The throughput gain is
established on validation only.

---

## 2. SparseTrack + Adaptive Edge V1 on MOT17 / MOT20

### 2.1 Pipeline and controller (proven from the frozen runner)

Frozen manifest `FROZEN_CONTROLLER_MANIFEST.json` (Drive
`1IhAmHmoHxKMRk1xcI14bgDN81rqz-_wo`, folder
`FROZEN_ADAPTIVE_EDGE_V1_20260917T095152Z`); its SHA256
`5356d8270c2616bb12af52af56c56392062daf644cd51de8c4c200a9209083d7` is
quoted by every later stage. Frozen runner
`frozen_adaptive_edge_v1_runner.py`, SHA256 `e2f2ea62…997ca9`.

| Item | Value |
|---|---|
| Tracker | SparseTrack, commit `499844f32c5bb2332f9811f26cd70cf4e517d4e7` (tree clean); Detectron2 `a2f4a877…` |
| Detector | YOLOX, `bytetrack_ablation.pth.tar` (SHA256 `26cb8d28…c48b1a`; upstream: CrowdHuman + MOT17 half-train), input 800×1440, FP16, fused |
| Static detector | conf 0.01 (asserted), baseline NMS 0.70 |
| Tracker (frozen, asserted) | track_thresh 0.60, track_buffer 30, match_thresh 0.85; sequence-specific overrides disabled |
| Development split | MOT17 val_half, 7 FRCNN sequences, 2652 frames, GT `gt_val_half.txt` |

Controller rule (`AdaptiveNMSWrapper` in the frozen runner): every 10th
frame (1, 11, 21, …) the BGR frame is downscaled ×0.25, converted to gray,
Canny(50, 120); `edge_norm = min(mean(edges)/255 / 0.14, 1)`;
`edge_smooth7` = mean of the last 7 `edge_norm` values;
**NMS = 0.70 if edge_smooth7 ≤ 0.9675843253968254, else 0.80**; held until
the next analysis frame; reset to 0.70 at frame 1 of each sequence. Only the
detector NMS threshold changes. It is named "Adaptive Edge V1" and is **not**
Paper 1's Q policy: different cue (single edge cue, not the SCI), different
action (NMS only), different detector, tracker and dataset.

Threshold origin: `STEP7H_EDGE_CONTROLLER_CANDIDATE.json` — a one-feature
decision stump from a leave-one-sequence-out search on the same 7
sequences (modal threshold in 6 of 7 folds). The feature is nearly constant
within a sequence (STEP7H_EDGE_CONTROLLER_SUMMARY.csv): MOT17-02 and -05
stay at 1.0 (always 0.80), MOT17-04 stays at 0.85 (always 0.70); 12 decision
switches in 268 decisions overall.

### 2.2 Validation (MOT17 val_half; development, not held-out)

Evaluator: motmetrics `compare_to_groundtruth` (IoU 0.5), `mot15-2D`,
GT `min_confidence=1`, implemented inside the runner (`compute_metrics`),
with the same NumPy 2 `np.asfarray` alias. Not TrackEval; no HOTA.

Static NMS sweep (STEP7B_NMS_COMPARISON.csv; 0.70 row "S1_FROZEN_REUSED"):

| NMS | MOTA | IDF1 | IDS | FP | FN |
|---|---|---|---|---|---|
| 0.60 | 75.8786 | 80.3162 | 126 | 2707 | 10166 |
| 0.70 (default) | 76.6747 | 81.6186 | 116 | 2850 | 9604 |
| 0.75 | 76.8714 | 81.4855 | 136 | 2937 | 9391 |
| 0.80 | 76.8788 | 81.4388 | 142 | 3047 | 9271 |
| 0.85 | 75.9937 | 80.5417 | 232 | 3559 | 9146 |

Matched deterministic comparison (FROZEN_CONTROLLER_MANIFEST.json,
FINAL_FREEZE_GATE_SUMMARY.json, metrics.csv):

| | MOTA | IDF1 | IDS | FP | FN |
|---|---|---|---|---|---|
| Static 0.75 (deterministic rerun) | 76.8881 | 81.4943 | 136 | 2927 | 9392 |
| Adaptive Edge V1 | 76.9252 | 81.5299 | 123 | 2910 | 9402 |
| Δ | +0.0371 pp, 95 % bootstrap CI [−0.041, +0.182], 4 wins / 3 losses, LOSO not all positive | mean per-sequence ΔIDF1 −0.003, CI [−0.359, +0.282] | | | |

The manifest itself states: "freeze does NOT mean superiority over static
0.75 has been statistically established." Adaptive frame split: 1131 frames
at NMS 0.70, 1521 at 0.80.

Derived (descriptive): moving the static NMS from 0.70 to 0.75 adds +0.20
MOTA (non-deterministic sweep) or +0.21 (deterministic 0.75 rerun vs the
reused 0.70 run); switching on top of the matched static adds +0.04 (n.s.).
The 0.70 and 0.75 rows come from different run batches, so this split is
descriptive only.

### 2.3 MOT17 test (held-out; no score)

`protocol.json` and `CROSSACCOUNT_AUDIT.json` in
`FROZEN_ADAPTIVE_EDGE_V1_CROSSACCOUNT_FIX2_20260917T104734Z`:
7 FRCNN test sequences, 5919 frames; controller manifest SHA256 `5356d827…`
(matches the freeze); `controller_retuned: false`,
`scientific_parameters_changed: false`, `test_gt_used: false`.
`CROSSACCOUNT_FIX2_PATCH.diff` changes only the output path and a
verification gate (AST check that the tracker threshold/buffer are only
restored to their frozen values). Controller: 2050 frames at 0.70, 3869 at
0.80, 593 analysed frames. A 21-file submission (FRCNN results copied to the
DPM/SDP names; `MOT17_SUBMISSION_MANIFEST.json` with per-file SHA256) and
`MOT17_FROZEN_ADAPTIVE_EDGE_V1_SUBMISSION.zip` exist. No MOTChallenge server
result was found, and no static-NMS test submission exists for comparison.
Timing: wall 6301.5 s (0.94 FPS), flagged by the protocol itself as not a
paper FPS (Drive-backed loader, 0 workers).

### 2.4 MOT20 external generalization (not run)

`SPARSETRACK_MOT20_EXTERNAL_GENERALIZATION_20260917T143453Z`:
`EXPERIMENT_PROTOCOL.json` is frozen before any result (four conditions:
static 0.70 / 0.75 / 0.80 and the frozen adaptive rule; primary comparator
static 0.75; TrackEval HOTA/CLEAR/Identity; 50 000-sample sequence bootstrap;
LOSO). `MOT20_PRE_ACQUISITION_STATUS.json`: data not present, no tracking,
no evaluation. All result folders are empty. **No MOT20 result exists.**

---

## 3. Evidence eligibility for Paper 1

"Frozen before evaluation" is proven by file timestamps plus hashes as
described above. "Held-out" means the sequences played no role in any
selection or calibration.

| pipeline | detector | tracker | dataset | policy | tuning/calibration used | frozen before evaluation? | evaluator | held-out? | statistics | timing | usable in Paper 1? | required caveat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U2MOT val | YOLOX-X (U2MOT VisDrone ckpt) | U2MOT (official VisDrone profile) | VisDrone val-7 | SCI V1 weights, recalibrated q33/q67 boundaries, 3 hand-set tiers (conf/NMS/size) | boundaries: unsupervised quantiles on these 7 sequences; tiers: no record | n/a (development) | U2MOT-repo motmetrics (runner-side) | no | none | matched: 3.707 → 4.663 FPS | usable with caveat (development only) | same sequences as the calibration; no HOTA, no CI |
| U2MOT test-dev | same | same | VisDrone test-dev, 17 seq | frozen controller above | none on test-dev | yes (21:35 freeze, 21:37 run, hashes match) | U2MOT-repo `eval_visdrone.py` (motmetrics) for the baseline; same protocol inferred for the controller | yes | none (aggregate only) | controller 4.08 FPS; no matched baseline timing | usable with caveat | not the official VisDrone toolkit; no HOTA; no CI; baseline IDF1 only at 1 decimal; ΔMOTA must be stated as −0.11 (from counts), not −0.13; controller evaluator call not logged; no speed claim on test-dev |
| SparseTrack val (static sweep) | YOLOX (ByteTrack ablation ckpt) | SparseTrack | MOT17 val_half, 7 seq | static NMS 0.60–0.85 | none (sweep) | n/a | runner motmetrics | no | none per setting | diagnostic wall FPS only | usable with caveat | development split; 0.70 row from a different batch; MOTA/IDF1 only |
| SparseTrack val (Adaptive Edge V1 vs static 0.75) | same | same | MOT17 val_half | edge_smooth7 → NMS 0.70/0.80 | threshold chosen by LOSO on the same 7 sequences | freeze after this comparison | runner motmetrics | no (in-sample) | sequence bootstrap CI, W/L, LOSO | diagnostic only | usable with caveat | in-sample; CI includes zero; cue nearly constant per sequence |
| SparseTrack MOT17 test | same | same | MOT17 test, 7 seq | frozen Adaptive Edge V1 | none | yes (manifest SHA256 matched; verification-gate-only patch) | none (no server score) | yes | none | not a paper FPS | not yet usable | needs a MOTChallenge server score, and a static-NMS submission for a comparison |
| SparseTrack MOT20 | same | same | MOT20 train, 4 seq | frozen Adaptive Edge V1 | none | protocol frozen | TrackEval (planned) | yes | planned | none | not yet usable | experiment not run |

Claims the files do **not** support:
- "the same frozen controller across three pipelines": the U2MOT controller
  kept only the SCI weights; its boundaries were recalibrated and its actions
  are U2MOT-specific; the SparseTrack controller is a different
  single-cue NMS rule.
- any statement that Adaptive Edge V1 is Q.
- any official VisDrone or MOTChallenge number for these two pipelines.
- a test-dev speed-up for U2MOT.

Superseded by the re-scoring (section 7). Before re-scoring this paragraph
said that switching added no measurable accuracy across the three pipelines;
that is not true for SparseTrack in HOTA against static NMS 0.75 and must not
be used.

## 4. Work that would make these rows fully usable (no tuning involved)

1. Re-score the archived U2MOT test-dev predictions of both systems with one
   evaluator in one run: full-precision motmetrics (U2MOT protocol) and
   TrackEval HOTA on the same filtered GT, per sequence, with a paired
   sequence bootstrap. Script: `tools/legacy_rescore/rescore_legacy_pipelines.py`.
2. Same for SparseTrack val_half (static 0.70/0.75/0.80 and adaptive),
   adding HOTA to the existing MOTA/IDF1.
3. SparseTrack MOT17 test: submit the frozen adaptive submission and a static
   NMS 0.75 submission to the MOTChallenge server (the frozen manifest names
   static 0.75 as the comparator).
4. MOT20: run the frozen protocol as written.
5. U2MOT test-dev matched timing: time the fixed baseline with the same
   dynamic runtime used for the controller (as done on validation).

## 5. Source file index (Drive IDs)

U2MOT: FINAL_METRICS.json `1OcvMkiUOMXFH6dbMIFGpBJm0WWo089-c`;
FINAL_TRACKING.log `1pZNPQVoyyZXl5QRl3tKeKpYw3HM49Cxu`;
BASELINE_RUN_SUMMARY.txt `1x0QwAyBRtjHMzUIjQRojz8B-W2T-ioi6`;
OFFICIAL_EVALUATION_FINAL.txt `1YRvE60ksGF58R2NOOW7_Hz0Z4L3iM9Fm`;
EVALUATION_COMPATIBILITY.txt `1Rg6IFwR6TOFCeKx3mOoxRbaR9s6f5skI`;
SCI_CALIBRATION.txt `1k0wiYVDJRONBL-PctxA7GqE5ubljzcDb`;
VALIDATION_COMPARISON_SUMMARY.txt `1CQBoTPNojzNAo44Kzy1455-YlEovu06F`;
MATCHED_RUNTIME_SUMMARY.txt `1oY1sjm_-t27IlacjObI4pySNv9kT8Z4v`;
FREEZE_DECLARATION.txt `1V1ZvjmEILgMroi8K5IRdrUtA_CJHRJc2`;
FROZEN_CONFIG.json `1WMdWrMPc-gJc4FdZNEh1TkwjOPcSiYKE`;
BASELINE_TRACKING.log (val) `10eytcqAeCfnFrNMoLxwz5uRrPMqcHGPW`;
baseline test-dev track_res folder `1Tt-2LIjFMIbeTI5Ku48ZMHUaAM6-oWc-`;
controller test-dev track_res folder `1ZSOUOk0xQ4RXiNPiWsdNtS3f6cWaRALd`;
VisDrone test-dev annotations folder `1zCZCDZv1lqEJ-Jbjr-YrSwX1GI2dxwiJ`.

SparseTrack: FROZEN_CONTROLLER_MANIFEST.json `1IhAmHmoHxKMRk1xcI14bgDN81rqz-_wo`;
frozen runner `1ybeR6-xMWcmhDYt1B8FbhKShhbT9MRQj`;
validation metrics.csv `1mfEpXge_sn7BOI6uk2-4R9xc5KW2nngu`;
controller_summary.json (val) `1fA6SeAYkWmhITvxDpR5X8txeRV2Pv3Dp`;
FINAL_FREEZE_GATE_SUMMARY.json `1HTXSUKUF0lNjZCaQ-8rLlKzDfFw65Iaq`;
ADAPTIVE_VS_STATIC075_PER_SEQUENCE.csv `14E4e8JIU7vDV6BM5vZz1T0nG5QSymmPa`;
LOO_STATIC075.csv `1mu53nNpZ5ngmS_O0jzEmMnjV4j2pu3Hm`;
STEP7B_NMS_COMPARISON.csv `1jLwsDLDk2Xxux9pA5QWhUJIbeGZuDkER`;
STEP7H_EDGE_CONTROLLER_CANDIDATE.json `1Fj5MVRLg5elhaarTFeqxmC3UiBSU52LH`;
STEP7H_EDGE_CONTROLLER_SUMMARY.csv `1bpDt_zQ4XaxCWkHEoK5bBWHiQYIYRXWL`;
MOT17 test protocol.json `1hpDhuiUZmYp0u1J2EXDrWVnTuBn_gIq3`;
CROSSACCOUNT_AUDIT.json `1MwPlmr3FH_T3dduhjW5e6EI4QK1MkmlO`;
CROSSACCOUNT_FIX2_PATCH.diff `1HrG14b9ppEABt6EFQTXfwBAntLZNyH0i`;
MOT17 test controller_summary.json `1qyC_KrIdgIekY3ihAJKzWzW-Cs9wjVeQ`;
MOT17_SUBMISSION_MANIFEST.json `18gvHpE8AJWWmHUDfiSDXEnLzqBnDpEsy`;
MOT20 EXPERIMENT_PROTOCOL.json `14unHmvv-hsiDT4CyBQyJEWNL-UxBLxQ5`;
MOT20_PRE_ACQUISITION_STATUS.json `1y_I6CACSCIdw3Osfo7tDkGB6-BIAKJOA`.

Upstream U2MOT code checked: `tools/track.py` (parse_benchmark, lines
344–404; tracker built in image_track at line 196 after the override) and
`tools/utils/eval_visdrone.py` at commit 7411211.

## 6. Interpretation rules, fixed before the re-scoring is run

Written on 2026-10-01, before any output of
`tools/legacy_rescore/rescore_legacy_pipelines.py` exists for the real data.

1. Primary metric HOTA (as in Paper 1); MOTA and IDF1 secondary. Every
   comparison is reported as Δ, 95 % sequence-bootstrap CI (10 000
   resamples, seed 0), win/tie/loss, and leave-one-out range.
2. A row is used only if its reproduction check matches the archived
   numbers. If the U2MOT controller row does not reproduce, the archived
   controller numbers are reported as not reproducible and the row is not
   used.
3. CI of ΔHOTA includes 0 → "no measurable change in aggregate accuracy".
   CI excludes 0 → the direction and size are reported as measured, for that
   pipeline only, without generalisation.
4. U2MOT comparator wording: "the author-calibrated static operating point
   of U2MOT" (conf 0.09, NMS 0.70, 1600×896, which is also the controller's
   hard tier), not "a matched static anchor"; the controller is described
   as a separately recalibrated controller of the same design family, never
   as the Paper 1 controller transferred unchanged. ΔMOTA from the archive
   is −0.11 (exact counts), not −0.13.
5. SparseTrack: primary pair Adaptive Edge V1 vs static NMS 0.75 (named in
   the frozen manifest). Reported as a supporting validation experiment
   (in-sample threshold), not as held-out or external evidence, until the
   MOT17 test or MOT20 result exists. The static 0.70 → 0.75 change is
   descriptive (different run batches).
6. No test-dev speed claim for U2MOT until a matched test-dev timing exists;
   the +25.8 % applies to validation only.
7. Response to the "one detector + one tracker" comment: the concern is
   reduced, not resolved, because the same qualitative outcome recurs in
   other pipelines while the policies are not identical.

## 7. Re-scoring results (research/transfer_legacy/rescore/, commit 97853c6)

Reproduction gate: passed for all systems (U2MOT baseline and controller;
SparseTrack static 0.70, 0.75, 0.80, adaptive). Values below are TrackEval;
Δ = first − second, 95 % sequence-bootstrap CI, W/T/L per sequence.

U2MOT, VisDrone2019-MOT-test-dev, 17 sequences (held-out; post-hoc TrackEval
HOTA of frozen outputs; U2MOT class/ignore filtering):

| | HOTA | DetA | AssA | MOTA | IDF1 |
|---|---|---|---|---|---|
| author-calibrated static | 55.00 | 48.51 | 63.28 | 53.82 | 69.78 |
| controller | 54.94 | 48.28 | 63.39 | 53.73 | 69.85 |
| Δ | −0.06 [−0.31, +0.22], 9/0/8 | −0.23 [−0.57, +0.07] | +0.11 [−0.25, +0.51] | −0.09 [−0.68, +0.45] | +0.07 [−0.30, +0.46] |

motmetrics (U2MOT protocol, exact): baseline MOTA 53.873, IDF1 69.775;
controller 53.767, 69.849.

SparseTrack, MOT17 val_half, 7 sequences (development; adaptive threshold
chosen on these sequences; official TrackEval MOT17 preprocessing):

| | HOTA | DetA | AssA | MOTA | IDF1 |
|---|---|---|---|---|---|
| static 0.70 (SparseTrack default) | 69.17 | 66.71 | 72.20 | 77.87 | 82.10 |
| static 0.75 | 68.96 | 66.57 | 71.93 | 77.94 | 81.97 |
| static 0.80 | 68.92 | 66.63 | 71.78 | 78.06 | 81.93 |
| Adaptive Edge V1 | 69.10 | 66.75 | 72.04 | 78.12 | 82.01 |

| pair | ΔHOTA | ΔMOTA |
|---|---|---|
| adaptive − static 0.75 | +0.13 [+0.02, +0.24], 6/0/1 | +0.18 [+0.08, +0.28] |
| adaptive − static 0.70 | −0.07 [−0.54, +0.19], 4/0/3 | +0.25 [+0.09, +0.35], 7/0/0 |
| static 0.75 − static 0.70 | −0.20 [−0.62, +0.01], 2/0/5 | +0.07 [−0.13, +0.19] |
| static 0.80 − static 0.70 | −0.25 [−0.52, +0.19], 2/0/5 | +0.19 [+0.07, +0.35] |

Reading. In HOTA the best static operating point of the sweep is the
default NMS 0.70, not 0.75; static 0.75 was the comparator named in the
frozen manifest because it was chosen by MOTA. Against static 0.75 the
adaptive rule gains +0.13 HOTA (CI excludes 0); against the HOTA-best static
0.70 it is −0.07 (CI includes 0). In MOTA the adaptive rule is above every
static setting. So the SparseTrack gain depends on the metric and on which
static point is the comparator; it does not show that switching beats the
best static operating point in HOTA. All of this is in-sample validation.
The earlier statement "static 0.70 → 0.75 adds +0.20 MOTA" was from
motmetrics; in TrackEval HOTA the same change is −0.20.
