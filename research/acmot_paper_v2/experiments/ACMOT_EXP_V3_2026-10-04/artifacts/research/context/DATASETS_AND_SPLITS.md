# DATASETS AND SPLITS

Leakage status is time-dependent: every evaluation, inspection or tuning use
changes it. Update this file in the same commit as any new use.
Annotation fingerprint = sha256 over sorted "file:sha256(file)" lines of
`annotations/*.txt` (same definition as tools/make_train_split.py).
Evaluation filter everywhere: GT classes {1,4,5,6,9}, score==1, trunc<2,
occ<2, class-agnostic, IoU 0.5 (internal protocol, not official VisDrone).

## Dataset: VisDrone2019-MOT
### Split: VisDrone2019-MOT-val — DEVELOPMENT (heavily used), SECONDARY CHECK for V5-TF
- Path (Mac): `/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val`
- 7 sequences, 2,846 frames, 67,345 GT boxes (research/EXPERIMENT_REGISTRY.md header)
- Fingerprint: a22a993fd46e77fa411f2b3b8ae31e7a8f4ef05b9a872fe7820da02ab3554d28 (computed 2026-09-27)
- Used for: all V1–V4 selection, V5 S1–S3 attempts (E32–E35). Inspected: yes.
- Clean for transfer: NO. For V5-TF: secondary check after freeze only.

### Split: VisDrone2019-MOT-train development-40 — DEVELOPMENT for V5-TF
- Path: Google Drive `…/My Drive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-train`
- Split file: `research/TRAIN_SPLIT_V5.json` (seed 20260927, length-quartile
  stratified, metadata only; created before any V5 train result)
- 40 sequences, 17,167 frames. Whole train fingerprint (56 files):
  045620e4ba7009dba97e495271254b468b50c01db85ca80c4fe90871a1e17e81
- Used for: V5-TF rule validation/ablation/stress (validate, never fit). No
  result yet (E36 pending). Caches: `outputs/det_cache_train*/` (in progress);
  `outputs/det_cache_train_native/{rtdetr,visual_cues}` are symlinks to `outputs/det_cache_train/`.

### Split: VisDrone2019-MOT-train confirmation-16 — PROTECTED
- 16 sequences, 7,034 frames (list in TRAIN_SPLIT_V5.json)
- Untouched: no metric computed. Caching is allowed (the train cache queue
  caches all 56 sequences; caches ≠ evaluation).
- Evaluated ONCE after the V5-TF freeze (V5-TF vs V4, Amendment-5a rule).

### Split: VisDrone2019-MOT-test-dev — HELD-OUT, USED ONCE → post-hoc only
- Path: Google Drive shortcut `…/visdrone goda1/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev`
- 17 sequences, 6,635 frames; fingerprint 89d9dccbad509f1f8d34bb2afde32cb82490d34242b266345188107… (full value in research/TESTDEV_LOCK_V4.json)
- History: legacy AC-MOT exploratory comparisons (2026-09-10/11, README
  states "exploratory test-dev comparison, not a held-out benchmark");
  Universal V4 evaluated once (E31). Clean for V5-TF: NO (post-hoc label mandatory).

### Diagnostic subsets
- val sequences uav0000137_00458_v, uav0000268_05773_v ("0137", "0268"): V2-era diagnostics (E03–E15).
- Fidelity-gate subset (Amendment 5f): uav0000020_00406_v, uav0000315_00000_v,
  uav0000316_01288_v, uav0000342_04692_v (development-40 members; YOLOv8n,
  RT-DETR-L at 736). Never confirmation sequences.
- `outputs/calibration/{yolov8,rtdetr}_pr_50f.csv` (2026-09-26, untracked
  tools/calibrate_detector_pr.py, GT-based detection PR on 50 frames of ONE
  sequence at 640): which sequence/split — NEEDS VERIFICATION. If it was a
  train confirmation-16 sequence, record it as a detection-level GT inspection
  (created before the split existed; no tracking metric).

## Dataset: UAVDT
### Split: UAVDT test — PROTECTED (unseen dataset for Universal AC-MOT)
- Source: Google Drive `…/AC-MOT-shared/UAVDT_EXTERNAL_GENERALIZATION`;
  VisDrone-layout view `outputs/uavdt_view` (tools/build_uavdt_view.py)
- 20 sequences, 16,592 frames; view fingerprint
  aceae1c89eb4de05526bfa96f42da38ff1c6a474861bddaa1fbfd3044728184a
- Protocol: UAVDT_EXTERNAL_PROTOCOL_FREEZE.json + UAVDT_ADAPTER_V1_FREEZE.json (frozen 2026-09-12)
- Universal AC-MOT quality metrics: none (locks LOCKED_BEFORE_EVALUATION; no transfer output dir).
- CAVEAT: the legacy AC-MOT line ran a frozen UAVDT external generalization
  (branch origin/freeze/final-after-uavdt-2026-09-12). Whether any legacy UAVDT
  result informed Universal design: NEEDS VERIFICATION (no evidence found in
  research/). Caches `outputs/det_cache_uavdt*/` exist (caching only).
