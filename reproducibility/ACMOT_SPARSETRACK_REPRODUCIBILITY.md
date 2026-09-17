# AC-MOT + SparseTrack — Reproducibility and Colab Handoff

**Status:** frozen controller, validation tuning closed  
**Date:** 2026-09-17

This file is the authoritative handoff for running the AC-MOT + SparseTrack experiment again in a fresh Google Colab session or from another Google account. Read it before running anything.

## Scientific state

The experiment uses the released SparseTrack code/checkpoint as the baseline and adds AC-MOT logic only on the detector side. SparseTrack is **not retrained**. Tracker internals remain fixed.

Scientific labels:

- `S0-LIT`: literature result only.
- `S1-RUNTIME`: reproduced released code/checkpoint under the pinned validation protocol.
- `S2-ACMOT`: same code/checkpoint/dataset/evaluator/tracker plus detector-side adaptive control.

Do not mix literature results with reproduced runtime results.

The confirmed runtime reproducibility issue was a worker-setting alias mismatch. Detectron2 LazyConfig loaded the dataset builder dynamically, therefore changing only the top-level `datasets.builder.opt` did not necessarily modify the loader actually used. The authoritative runtime patch is:

```python
target.__globals__["opt"].DATALOADER.NUM_WORKERS = 0
```

Determinism is a combined protocol, not a workers-only claim:

```text
PYTHONHASHSEED=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
actual DataLoader NUM_WORKERS=0
fresh subprocess for scientific runs
```

A deterministic A/A run reproduced exactly across Google accounts. Canonical deterministic static `NMS=0.80` metrics and controller traces were byte-exact.

## Frozen adaptive controller

The final controller is **Adaptive Edge V1**:

```text
feature          = edge_smooth7
threshold        = 0.9675843253968254
if edge_smooth7 <= threshold -> detector NMS 0.70
if edge_smooth7 >  threshold -> detector NMS 0.80
analysis stride  = 10 frames
smoothing window = 7
downscale        = 0.25
Canny            = 50 / 120
edge divisor     = 0.14
detector conf    = 0.01
tracker           = track_thresh 0.60 / buffer 30 / match_thresh 0.85
```

The controller used both states on validation:

```text
NMS 0.70 -> 1131 frames -> 42.65%
NMS 0.80 -> 1521 frames -> 57.35%
```

The controller is physically frozen at:

```text
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/adaptive/FROZEN_ADAPTIVE_EDGE_V1_20260917T095152Z
```

Freeze manifest SHA256:

```text
5356d8270c2616bb12af52af56c56392062daf644cd51de8c4c200a9209083d7
```

After this freeze, validation tuning is closed. Do not change threshold, NMS states, stride, smoothing, image size, detector confidence, or tracker settings.

## Matched deterministic validation results

| Condition | MOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static NMS 0.70 | 76.8658 | 81.6698 | 118 | 2794 | 9555 | 94.0716 | 82.2694 |
| Static NMS 0.75 | 76.8881 | 81.4943 | 136 | 2927 | 9392 | 93.8281 | 82.5719 |
| Static NMS 0.80 | 76.8528 | 81.1935 | 150 | 3063 | 9261 | 93.5775 | 82.8150 |
| Adaptive Edge V1 | **76.9252** | 81.5299 | 123 | 2910 | 9402 | 93.8605 | 82.5533 |

Adaptive versus static `0.75`:

```text
Delta MOTA = +0.037113 percentage points
Delta IDF1 = +0.035546 percentage points
Delta IDS  = -13
Delta FP   = -17
Delta FN   = +10
```

Final validation stability gate against static `0.75`:

```text
MOTA W/T/L across 7 sequences = 4 / 0 / 3
Leave-one-sequence-out all positive = False
Bootstrap 95% CI Delta MOTA = [-0.040608, +0.182353] pp
Bootstrap P(Delta MOTA > 0) = 0.755060
```

Correct interpretation: Adaptive Edge V1 has a small positive aggregate MOTA gain, but superiority over static `0.75` is **not statistically established** from these seven validation sequences. Do not claim otherwise.

## No-leakage state

```text
Training                     : NO
MOT17 test used for tuning   : NO
Controller retuned after gate: NO
Controller status            : FROZEN
Validation tuning            : CLOSED
Validation split             : MOT17 train -> val_half
Validation frames            : 2652
Validation sequences         : 7
GT objects                   : 53890
```

## Pinned source

SparseTrack:

```text
commit 499844f32c5bb2332f9811f26cd70cf4e517d4e7
```

Detectron2:

```text
commit a2f4a8771ab77e8411c26b27f24f9489a28a2453
```

pbcvt:

```text
commit ea95d4c2cf72265c8ff6610cc212cc2ceb662322
```

Checkpoint:

```text
bytetrack_ablation.pth.tar
size 792835795 bytes
local SHA256 26cb8d2808664e5068a4c812d53becbc948b47fd6eacf2b45db049ab40c48b1a
```

The checkpoint hash is a locally computed project checksum, not an author-published official hash.

## Pinned SparseTrack evaluation configuration

```text
test_size           = (800, 1440)
infer_batch         = 1
YOLOX depth         = 1.33
YOLOX width         = 1.25
num_classes         = 1
detector confidence = 0.01
track_thresh        = 0.60
track_buffer        = 30
match_thresh        = 0.85
min_box_area        = 100
down_scale          = 4
depth_levels        = 1
depth_levels_low    = 8
confirm_thresh      = 0.70
mot20               = False
byte                = False
deep                = True
bot                 = False
sort                = False
ocsort              = False
fp16                = True
fuse                = True
val_ann             = "val_half.json"
is_public           = False
```

## Validation sequences

```text
MOT17-02-FRCNN    299 frames
MOT17-04-FRCNN    524 frames
MOT17-05-FRCNN    418 frames
MOT17-09-FRCNN    262 frames
MOT17-10-FRCNN    326 frames
MOT17-11-FRCNN    449 frames
MOT17-13-FRCNN    374 frames
Total            2652 frames
```

Do not use MOT17 test for controller tuning.

## Canonical saved environment

Project root:

```text
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW
```

Environment bundle:

```text
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/runtime/env_bundles/ACMOT_SPARSETRACK_ENV_20260916T205448Z
```

Pointer:

```text
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/runtime/LATEST_ENV_BUNDLE.txt
```

Restore entrypoint:

```text
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/runtime/restore_latest_env.py
```

Base runtime:

```text
Python              3.13.15
PyTorch             2.11.0+cu128
torchvision         0.26.0+cu128
CUDA                12.8
GPU                 Tesla T4
Compute capability  7.5
NumPy               2.1.3
Python OpenCV        5.0.0
lap                 0.5.13
```

Fail fast if Python/Torch/CUDA do not match. Do not silently continue.

Critical Python package versions are stored in `requirements_sparse_track.lock.txt`. The authoritative restore is still the saved Drive bundle because it also contains wheels, native `.deb` files, Detectron2, pbcvt, Boost.Python, overlays and checksums.

Native OpenCV system packages are Ubuntu `4.6.0+dfsg-13.1ubuntu1` t64 builds required by pbcvt. Python `cv2` remains 5.0.0. This coexistence is intentional.

Boost.Python 1.88.0 is restored from the saved environment. The exact library is preloaded because changing `LD_LIBRARY_PATH` after Python startup is not always sufficient:

```text
libboost_python313.so.1.88.0
```

## Package-shadowing fix

A cross-account run exposed `datasets` resolving to Hugging Face instead of SparseTrack. The runtime-only shim is:

```text
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/runtime/overlay/datasets/__init__.py
```

The overlay must be before external packages in `PYTHONPATH`. Do not modify the pinned SparseTrack source to solve this.

## Fresh Colab procedure

1. Select a Tesla T4 runtime.
2. Mount Google Drive.
3. Ensure the project is accessible at `/content/drive/MyDrive/AC-MOT-SparseTrack-NEW`.
4. Run the GitHub bootstrap:

```python
!curl -fsSL \
  https://raw.githubusercontent.com/AhmedCode110/AC-MOT/main/reproducibility/colab_sparse_track_bootstrap.py \
  -o /content/colab_sparse_track_bootstrap.py

%run /content/colab_sparse_track_bootstrap.py
```

5. Do not start science unless it ends with:

```text
AC-MOT + SparseTrack bootstrap: PASS
```

The bootstrap restores the authoritative Drive environment, verifies versions and source commits, checks imports and determinism, detects the frozen controller, and writes an audit record back to Drive. It does **not** start tracking, training or test evaluation.

## Scientific child-process environment

Use fresh subprocesses and preserve:

```text
PYTHONHASHSEED=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONPATH=<overlay_tracker>:<overlay>:<SparseTrack>:<Detectron2>
LD_LIBRARY_PATH=<Boost.Python lib>:...
```

Also enforce:

```text
cudnn.benchmark=False
cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
actual DataLoader NUM_WORKERS=0
```

## Result preservation

Never reuse an old output directory. Every scientific run gets a unique timestamped Drive path.

Preserve at least:

```text
runner.log
metrics.csv
summary.json
controller_trace.csv
controller_summary.json
COMPLETE.txt
environment_verification.json
source_commit.txt
protocol.json
```

Save SHA256 hashes for final scientific artifacts.

Do not report wall-clock diagnostic FPS as paper FPS.

## Cross-account behavior

The same project has already reproduced deterministically from another Google account. A known Drive shortcut target is:

```text
1iE_8E7msCxbPCcOKJ0eKQIABg8_HiB9W
```

and may resolve inside Colab to:

```text
/content/drive/.shortcut-targets-by-id/1iE_8E7msCxbPCcOKJ0eKQIABg8_HiB9W/AC-MOT-SparseTrack-NEW
```

Cross-account deterministic evidence established byte-exact metrics, exact controller trace and 7/7 exact sequence result hashes.

## Rules after freeze

1. Do not tune on MOT17 test.
2. Do not change controller threshold.
3. Do not change low/high NMS values.
4. Do not change tracker thresholds.
5. Do not change image size in this experiment.
6. Do not retrain unless a separately defined experiment is created.
7. Do not use historical nondeterministic numbers as matched final comparators.
8. Report the `0.75` stability result honestly.
9. If frozen-test performance is poor, report it; do not retune on test.
10. Keep validation tuning closed.

## Authoritative paths

```text
PROJECT
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW

SPARSETRACK
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/SparseTrack

RUNTIME
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/runtime

RESTORE
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/runtime/restore_latest_env.py

DETERMINISTIC RUNNER
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/runtime/tests/step7k_deterministic_aa_A.py

MATCHED ADAPTIVE RESULT
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/adaptive/STEP7K_E_FIX2_MATCHED_DETERMINISTIC_20260917T090646Z/ADAPTIVE_EDGE_V1/NMS_070

STATIC 0.75 RESULT
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/adaptive/STEP7K_H_STATIC075_20260917T093716Z/STATIC_075/NMS_075

FREEZE EVIDENCE
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/adaptive/STEP7K_I_FINAL_FREEZE_GATE_20260917T095010Z

FROZEN CONTROLLER
/content/drive/MyDrive/AC-MOT-SparseTrack-NEW/adaptive/FROZEN_ADAPTIVE_EDGE_V1_20260917T095152Z
```

## Final instruction to any future assistant

Do not rebuild this project from scratch. Restore the pinned environment from Drive, run the bootstrap, verify `PASS`, use the exact frozen controller, preserve outputs under new unique paths, and never reopen validation tuning after the freeze.