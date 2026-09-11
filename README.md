# AC-MOT: Adaptive Complexity-Aware Multi-Object Tracking

**Paper release candidate — final held-out evaluation locked on 2026-09-11.**

AC-MOT is a real-time multi-object tracking research pipeline built around **YOLOv8n**, **ByteTrack**, and a lightweight **Scene Complexity Index (SCI)** controller. The project studies whether detector operating points can be selected from scene-derived cues while preserving real-time performance and improving tracking quality.

This branch is based on the frozen research commit:

```text
a6c1fa49fce1d402513c2df05b7d04b962a6e89e
```

The frozen held-out protocol was created before test-dev exposure, and the final lock explicitly records that no selection or tuning was performed on the held-out test set.

> **Evaluation caveat:** all reported MOTA/HOTA/IDF1 results in this release use the project's **custom class-agnostic AC-MOT TrackEval protocol**. They are **not official VisDrone leaderboard scores**.

---

## Final held-out result

Dataset split: `VisDrone2019-MOT-test-dev`  
Sequences: **17**  
Frames: **6635**  
GPU class: **NVIDIA Tesla T4**  
Real-time gate: **25 processing FPS**

| System | MOTA ↑ | HOTA ↑ | IDF1 ↑ | IDS ↓ | FN ↓ | FP ↓ | FPS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline_Default | 19.729% | 28.430% | 32.724% | 1235 | 155,051 | **16,308** | 36.53 |
| Old_ACMOT_Frozen | 23.236% | 32.698% | 39.516% | **1061** | 138,967 | 25,025 | **41.96** |
| **New_ACMOT_Frozen** | **26.948%** | **33.835%** | **41.546%** | 1184 | **136,858** | 19,029 | 38.98 |

### New AC-MOT vs baseline

- MOTA: **+7.2195 percentage points**
- HOTA: **+5.4051 percentage points**
- IDF1: **+8.8220 percentage points**
- IDS: **1235 → 1184** (`-51`)
- FN: **155051 → 136858** (`-18193`)
- FP: **16308 → 19029** (`+2721`)
- FPS: **36.53 → 38.98**

### New AC-MOT vs historical Old AC-MOT

The new system achieves higher MOTA, HOTA, and IDF1, and lower FN/FP than the old heuristic system. The historical Old AC-MOT configuration still has the **lowest absolute IDS** and the **highest FPS**. The release therefore does **not** claim that New AC-MOT wins every metric.

Canonical final artifacts are in [`paper_artifacts/final/`](paper_artifacts/final/).

---

## System overview

```text
Frame
  ↓
Scene analysis / SCI
  ↓
Frozen controller
  ↓
YOLOv8n detector
  ↓
ByteTrack association
  ↓
{id, box}
```

The final optimized system uses:

- Detector: fixed pretrained **YOLOv8n**
- Tracker: fixed tuned **ByteTrack**
- SCI smoothing window: **7**
- Analysis stride: **10**
- Resolution levels: **512 / 928 / 960**
- Validation-selected Trial: **24**
- Real-time constraint: **FPS ≥ 25**

### Frozen optimized SCI parameters

```text
crowd = 0.12949277455301997
tiny  = 0.22174766876599927
edge  = 0.43371337893805056
night = 0.05355765312756694
blur  = 0.16148852461536325

conf_easy = 0.30
conf_hard = 0.40

nms_easy = 0.35
nms_hard = 0.35

threshold_mid  = 0.13534938199219218
threshold_high = 0.28728676236279177
```

**Important:** the final optimized configuration does **not** implement adaptive NMS in practice because `nms_easy == nms_hard == 0.35`. The adaptive behavior retained by the frozen controller is therefore not described as “adaptive NMS” in the paper release.

The canonical frozen configuration is preserved in [`paper_artifacts/config/FROZEN_DEFENSIBLE_ACMOT_CONFIG.json`](paper_artifacts/config/FROZEN_DEFENSIBLE_ACMOT_CONFIG.json).

---

## Validation and model selection

Optimization was performed on `VisDrone2019-MOT-val`, not on test-dev.

The frozen selection rule was:

1. satisfy the real-time gate,
2. satisfy the IDS constraint relative to historical Old-A3,
3. maximize MOTA,
4. use lower IDS, then HOTA, IDF1, and FPS as tie-breakers.

The frozen candidate was **Optuna Trial 24**:

| Metric | Trial 24 validation |
|---|---:|
| MOTA | 23.0381% |
| HOTA | 36.1102% |
| IDF1 | 40.7578% |
| IDS | 270 |
| FPS | 37.1686 |

No held-out test result was used to select these parameters.

---

## New component ablation

This validation-only ablation was completed before the final held-out protocol was frozen.

| Variant | MOTA | HOTA | IDF1 | IDS | FPS |
|---|---:|---:|---:|---:|---:|
| A0 — Static anchor | **23.081%** | 36.078% | 40.600% | **266** | **37.92** |
| A1 — Adaptive confidence | 23.013% | 36.109% | 40.727% | 271 | 35.35 |
| A2 — Adaptive confidence + NMS stage | 23.013% | 36.109% | 40.727% | 271 | 35.53 |
| A3 — Full New AC-MOT | 23.038% | **36.110%** | **40.758%** | 270 | 35.45 |
| HIST — Old AC-MOT W7/S10 | 18.165% | 33.064% | 36.296% | 271 | 37.98 |

The ablation is intentionally reported without claiming that the adaptive configuration beats every static operating point on every metric. Raw values are preserved in [`paper_artifacts/validation/NEW_ACMOT_COMPONENT_ABLATION.csv`](paper_artifacts/validation/NEW_ACMOT_COMPONENT_ABLATION.csv).

---

## Historical Old AC-MOT ablation

| Stage | Description | MOTA | HOTA | IDF1 | IDS | FPS |
|---|---|---:|---:|---:|---:|---:|
| OLD-A0 | Static detector + default ByteTrack | 17.633% | 29.837% | 30.892% | 283 | 44.09 |
| OLD-A1 | Static detector + tuned ByteTrack | 17.802% | 30.864% | 32.854% | 217 | 44.51 |
| OLD-A2 | Adaptive conf/NMS, resolution fixed | 17.636% | 31.384% | 33.727% | **210** | 40.65 |
| OLD-A2R | Adaptive resolution only | **18.425%** | 32.446% | 35.669% | 241 | 39.56 |
| OLD-A3 | Full historical Old AC-MOT | 18.165% | **33.064%** | **36.296%** | 271 | 37.96 |

Raw values are preserved in [`paper_artifacts/validation/OLD_ACMOT_COMPONENT_ABLATION.csv`](paper_artifacts/validation/OLD_ACMOT_COMPONENT_ABLATION.csv).

---

## Frozen held-out protocol

Three systems were declared before held-out evaluation:

1. `Baseline_Default`
   - confidence `0.25`
   - NMS IoU `0.45`
   - image size `640`
   - default ByteTrack profile
2. `Old_ACMOT_Frozen`
   - historical heuristic SCI controller
   - tuned ByteTrack
   - smoothing `7`, stride `10`
   - resolutions `640 / 736 / 832`
3. `New_ACMOT_Frozen`
   - validation-selected empirical SCI controller
   - tuned ByteTrack
   - smoothing `7`, stride `10`
   - resolutions `512 / 928 / 960`

The protocol records:

```text
selection_or_tuning_on_test = false
official_visdrone = false
minimum_processing_fps = 25
required_gpu_class = NVIDIA T4
TrackEval commit = 12c8791b303e0a0b50f753af204249e622d0281a
```

Ground-truth filtering:

```text
categories = [1, 4, 5, 6, 9]
score = 1
occlusion < 2
truncation < 2
```

See [`paper_artifacts/final/FINAL_TEST_3WORKER_PROTOCOL.json`](paper_artifacts/final/FINAL_TEST_3WORKER_PROTOCOL.json).

---

## Reproducibility identity

Frozen research source:

```text
Repository: https://github.com/AhmedCode110/AC-MOT
Frozen branch: acmot-final-frozen-2026-09-11
Frozen commit: a6c1fa49fce1d402513c2df05b7d04b962a6e89e
Paper-release branch: paper-release-2026-09-11
```

Pinned evaluator:

```text
TrackEval commit:
12c8791b303e0a0b50f753af204249e622d0281a
```

Canonical prerequisite hashes recorded by the frozen protocol:

```text
FROZEN_DEFENSIBLE_ACMOT_CONFIG.json
8eeb7b916e7085b290313349cb0ebb95fa97f7f3956caefedd902fcf2890379c

DETECTOR_DERIVED_CUE_CALIBRATION.json
fd42b22987365236944e90d3ebd646a28a56e5b60986cf1ad9f05881e7958c78

NEW_ACMOT_COMPONENT_ABLATION_DONE.json
4757d76f3414e3d040d3fc16f2662c84cf8a814800a08b38e77c7f288bb09ad6

FINAL_TEST_3WORKER_PROTOCOL.json
f985b25651d622368448a21ac62712e1b9ae00849ca1012fdaaf3f511bb10720
```

The final held-out lock is [`paper_artifacts/final/FINAL_TEST_DONE.json`](paper_artifacts/final/FINAL_TEST_DONE.json). Once this file exists, this test-dev result is considered exposed and locked. Future tuning must use a new protocol/split and must not be described as the original unbiased final held-out evaluation.

---

## Repository layout

```text
AC-MOT/
├── core*.py, experiment*.py, evaluate.py
├── scripts/                     # experiment, validation and final-test code
├── notebooks/                   # Colab entry points and Optuna workflows
├── configs/                     # runtime/config templates
├── legacy/                      # preserved historical implementations
├── docs/                        # historical and protocol documentation
└── paper_artifacts/
    ├── config/                  # frozen publication configuration
    ├── validation/              # validation ablations and selection evidence
    └── final/                   # locked held-out protocol + results
```

The dataset, model weights, raw frame-level checkpoints, and large caches are not committed to GitHub. They must be obtained separately and are intentionally excluded from source control.

---

## Running / auditing the code

For source-level inspection:

```bash
git clone https://github.com/AhmedCode110/AC-MOT.git
cd AC-MOT
git checkout paper-release-2026-09-11
python -m pip install -r requirements.txt
python evaluate.py --help
```

For exact frozen-source inspection:

```bash
git checkout acmot-final-frozen-2026-09-11
git rev-parse HEAD
# expected:
# a6c1fa49fce1d402513c2df05b7d04b962a6e89e
```

The final-test scripts are under `scripts/`. Do **not** rerun or retune the original held-out protocol and then describe the new run as the same unbiased final evaluation.

---

## What is and is not claimed

This release supports the following statement:

> On the locked 17-sequence held-out evaluation under the custom class-agnostic AC-MOT protocol, New AC-MOT achieved the highest MOTA, HOTA, and IDF1 while remaining real-time. Historical Old AC-MOT achieved the lowest absolute IDS and highest FPS.

This release does **not** claim:

- official VisDrone leaderboard results,
- that New AC-MOT has the lowest IDS,
- that New AC-MOT improves every metric over every static operating point,
- adaptive NMS in the final optimized system,
- test-time tuning or selection.

---

## Citation

If you use this repository, cite the associated AC-MOT paper/thesis once its bibliographic record is available. A `CITATION.cff` file is included in this paper-release branch for repository citation metadata.

## Authors

Ahmed Gouda Ismail  
Computers Engineering and Artificial Intelligence Department  
Military Technical College, Cairo, Egypt

Research collaborators/supervision information can be added to the final paper citation without changing the frozen experimental artifacts.
