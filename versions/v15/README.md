# AC-MOT v15 — Portable Colab + Live-Frame Realtime Gate

## Status

Current candidate version. v12, v13 and v14 remain historical versions and are not redefined by v15.

v15 is **not yet labeled realtime**. `PASS_REALTIME_PROCESSING` requires a new saved run at or above 25 FPS under the v15 timing definition below.

## Previous verified output: v14

Verified Tesla T4 run:

`/content/drive/MyDrive/VisDrone_Results/ACMOT_IDS/speedtests/speedtest_v14_20260906T224714_d9463022`

| System | v14 FPS | p95 total | Backend | Final cap | Status |
|---|---:|---:|---|---:|---|
| LIVE_ADAPTIVE_RT_V14 | 13.5702 | 108.01 ms | PyTorch FP16 | 320 | FAIL_REALTIME |
| FIXED_736_REFERENCE | 11.8136 | 127.50 ms | PyTorch FP16 | 736 | FAIL_REALTIME |
| FIXED_832_REFERENCE | 11.2828 | 137.66 ms | PyTorch FP16 | 832 | FAIL_REALTIME |

### v14 adaptive stage profile

| Stage | Mean | p95 |
|---|---:|---:|
| local JPEG read/decode | 42.69 ms | 75.44 ms |
| SceneAnalyzer | 0.25 ms | 2.27 ms |
| Controller | 0.07 ms | 0.19 ms |
| detector | 27.32 ms | 42.38 ms |
| ByteTrack | 3.28 ms | 5.57 ms |
| total | 73.69 ms | 108.01 ms |

The v14 profiler proved that local JPEG file read/decode alone averaged 42.69 ms, already exceeding the complete 40 ms frame budget required for 25 FPS.

The AC-MOT processing stages after a decoded frame was available averaged about 30.9 ms for the adaptive system. This suggested roughly 32 FPS processing capacity, but v14 correctly remained `FAIL_REALTIME` because its gate included JPEG file decode.

## Problem v15 solves

v14 combined two different questions into one FPS number:

1. How fast can AC-MOT process a frame that a live capture pipeline has already delivered as an image buffer?
2. How fast can Colab serially open/decode VisDrone JPEG files and then run AC-MOT?

Those are both useful measurements, but they are not the same workload.

v15 separates them and reports both.

## v15 realtime definition

### Processing FPS — realtime gate

The 25 FPS gate measures only:

- SceneAnalyzer
- Controller
- exactly one fresh YOLOv8n inference per frame
- ByteTrack

Input to the measured section is an already decoded image buffer.

This matches the AC-MOT processing interface expected after a camera/capture stage has produced a frame.

### Dataset playback FPS — reported separately

v15 also measures JPEG read/decode time for every VisDrone frame and reports:

- JPEG decode mean/p95
- serialized dataset playback FPS = JPEG decode + AC-MOT processing

JPEG decode is therefore **not hidden**. It is simply not mixed into the decoded/live-frame processing gate.

Drive-to-local copy, model/backend initialization, calibration/warmup and output serialization remain excluded and are documented separately.

## Realtime integrity rules preserved

- exactly one fresh YOLO inference per measured frame
- no detection cache used as deployment FPS
- no frame skipping
- no fake duplicated frames
- ByteTrack state remains causal and sequential
- original current SCI/Controller behavior remains
- tracker thresholds remain unchanged
- semantic detector confidence pruning from v14 remains
- adaptive runtime resolution governor remains only on the adaptive system
- fixed 736 and fixed 832 remain fixed references

## Backend choice

v15 defaults to **PyTorch FP16 on Tesla T4**.

Reason: v14 attempted TensorRT but the verified run fell back to PyTorch FP16. Making TensorRT mandatory would reduce portability without a verified benefit in the current environment.

TensorRT can be revisited in a later numbered version if required.

## Portable Colab design

v15 removes the Ahmed-specific dataset/output path requirement.

The repository config uses:

- `dataset: AUTO`
- `output_root: AUTO`

At runtime `portable_v15.py` resolves account-specific paths.

### Dataset discovery order

v15 checks:

1. `ACMOT_DATASET` environment variable, if supplied
2. common VisDrone paths in the current account's MyDrive
3. current MyDrive folder tree
4. accessible Shared drives
5. accessible Google Drive shortcuts

A candidate is accepted only if it contains both `sequences/` and `annotations/` and matches the configured 17 sequences / 6635 frames.

### Results destination

For `output_root: AUTO`, results go to the current account's:

`MyDrive/AC-MOT-results/v15/speedtests`

Each invocation still creates a unique versioned run envelope.

## Requirements for another Colab account

The account must have:

1. Google Drive access to a valid VisDrone2019-MOT-test-dev copy.
2. A Colab Secret named `GITHUB_TOKEN`.
3. The token must belong to a GitHub account authorized to read the private `AhmedCode110/AC-MOT` repository.
4. Notebook access enabled for that secret.
5. A Tesla T4 runtime for the pinned realtime test.

An unauthorized or incomplete account fails during the visible preflight stages with a specific message instead of silently using another path/runtime.

## Notebook behavior

The main Colab notebook now shows:

- current stage
- what is running now
- why the step is needed
- elapsed time
- remaining stages
- resolved dataset path
- resolved output path
- exact active version and runner
- processing FPS
- dataset playback FPS
- per-stage latency
- sequence/frame progress
- ETA when measurable

## Expected result before verification

Based only on the verified v14 stage profile, the adaptive processing path may exceed 25 FPS once JPEG decode is separated from processing timing. This is an expectation, not a result.

The fixed 736 and 832 references may still remain below 25 processing FPS because their v14 detector/tracker compute was already heavier.

The authoritative answer will be the new v15 `speedtest_results.json` after the 300-frame gate.

## Files introduced/changed for v15

- `portable_v15.py`
- `scripts/speedtest_top3_v15.py`
- `configs/speedtest_top3_v15.json`
- `scripts/run.py` — portable resolution and v15 argument routing only
- `notebooks/AC_MOT_Colab.ipynb` — portable v15 workflow and explanations
- `versions/v15/README.md`

## Why a later version would be required

Any new algorithm, timing, backend, threshold, dataset, portability or deployment change after this v15 definition must become v16. v15 itself must not be silently repurposed after its verified result is recorded.
