# AC-MOT v14 — Realtime Compute Fix + Stage Profiler

## Status

Current active development version. v12 and v13 remain archived and unchanged.

## Previous verified output: v13

Tesla T4, 300-frame gate, target 25 FPS:

| System | FPS | p95 latency | Status |
|---|---:|---:|---|
| LIVE_ADAPTIVE_NO_STABILITY | 8.9643 | 178.16 ms | FAIL_REALTIME |
| FIXED_736 | 10.0772 | 152.13 ms | FAIL_REALTIME |
| FIXED_832 | 9.7389 | 160.56 ms | FAIL_REALTIME |

v13 already used FP16 and local Colab SSD. Therefore Drive I/O and FP32 alone did not explain the remaining slowdown.

## v13 problem

The v13 detector called YOLO with `conf=0.01` and `max_det=1000` on every frame. The Controller then used its final per-frame confidence threshold and `track()` discarded boxes below that threshold. With recovery enabled, that final threshold is `0.04`.

This means v13 could spend substantial NMS/post-processing time on low-score candidates that were guaranteed to be deleted before ByteTrack. Dense VisDrone frames amplify this cost.

v13 also reported only total frame latency, so it could not prove whether the remaining cost was frame read, SceneAnalyzer, Controller, detector or ByteTrack.

## What v14 changes

### 1. Semantic detector confidence pruning

YOLO now receives the Controller's exact final keep threshold for the current frame. Detections below that threshold were already discarded immediately in v13, so this removes wasted pre-NMS/post-processing work without intentionally removing any detection that could survive into the tracker.

### 2. TensorRT FP16 preferred backend

On Tesla T4, v14 first tries a cached dynamic TensorRT FP16 YOLOv8n engine. If no engine exists, it builds one before measured timing. Backend build/export is excluded from deployment FPS.

If TensorRT is unavailable, the reason is printed and v14 explicitly falls back to PyTorch FP16 instead of silently changing behavior.

### 3. Realtime resolution governor for the adaptive system

`LIVE_ADAPTIVE_RT_V14` keeps the original SCI/Controller requested resolution but adds a deployment-only maximum resolution cap.

Before measured timing, v14 benchmarks detector latency at multiple sizes and chooses the highest size that fits a conservative detector share of the 40 ms frame budget required for 25 FPS.

During the first 300 measured frames, the cap can only move downward if recent FPS remains below the safety target. One fresh YOLO inference is still executed for every measured frame.

The fixed 736 and 832 systems remain fixed reference systems; their governor is disabled so they are not mislabeled as fixed while secretly changing resolution.

### 4. Per-stage profiler

Every measured frame records:

- local frame read ms
- SceneAnalyzer ms
- Controller ms
- detector ms
- ByteTrack ms
- total frame ms

The final JSON stores mean and p95 values for each stage.

### 5. Progress and ETA

Long stages now print visible progress with:

- current stage
- percent complete
- elapsed time
- ETA when measurable
- sequence/frame
- requested and actually used resolution
- SCI/scene
- cumulative and recent FPS
- per-frame stage latency

No fake ETA is produced for stages where total work is unknown.

## Realtime definition

Target: 25 FPS.

Gate: 300 measured frames.

Measured time includes:

- local frame read
- SceneAnalyzer
- Controller
- exactly one fresh YOLO inference per frame
- ByteTrack

Excluded:

- Drive-to-local sequence copy
- TensorRT engine export/build
- calibration/warmup
- output serialization

## Verification rule

v14 is not considered realtime until the new Colab run records `PASS_REALTIME` at or above 25 FPS in `speedtest_results.json`.

If v14 still fails, its `stage_profile` is the authoritative evidence for creating v15. v14 itself must then remain unchanged.
