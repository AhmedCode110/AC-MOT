# AC-MOT v13 — Local SSD + FP16 Realtime Attempt

## Status

Historical/immutable once archived. v13 is the first deployment-style speed attempt after the strict v12 throughput gate.

## Starting point: v12

Verified v12 Tesla T4 throughput at the 300-frame gate:

| System | FPS | Status |
|---|---:|---|
| LIVE_ADAPTIVE_NO_STABILITY | 8.48 | FAIL_REALTIME |
| FIXED_736 | 9.11 | FAIL_REALTIME |
| FIXED_832 | 9.22 | FAIL_REALTIME |

v12 used FP32 and read frames directly from Google Drive.

## What changed in v13

The research architecture and tracker/controller thresholds were intentionally preserved.

Deployment execution changes only:

- copy each VisDrone sequence from Google Drive to Colab local `/content` storage before timing
- run YOLOv8n in FP16 on Tesla T4
- keep exactly one fresh YOLO inference per measured frame
- keep the 25 FPS target and 300-frame gate
- keep SceneAnalyzer, Controller and ByteTrack inside measured time
- exclude Drive-to-local copy, detector warmup and output serialization

## Verified v13 output

Latest verified v13 run on Tesla T4:

| System | FPS | p95 latency | Status |
|---|---:|---:|---|
| LIVE_ADAPTIVE_NO_STABILITY | 8.9643 | 178.16 ms | FAIL_REALTIME |
| FIXED_736 | 10.0772 | 152.13 ms | FAIL_REALTIME |
| FIXED_832 | 9.7389 | 160.56 ms | FAIL_REALTIME |

The result artifact explicitly records FP16 and Colab local SSD frame source.

## Problem discovered

Moving to local SSD and FP16 did not recover realtime speed. Therefore the remaining bottleneck is in the compute/post-processing path rather than Google Drive I/O or FP32 alone.

A key v13 inefficiency is that detector inference uses a hard-coded model-level confidence threshold of `0.01` and `max_det=1000`, while the controller later discards detections below its final per-frame keep threshold (for recovery configurations this is `0.04`). On dense VisDrone frames this can force NMS/post-processing to process many detections that can never survive into ByteTrack.

v13 also has no per-stage latency breakdown, so it cannot distinguish detector, analysis, controller, tracker and frame-read costs precisely.

## Why v14 was created

v14 is the next version and does not overwrite v13. It adds:

1. per-stage latency profiling
2. detector confidence pruning using the exact Controller keep threshold for that frame
3. TensorRT FP16 as the preferred Tesla T4 backend, with explicit PyTorch FP16 fallback
4. an FPS-aware deployment resolution cap for the adaptive system
5. visible progress, elapsed time and ETA for copy, calibration, warmup and measured processing

v14 still preserves exactly one fresh detector inference per measured frame.

## Archived files

This folder stores the v13 runner/config/notebook snapshots used before v14.
