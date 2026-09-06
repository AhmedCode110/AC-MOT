# AC-MOT v12 — Verified Strict Throughput Gate Baseline

## Status

Historical/immutable baseline for comparison with v13 and later versions.

## Purpose

v12 measured the selected Top-3 systems using a strict synchronized end-to-end timing protocol.

## Verified output

Top-3 speed-test result on Tesla T4, 300 measured frames per system, target 25 FPS:

| System | FPS | p95 latency | Status |
|---|---:|---:|---|
| LIVE_ADAPTIVE_NO_STABILITY | 8.4812 | 175.98 ms | FAIL_REALTIME |
| FIXED_736 | 9.1112 | 161.02 ms | FAIL_REALTIME |
| FIXED_832 | 9.2194 | 170.39 ms | FAIL_REALTIME |

The run stopped each system at the 300-frame realtime gate because all were below 25 FPS.

## v12 timing protocol

Included in measured time:

- frame read directly from Google Drive
- SceneAnalyzer
- Controller
- one fresh YOLOv8n inference per frame
- ByteTrack
- CUDA synchronization around each measured frame

Excluded:

- detector warmup
- output serialization

Detector precision: FP32.

## Problem discovered

The measured 8–9 FPS was not directly comparable to the older v10 realtime work because the execution protocol differed materially.

The original v10 notebook used:

- FP16 on the T4
- sequence frames copied to Colab local `/content` storage before timing

Therefore v12 mixed AC-MOT compute cost with slower Google Drive I/O and used FP32, while the historical realtime implementation used deployment-oriented FP16/local storage.

## What v12 intentionally preserves

- current AC-MOT SceneAnalyzer and SCI behavior
- current Controller behavior
- current ByteTrack parameters
- Top-3 selected systems
- exactly one fresh YOLO inference per measured frame
- 25 FPS realtime target
- 300-frame realtime gate

## Why v13 was created

v13 keeps the research architecture unchanged but changes the realtime execution environment to local-SSD frame reads and YOLOv8n FP16, then adds transparent progress/elapsed/ETA reporting so every long-running stage is visible.

## Evidence

Verified v12 result artifact:
`/content/drive/MyDrive/VisDrone_Results/ACMOT_IDS/speedtests/.../result/speedtest_results.json`

Key pre-v13 result values are recorded above so future versions always have a stable comparison point.
