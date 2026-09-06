# AC-MOT v14 — Realtime Compute Fix + Stage Profiler

## Status

Historical verified version. The v14 implementation is now frozen. v15 was created from the evidence recorded below.

## Previous verified output: v13

Tesla T4, 300-frame gate, target 25 FPS:

| System | FPS | p95 latency | Status |
|---|---:|---:|---|
| LIVE_ADAPTIVE_NO_STABILITY | 8.9643 | 178.16 ms | FAIL_REALTIME |
| FIXED_736 | 10.0772 | 152.13 ms | FAIL_REALTIME |
| FIXED_832 | 9.7389 | 160.56 ms | FAIL_REALTIME |

v13 already used FP16 and local Colab SSD. Therefore Drive I/O and FP32 alone did not explain the remaining slowdown.

## What v14 changed

### 1. Semantic detector confidence pruning

YOLO received the Controller's exact final keep threshold for the current frame instead of always starting at `conf=0.01`. This removed detector candidates that could never survive into ByteTrack.

### 2. TensorRT FP16 preferred backend

v14 tried TensorRT FP16 first and explicitly fell back to PyTorch FP16 if TensorRT was unavailable.

### 3. Realtime resolution governor

`LIVE_ADAPTIVE_RT_V14` kept the SCI/Controller requested resolution but allowed a deployment-only maximum resolution cap. The fixed 736 and 832 systems remained fixed references.

### 4. Per-stage profiler

Every measured frame recorded local JPEG read/decode, SceneAnalyzer, Controller, detector, ByteTrack, and total latency.

## Verified v14 output

Run folder:

`/content/drive/MyDrive/VisDrone_Results/ACMOT_IDS/speedtests/speedtest_v14_20260906T224714_d9463022`

Tesla T4, 300 measured frames per system, target 25 FPS:

| System | FPS | p95 total | Backend | Final cap | Status |
|---|---:|---:|---|---:|---|
| LIVE_ADAPTIVE_RT_V14 | 13.5702 | 108.01 ms | PyTorch FP16 | 320 | FAIL_REALTIME |
| FIXED_736_REFERENCE | 11.8136 | 127.50 ms | PyTorch FP16 | 736 | FAIL_REALTIME |
| FIXED_832_REFERENCE | 11.2828 | 137.66 ms | PyTorch FP16 | 832 | FAIL_REALTIME |

The TensorRT preference did not become the active backend in this verified run; the saved runtime field reports `pytorch_fp16`.

## Authoritative v14 bottleneck profile

### LIVE_ADAPTIVE_RT_V14

| Stage | Mean | p95 |
|---|---:|---:|
| local JPEG read/decode | 42.69 ms | 75.44 ms |
| SceneAnalyzer | 0.25 ms | 2.27 ms |
| Controller | 0.07 ms | 0.19 ms |
| detector | 27.32 ms | 42.38 ms |
| ByteTrack | 3.28 ms | 5.57 ms |
| total | 73.69 ms | 108.01 ms |

The key finding is that JPEG file read/decode alone averaged 42.69 ms. That is already slower than the complete 40 ms frame budget required for 25 FPS, before detector/tracker processing is added.

The actual AC-MOT processing stages after a decoded frame is available averaged approximately 30.9 ms for the adaptive system (SceneAnalyzer + Controller + detector + ByteTrack), corresponding to roughly 32 FPS processing capacity. This is an inference from the measured stage profile; v14 itself still used the stricter file-playback gate and therefore correctly remained FAIL_REALTIME.

## Why v15 was created

v15 separates two different questions that v14 mixed into one number:

1. **Dataset playback FPS** — includes JPEG file read/decode from the VisDrone image sequence.
2. **Live-frame processing FPS** — starts from an already decoded image buffer, matching the interface a live camera/capture pipeline normally provides to AC-MOT.

v15 must report both numbers. JPEG decode is never hidden; it is measured separately. The 25 FPS realtime processing gate is applied only to the documented decoded/live-frame processing path.

v15 also adds portable Colab account setup so an authorized account can locate its own accessible VisDrone copy, choose its own Drive results root, validate GitHub access, and run without Ahmed-specific absolute Drive paths.

## Research integrity

v14 remains a historical FAIL_REALTIME result under its documented timing protocol. v15 does not rewrite that result. Any v15 realtime claim must come from a new saved 300-frame measurement and must state whether it refers to live-frame processing or serialized dataset playback.
