# SCI + V7f / G1 — runtime

**Device: GitHub-hosted CPU runner (AMD EPYC 7763, torch 2.14.0 CPU, 2 threads), not a T4.** No GPU was
reachable from this environment. These numbers are diagnostic; the T4 measurement of the frozen G1 is
prepared in `notebooks/G1_T4_runtime.ipynb` (CUDA-synchronised stages, same script).

Script: `tools/sci_v7/benchmark.py` (GitHub Actions run 37009596192). Real VisDrone val frames
(uav0000086_00000_v and uav0000339_00001_v, first 160 frames each, first 10 per sequence excluded),
live detector, batch 1, fp32, sequential; frame decoding timed separately and excluded from the totals.
Raw files: `research/final/sci_v7f/G1_runtime/`.

| detector | host alone total (ms) | G1 total (ms) | V7f layer (ms) | G1 overhead | P95 G1 / host alone (ms) |
|---|---|---|---|---|---|
| YOLOv8n | 86.7 | 92.8 | 5.23 | +7.1% | 96.1 / 90.2 |
| RT-DETR-L | 1204.2 | 1224.1 | 7.38 | +1.7% | 1293.8 / 1273.6 |
| RetinaNet | 941.7 | 952.3 | 13.14 | +1.1% | 971.1 / 950.1 |

The V7f time includes its phase-correlation motion cue on a quarter-resolution frame. The tracker is
faster with V7f (fewer candidates reach it), e.g. 3.27 → 2.76 ms with YOLOv8n. The historical SCI arm
adds 3.1 ms of image analysis (edges, gray level, blur) and 0.02 ms of decision per frame.

Detector latency per input setting on one host (ms; `latency_curve_*.json`, 50 timed frames):

| detector | 512 | 576 | 640 | 704 | 736 | 768 | 832 | 896 | 960 |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | 51.7 | 66.1 | 77.8 | 83.8 | 90.6 | 98.3 | 109.8 | 117.3 | 125.3 |
| RT-DETR-L | 649.1 | 742.1 | 906.8 | 1074.1 | 1147.8 | 1258.5 | 1476.8 | 1652.6 | 1882.4 |
| RetinaNet | 467.0 | 630.3 | 762.9 | 890.9 | 930.8 | 1038.0 | 1213.7 | 1397.9 | 1563.5 |

On this CPU none of the pipelines reaches a video frame rate; no FPS is converted to other devices.
