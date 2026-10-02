# V7 runtime — post-freeze, one fixed device

Command: `tools/v7/benchmark.py --weights yolov8n.pt --seqs 0001 0009 0019`
(raw numbers: `V7_REALTIME_yolov8n.json`). Frozen policy V7f.

| Item | Value |
|---|---|
| Device | cloud VM, Intel Xeon @ 2.10 GHz, 4 vCPU, 15 GB RAM, no GPU (torch 2.14.0 CPU, 4 threads) |
| Detector | YOLOv8n (sha256 f59b3d83…), fp32, 736 px, conf floor 0.01, NMS 0.7, batch 1 |
| Tracker | ultralytics ByteTrack (host defaults 0.25 / 0.25 / 0.1, match 0.8) |
| Data | KITTI tracking training 0001, 0009, 0019 (live frames), 2279 timed frames per arm (first 10 per sequence excluded); page cache pre-warmed for both arms |
| Mode | sequential, live detector, both arms on the same frames back to back |

| Stage (ms) | Baseline mean / P95 | + V7f mean / P95 |
|---|---|---|
| frame decode | 13.88 / 16.21 | 13.88 / 16.51 |
| detector | 41.91 / 58.41 | 42.17 / 59.43 |
| AC-MOT controller (motion cue + step + observe) | — | **2.95 / 4.39** |
| tracker | 2.04 / 3.57 | 1.43 / 2.64 (fewer candidates reach it) |
| detector + controller + tracker | 43.97 / 61.37 | 46.56 / 65.24 |
| end-to-end (incl. decode) | 57.85 / 76.35 | 60.44 / 80.78 |
| FPS end-to-end | 17.29 | 16.55 |

Overhead of AC-MOT: +2.59 ms per frame end-to-end (+4.5%), controller alone
2.95 ms mean / 4.39 ms P95 (it includes the phase-correlation motion cue on a
1/4-resolution frame). On this CPU the full pipeline is below the 30 FPS
reference in both arms (the detector dominates); no FPS is converted to other
devices. A first run without page-cache warming gave a biased decode time for
the first arm (89.5 ms vs 13.9 ms) and is superseded by the table above; the
controller cost was the same (2.97 ms).
