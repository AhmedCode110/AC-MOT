# Held-out VisDrone2019-MOT test-dev — Universal AC-MOT V4 (evaluated once)

Lock: research/TESTDEV_LOCK_V4.json · code commit a55b138 (policy files hash-verified) · 17 sequences, 6635 frames.
Internal class-agnostic protocol (NOT official VisDrone). HOTA TrackEval; others motmetrics. ncat = sequences with MOTA < 0.

| Tracker | System | Detector | MOTA | HOTA | IDF1 | IDS | FP | FN | Prec | Rec | ncat |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bytetrack | default | yolov8 | 20.27 | 30.00 | 34.97 | 1110 | 21593 | 148738 | 75.4 | 30.8 | 0 |
| bytetrack | default | rtdetr | 8.78 | 37.09 | 42.89 | 2223 | 99793 | 94117 | 54.8 | 56.2 | 6 |
| bytetrack | shared_static_736 | yolov8 | 21.35 | 29.80 | 34.41 | 464 | 10660 | 157988 | 84.3 | 26.5 | 0 |
| bytetrack | shared_static_736 | rtdetr | 25.67 | 38.82 | 46.58 | 983 | 42164 | 116667 | 70.0 | 45.7 | 0 |
| bytetrack | shared_static_832 | yolov8 | 24.04 | 32.02 | 37.77 | 572 | 13414 | 149349 | 83.0 | 30.5 | 0 |
| bytetrack | shared_static_832 | rtdetr | 26.26 | 39.60 | 47.38 | 1009 | 45320 | 112233 | 69.4 | 47.8 | 1 |
| bytetrack | oracle | yolov8 | 25.81 | 33.94 | 40.69 | 868 | 20757 | 137893 | 78.8 | 35.9 | 0 |
| bytetrack | oracle | rtdetr | 26.26 | 39.60 | 47.38 | 1009 | 45320 | 112233 | 69.4 | 47.8 | 1 |
| bytetrack | V1 | yolov8 | 3.71 | 33.44 | 39.91 | 1862 | 91093 | 114089 | 52.6 | 46.9 | 7 |
| bytetrack | V1 | rtdetr | -60.21 | 31.96 | 34.47 | 4415 | 265358 | 74711 | 34.6 | 65.3 | 12 |
| bytetrack | V2b | yolov8 | 13.99 | 25.97 | 27.65 | 313 | 12844 | 171773 | 77.1 | 20.1 | 0 |
| bytetrack | V2b | rtdetr | 14.56 | 26.75 | 25.83 | 118 | 4386 | 179199 | 89.1 | 16.7 | 0 |
| bytetrack | V3 | yolov8 | 23.62 | 33.12 | 40.27 | 682 | 21776 | 141772 | 77.1 | 34.1 | 0 |
| bytetrack | V3 | rtdetr | 23.00 | 34.96 | 40.08 | 329 | 16868 | 148373 | 79.8 | 31.0 | 0 |
| bytetrack | V4_640 | yolov8 | 21.01 | 31.59 | 37.72 | 690 | 21794 | 147352 | 75.6 | 31.5 | 0 |
| bytetrack | V4_640 | rtdetr | 22.50 | 35.18 | 40.38 | 369 | 19540 | 146727 | 77.8 | 31.8 | 0 |
| bytetrack | V4_736 | yolov8 | 23.89 | 33.72 | 40.57 | 749 | 22949 | 139955 | 76.6 | 34.9 | 0 |
| bytetrack | V4_736 | rtdetr | 23.60 | 36.33 | 42.56 | 386 | 22440 | 141441 | 76.6 | 34.2 | 0 |
| bytetrack | V4_832 | yolov8 | 26.19 | 35.75 | 43.60 | 844 | 26186 | 131674 | 76.1 | 38.8 | 0 |
| bytetrack | V4_832 | rtdetr | 25.31 | 37.33 | 43.71 | 428 | 22202 | 137956 | 77.6 | 35.8 | 0 |
| bytetrack | V4_736_nogate | yolov8 | 5.51 | 33.82 | 40.03 | 2321 | 89126 | 111725 | 53.7 | 48.0 | 7 |
| bytetrack | V4_736_nogate | rtdetr | -55.09 | 31.98 | 34.57 | 4817 | 255243 | 73413 | 35.7 | 65.9 | 12 |
| botsort | default | yolov8 | 21.69 | 33.88 | 40.11 | 610 | 24932 | 142843 | 74.3 | 33.6 | 0 |
| botsort | default | rtdetr | 2.76 | 41.09 | 48.69 | 1276 | 122748 | 85065 | 51.4 | 60.4 | 6 |
| botsort | shared_static_832 | yolov8 | 25.52 | 35.80 | 43.17 | 238 | 15145 | 144759 | 82.3 | 32.7 | 0 |
| botsort | shared_static_832 | rtdetr | 25.80 | 43.89 | 53.45 | 343 | 53597 | 105605 | 67.1 | 50.9 | 2 |
| botsort | V4_736 | yolov8 | 25.11 | 37.31 | 45.83 | 353 | 25649 | 135030 | 75.7 | 37.2 | 0 |
| botsort | V4_736 | rtdetr | 24.63 | 40.25 | 48.21 | 171 | 29862 | 132032 | 73.5 | 38.6 | 0 |
| botsort | V4_832 | yolov8 | 27.16 | 39.21 | 48.67 | 366 | 29131 | 127114 | 75.1 | 40.9 | 0 |
| botsort | V4_832 | rtdetr | 24.47 | 41.26 | 49.16 | 187 | 33837 | 128369 | 71.9 | 40.3 | 1 |

## Paired sequence bootstrap (10,000 resamples, seed 42, 95% CI)

| Tracker | Detector | System | Baseline | Metric | Δ | 95% CI | P(Δ≤0) |
|---|---|---|---|---|---:|---|---:|
| bytetrack | yolov8 | V4_736 | default | HOTA | +3.72 | [+2.61, +4.96] | 0.000 |
| bytetrack | yolov8 | V4_736 | default | IDF1 | +5.60 | [+3.70, +7.67] | 0.000 |
| bytetrack | yolov8 | V4_736 | default | MOTA | +3.62 | [+2.07, +5.57] | 0.000 |
| bytetrack | rtdetr | V4_736 | default | HOTA | -0.75 | [-2.88, +1.61] | 0.752 |
| bytetrack | rtdetr | V4_736 | default | IDF1 | -0.32 | [-4.09, +4.04] | 0.575 |
| bytetrack | rtdetr | V4_736 | default | MOTA | +14.82 | [+1.62, +31.16] | 0.011 |
| bytetrack | yolov8 | V4_736 | V1 | HOTA | +0.28 | [-1.18, +2.25] | 0.381 |
| bytetrack | yolov8 | V4_736 | V1 | IDF1 | +0.66 | [-1.95, +4.01] | 0.334 |
| bytetrack | yolov8 | V4_736 | V1 | MOTA | +20.18 | [+8.01, +36.79] | 0.000 |
| bytetrack | rtdetr | V4_736 | V1 | HOTA | +4.37 | [+0.84, +8.22] | 0.006 |
| bytetrack | rtdetr | V4_736 | V1 | IDF1 | +8.10 | [+1.80, +14.92] | 0.004 |
| bytetrack | rtdetr | V4_736 | V1 | MOTA | +83.82 | [+42.89, +142.45] | 0.000 |
| bytetrack | yolov8 | V4_736 | V2b | HOTA | +7.75 | [+4.96, +10.59] | 0.000 |
| bytetrack | yolov8 | V4_736 | V2b | IDF1 | +12.92 | [+7.64, +17.94] | 0.000 |
| bytetrack | yolov8 | V4_736 | V2b | MOTA | +9.90 | [+6.68, +12.79] | 0.000 |
| bytetrack | rtdetr | V4_736 | V2b | HOTA | +9.58 | [+5.90, +13.60] | 0.000 |
| bytetrack | rtdetr | V4_736 | V2b | IDF1 | +16.73 | [+10.78, +23.26] | 0.000 |
| bytetrack | rtdetr | V4_736 | V2b | MOTA | +9.04 | [+3.39, +13.76] | 0.001 |
| bytetrack | yolov8 | V4_736 | V3 | HOTA | +0.61 | [+0.15, +1.04] | 0.004 |
| bytetrack | yolov8 | V4_736 | V3 | IDF1 | +0.30 | [-0.44, +0.95] | 0.214 |
| bytetrack | yolov8 | V4_736 | V3 | MOTA | +0.27 | [-0.60, +1.03] | 0.258 |
| bytetrack | rtdetr | V4_736 | V3 | HOTA | +1.37 | [-0.15, +3.26] | 0.059 |
| bytetrack | rtdetr | V4_736 | V3 | IDF1 | +2.49 | [-0.01, +5.93] | 0.027 |
| bytetrack | rtdetr | V4_736 | V3 | MOTA | +0.61 | [-1.99, +3.35] | 0.377 |
| bytetrack | yolov8 | V4_736 | shared_static_736 | HOTA | +3.92 | [+2.84, +5.16] | 0.000 |
| bytetrack | yolov8 | V4_736 | shared_static_736 | IDF1 | +6.16 | [+4.35, +8.19] | 0.000 |
| bytetrack | yolov8 | V4_736 | shared_static_736 | MOTA | +2.54 | [+0.23, +4.84] | 0.015 |
| bytetrack | rtdetr | V4_736 | shared_static_736 | HOTA | -2.49 | [-4.39, -0.76] | 0.999 |
| bytetrack | rtdetr | V4_736 | shared_static_736 | IDF1 | -4.02 | [-7.08, -1.22] | 1.000 |
| bytetrack | rtdetr | V4_736 | shared_static_736 | MOTA | -2.07 | [-6.28, +1.77] | 0.854 |
| bytetrack | yolov8 | V4_832 | shared_static_832 | HOTA | +3.73 | [+2.48, +4.99] | 0.000 |
| bytetrack | yolov8 | V4_832 | shared_static_832 | IDF1 | +5.84 | [+3.79, +7.85] | 0.000 |
| bytetrack | yolov8 | V4_832 | shared_static_832 | MOTA | +2.15 | [-0.84, +4.70] | 0.074 |
| bytetrack | rtdetr | V4_832 | shared_static_832 | HOTA | -2.28 | [-3.66, -1.06] | 1.000 |
| bytetrack | rtdetr | V4_832 | shared_static_832 | IDF1 | -3.67 | [-5.79, -1.68] | 1.000 |
| bytetrack | rtdetr | V4_832 | shared_static_832 | MOTA | -0.94 | [-5.47, +3.99] | 0.666 |
| bytetrack | yolov8 | V4_832 | oracle | HOTA | +1.81 | [+0.82, +2.71] | 0.000 |
| bytetrack | yolov8 | V4_832 | oracle | IDF1 | +2.92 | [+1.17, +4.47] | 0.000 |
| bytetrack | yolov8 | V4_832 | oracle | MOTA | +0.38 | [-1.36, +1.97] | 0.318 |
| bytetrack | rtdetr | V4_832 | oracle | HOTA | -2.28 | [-3.66, -1.06] | 1.000 |
| bytetrack | rtdetr | V4_832 | oracle | IDF1 | -3.67 | [-5.79, -1.68] | 1.000 |
| bytetrack | rtdetr | V4_832 | oracle | MOTA | -0.94 | [-5.47, +3.99] | 0.666 |
| bytetrack | yolov8 | V4_736 | V4_736_nogate | HOTA | -0.10 | [-1.66, +1.83] | 0.550 |
| bytetrack | yolov8 | V4_736 | V4_736_nogate | IDF1 | +0.54 | [-2.18, +3.70] | 0.353 |
| bytetrack | yolov8 | V4_736 | V4_736_nogate | MOTA | +18.38 | [+8.33, +31.07] | 0.000 |
| bytetrack | rtdetr | V4_736 | V4_736_nogate | HOTA | +4.35 | [+0.77, +8.24] | 0.007 |
| bytetrack | rtdetr | V4_736 | V4_736_nogate | IDF1 | +7.99 | [+1.78, +14.74] | 0.005 |
| bytetrack | rtdetr | V4_736 | V4_736_nogate | MOTA | +78.70 | [+40.63, +133.00] | 0.000 |
| botsort | yolov8 | V4_736 | default | HOTA | +3.44 | [+2.04, +5.03] | 0.000 |
| botsort | yolov8 | V4_736 | default | IDF1 | +5.73 | [+3.27, +8.27] | 0.000 |
| botsort | yolov8 | V4_736 | default | MOTA | +3.42 | [+1.67, +5.47] | 0.000 |
| botsort | rtdetr | V4_736 | default | HOTA | -0.84 | [-3.66, +2.18] | 0.718 |
| botsort | rtdetr | V4_736 | default | IDF1 | -0.48 | [-5.46, +4.97] | 0.582 |
| botsort | rtdetr | V4_736 | default | MOTA | +21.87 | [+5.60, +42.11] | 0.002 |
| botsort | yolov8 | V4_832 | shared_static_832 | HOTA | +3.41 | [+2.14, +4.76] | 0.000 |
| botsort | yolov8 | V4_832 | shared_static_832 | IDF1 | +5.50 | [+3.31, +7.74] | 0.000 |
| botsort | yolov8 | V4_832 | shared_static_832 | MOTA | +1.64 | [-1.72, +4.39] | 0.151 |
| botsort | rtdetr | V4_832 | shared_static_832 | HOTA | -2.63 | [-4.77, -0.54] | 0.995 |
| botsort | rtdetr | V4_832 | shared_static_832 | IDF1 | -4.29 | [-7.74, -0.88] | 0.996 |
| botsort | rtdetr | V4_832 | shared_static_832 | MOTA | -1.32 | [-5.87, +3.73] | 0.721 |

## Reading
- YOLOv8n: V4 is significantly better than the default, V2b, V3 (HOTA), the shared static operating point at matched compute, and even the detector-specific oracle (val-selected) at 832.
- RT-DETR-L: V4 removes the default's collapse (6→0 sequences with MOTA<0; MOTA +14.8) but is significantly below the shared static raw threshold 0.5 on HOTA (−2.5) and IDF1 (−4.0). H2 is rejected for RT-DETR. Diagnosis (no change made): V4 is recall-limited on RT-DETR (Rec 34% vs 46%); τ and s sit at the catastrophe-constraint edge set by clutter-heavy sequences.
- The gate is the essential component: without it V4 collapses like V1 (RT-DETR 12 catastrophic sequences).
- The shared static threshold's advantage relies on these two detectors' raw score scale; under a ×0.5 score rescaling it produces no tracks at all (E20), whereas V4 is exactly invariant to Platt/temperature recalibration (E23/E30).
