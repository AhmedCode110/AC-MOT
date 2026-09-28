# confirmation-16 (ONE-WAY, post-freeze) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conf16 | internal | yolov8 | V4 (frozen, VisDrone-tuned) | 0 | 20.55 | 34.07 | 38.06 | 295 | 14419 | 155662 | 80.30 | 27.41 |
| conf16 | internal | rtdetr | V4 (frozen, VisDrone-tuned) | 3 | 24.15 | 39.40 | 44.37 | 222 | 22458 | 139981 | 76.83 | 34.73 |
| conf16 | internal | yolov8 | Shared static (raw 0.5) | 0 | 17.37 | 29.90 | 30.24 | 190 | 5744 | 171265 | 88.26 | 20.14 |
| conf16 | internal | rtdetr | Shared static (raw 0.5) | 3 | 22.26 | 39.27 | 45.01 | 532 | 36654 | 129541 | 69.85 | 39.59 |
| conf16 | internal | yolov8 | Tracker default (raw 0.25) | 0 | 20.56 | 33.18 | 35.97 | 616 | 14146 | 155595 | 80.62 | 27.45 |
| conf16 | internal | rtdetr | Tracker default (raw 0.25) | 6 | 8.77 | 41.94 | 48.20 | 1467 | 103961 | 90228 | 54.44 | 57.93 |
| conf16 | internal | yolov8 | E41 (rejected V5-TF lock) | 4 | 11.03 | 35.03 | 39.48 | 1211 | 56952 | 132638 | 58.96 | 38.15 |
| conf16 | internal | rtdetr | E41 (rejected V5-TF lock) | 5 | 4.31 | 41.55 | 47.80 | 1103 | 110120 | 93993 | 52.24 | 56.17 |
| conf16 | internal | yolov8 | V6-TF (ours, frozen) | 0 | 20.20 | 33.58 | 37.90 | 334 | 16981 | 153825 | 78.12 | 28.27 |
| conf16 | internal | rtdetr | V6-TF (ours, frozen) | 1 | 26.63 | 39.71 | 45.96 | 312 | 23675 | 133357 | 77.40 | 37.82 |
