# confirmation-16 (ONE-WAY, post-freeze) — official protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conf16 | official | yolov8 | V4 (frozen, VisDrone-tuned) | 1 | 13.87 | 29.80 | 31.94 | 372 | 19554 | 177527 | 72.57 | 22.56 |
| conf16 | official | rtdetr | V4 (frozen, VisDrone-tuned) | 3 | 8.96 | 30.32 | 31.75 | 329 | 34065 | 174332 | 61.72 | 23.96 |
| conf16 | official | yolov8 | Shared static (raw 0.5) | 1 | 12.93 | 26.38 | 25.82 | 182 | 9052 | 190386 | 81.11 | 16.96 |
| conf16 | official | rtdetr | Shared static (raw 0.5) | 3 | 14.15 | 33.82 | 37.40 | 684 | 41397 | 154741 | 64.29 | 32.50 |
| conf16 | official | yolov8 | Tracker default (raw 0.25) | 1 | 15.13 | 29.61 | 31.01 | 682 | 17754 | 176127 | 74.95 | 23.17 |
| conf16 | official | rtdetr | Tracker default (raw 0.25) | 8 | -3.46 | 35.07 | 38.28 | 2823 | 108278 | 126087 | 48.79 | 45.00 |
| conf16 | official | yolov8 | E41 (rejected V5-TF lock) | 5 | 4.12 | 31.13 | 33.46 | 1558 | 61244 | 157009 | 54.12 | 31.51 |
| conf16 | official | rtdetr | E41 (rejected V5-TF lock) | 6 | -9.94 | 33.94 | 36.80 | 2886 | 117400 | 131763 | 45.37 | 42.53 |
| conf16 | official | yolov8 | V6-TF (ours, frozen) | 1 | 13.59 | 29.82 | 32.09 | 317 | 22381 | 175393 | 70.65 | 23.50 |
| conf16 | official | rtdetr | V6-TF (ours, frozen) | 1 | 13.58 | 33.11 | 36.00 | 325 | 35184 | 162624 | 65.44 | 29.06 |
