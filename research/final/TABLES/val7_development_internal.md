# val-7 (DEVELOPMENT sandbox; V4 in-sample) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val7 | internal | yolov8 | V4 (frozen, VisDrone-tuned) | 0 | 17.49 | 33.25 | 36.57 | 159 | 7639 | 47765 | 71.94 | 29.07 |
| val7 | internal | rtdetr | V4 (frozen, VisDrone-tuned) | 0 | 21.53 | 38.99 | 42.41 | 81 | 7084 | 45681 | 75.36 | 32.17 |
| val7 | internal | yolov8 | Shared static (raw 0.5) | 0 | 18.59 | 30.73 | 31.30 | 119 | 4206 | 50499 | 80.02 | 25.01 |
| val7 | internal | rtdetr | Shared static (raw 0.5) | 0 | 22.96 | 40.59 | 46.10 | 312 | 14698 | 36874 | 67.46 | 45.25 |
| val7 | internal | yolov8 | Tracker default (raw 0.25) | 0 | 18.40 | 31.62 | 33.62 | 320 | 8414 | 46219 | 71.52 | 31.37 |
| val7 | internal | rtdetr | Tracker default (raw 0.25) | 5 | -6.24 | 36.76 | 40.67 | 839 | 42646 | 28065 | 47.95 | 58.33 |
| val7 | internal | yolov8 | E41 (rejected V5-TF lock) | 3 | 4.04 | 32.82 | 36.34 | 640 | 25140 | 38844 | 53.13 | 42.32 |
| val7 | internal | rtdetr | E41 (rejected V5-TF lock) | 3 | -13.76 | 37.47 | 41.80 | 577 | 45968 | 30068 | 44.78 | 55.35 |
| val7 | internal | yolov8 | V6-TF (ours, frozen) | 0 | 18.56 | 34.26 | 38.60 | 159 | 8427 | 46260 | 71.45 | 31.31 |
| val7 | internal | rtdetr | V6-TF (ours, frozen) | 0 | 25.01 | 41.65 | 48.11 | 150 | 9474 | 40877 | 73.64 | 39.30 |
