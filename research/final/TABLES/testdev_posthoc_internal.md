# VisDrone test-dev (POST-HOC; V4 held-out E31) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| testdev | internal | yolov8 | V4 (frozen, VisDrone-tuned) | 0 | 23.89 | 33.72 | 40.57 | 749 | 22949 | 139955 | 76.58 | 34.91 |
| testdev | internal | rtdetr | V4 (frozen, VisDrone-tuned) | 0 | 23.60 | 36.33 | 42.56 | 386 | 22440 | 141441 | 76.63 | 34.22 |
| testdev | internal | yolov8 | Shared static (raw 0.5) | 0 | 21.35 | 29.80 | 34.41 | 464 | 10660 | 157988 | 84.25 | 26.52 |
| testdev | internal | rtdetr | Shared static (raw 0.5) | 0 | 25.67 | 38.82 | 46.58 | 983 | 42164 | 116667 | 69.99 | 45.74 |
| testdev | internal | yolov8 | Tracker default (raw 0.25) | 0 | 22.84 | 31.81 | 37.20 | 1265 | 24478 | 140168 | 75.36 | 34.81 |
| testdev | internal | rtdetr | Tracker default (raw 0.25) | 6 | 9.05 | 37.75 | 43.98 | 2271 | 104065 | 89227 | 54.73 | 58.50 |
| testdev | internal | yolov8 | V6-TF (ours, frozen) | 0 | 23.23 | 33.69 | 41.09 | 754 | 24211 | 140097 | 75.58 | 34.84 |
| testdev | internal | rtdetr | V6-TF (ours, frozen) | 0 | 27.58 | 36.41 | 43.81 | 614 | 20054 | 135042 | 79.95 | 37.19 |
