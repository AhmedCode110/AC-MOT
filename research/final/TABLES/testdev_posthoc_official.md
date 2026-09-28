# VisDrone test-dev (POST-HOC; V4 held-out E31) — official protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| testdev | official | yolov8 | V4 (frozen, VisDrone-tuned) | 1 | 17.58 | 29.98 | 35.33 | 870 | 25298 | 162857 | 72.44 | 28.99 |
| testdev | official | rtdetr | V4 (frozen, VisDrone-tuned) | 0 | 19.69 | 32.21 | 36.63 | 792 | 18867 | 164529 | 77.45 | 28.26 |
| testdev | official | yolov8 | Shared static (raw 0.5) | 0 | 16.24 | 26.61 | 29.71 | 441 | 13243 | 178408 | 79.36 | 22.21 |
| testdev | official | rtdetr | Shared static (raw 0.5) | 1 | 21.82 | 35.69 | 41.88 | 1122 | 37743 | 140438 | 70.20 | 38.76 |
| testdev | official | yolov8 | Tracker default (raw 0.25) | 1 | 17.81 | 28.54 | 32.77 | 1395 | 24882 | 162214 | 72.96 | 29.27 |
| testdev | official | rtdetr | Tracker default (raw 0.25) | 5 | 12.97 | 35.40 | 40.73 | 3783 | 79741 | 116070 | 58.69 | 49.39 |
| testdev | official | yolov8 | V6-TF (ours, frozen) | 1 | 17.40 | 30.27 | 35.97 | 715 | 26320 | 162410 | 71.78 | 29.18 |
| testdev | official | rtdetr | V6-TF (ours, frozen) | 1 | 23.12 | 33.84 | 39.63 | 594 | 20794 | 154934 | 78.16 | 32.44 |
