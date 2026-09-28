# tracker transfer: BoT-SORT on confirmation-16 (post-freeze) — official protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conf16 | official | yolov8 | Tracker default (raw 0.25) + BoT-SORT | 1 | 16.02 | 31.58 | 33.80 | 442 | 19305 | 172792 | 74.52 | 24.63 |
| conf16 | official | rtdetr | Tracker default (raw 0.25) + BoT-SORT | 10 | -6.43 | 36.77 | 40.55 | 2874 | 120478 | 120635 | 47.41 | 47.38 |
| conf16 | official | yolov8 | Shared static (raw 0.5) + BoT-SORT | 1 | 13.48 | 27.67 | 27.29 | 90 | 9486 | 188768 | 81.02 | 17.66 |
| conf16 | official | rtdetr | Shared static (raw 0.5) + BoT-SORT | 3 | 14.48 | 36.36 | 41.13 | 381 | 44315 | 151360 | 63.74 | 33.98 |
| conf16 | official | yolov8 | V4 (frozen, VisDrone-tuned) + BoT-SORT | 1 | 14.70 | 31.13 | 33.53 | 313 | 20357 | 174893 | 72.76 | 23.71 |
| conf16 | official | rtdetr | V4 (frozen, VisDrone-tuned) + BoT-SORT | 4 | 8.98 | 32.05 | 33.89 | 368 | 37733 | 170574 | 60.86 | 25.60 |
| conf16 | official | yolov8 | V6-TF (ours, frozen) + BoT-SORT | 1 | 14.31 | 31.05 | 33.51 | 231 | 22847 | 173365 | 70.98 | 24.38 |
| conf16 | official | rtdetr | V6-TF (ours, frozen) + BoT-SORT | 1 | 14.54 | 35.14 | 38.62 | 200 | 36177 | 159544 | 65.84 | 30.41 |
