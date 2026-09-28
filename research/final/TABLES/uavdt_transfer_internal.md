# UAVDT test (dataset transfer, post-freeze) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uavdt | internal | yolov8 | V4 (frozen, VisDrone-tuned) | 8 | 17.34 | 31.20 | 39.58 | 312 | 48373 | 233109 | 69.03 | 31.62 |
| uavdt | internal | rtdetr | V4 (frozen, VisDrone-tuned) | 4 | 25.57 | 40.55 | 50.83 | 258 | 61116 | 192374 | 70.85 | 43.57 |
| uavdt | internal | yolov8 | Shared static (raw 0.5) | 2 | 15.50 | 26.30 | 30.47 | 102 | 17828 | 270132 | 79.88 | 20.76 |
| uavdt | internal | rtdetr | Shared static (raw 0.5) | 4 | 24.24 | 40.91 | 51.39 | 190 | 71269 | 186802 | 68.38 | 45.20 |
| uavdt | internal | yolov8 | Tracker default (raw 0.25) | 6 | 17.53 | 29.01 | 34.58 | 508 | 37512 | 243137 | 72.27 | 28.68 |
| uavdt | internal | rtdetr | Tracker default (raw 0.25) | 7 | 3.29 | 40.07 | 48.59 | 1041 | 187041 | 141614 | 51.59 | 58.46 |
| uavdt | internal | yolov8 | V6-TF (ours, frozen) | 7 | 17.29 | 30.72 | 38.58 | 355 | 46297 | 235300 | 69.52 | 30.98 |
| uavdt | internal | rtdetr | V6-TF (ours, frozen) | 5 | 26.96 | 40.38 | 51.49 | 303 | 62829 | 185849 | 71.16 | 45.48 |
