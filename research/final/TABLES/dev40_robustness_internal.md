# development-40 robustness check (not iterated; X5 = V6-TF) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev40 | internal | yolov8 | V4 (frozen, VisDrone-tuned) | 0 | 25.13 | 35.19 | 42.19 | 1188 | 49574 | 375254 | 79.63 | 34.05 |
| dev40 | internal | rtdetr | V4 (frozen, VisDrone-tuned) | 3 | 26.31 | 38.41 | 44.41 | 753 | 48120 | 370453 | 80.49 | 34.90 |
| dev40 | internal | yolov8 | Shared static (raw 0.5) | 0 | 22.72 | 31.13 | 35.47 | 838 | 19967 | 418951 | 88.26 | 26.37 |
| dev40 | internal | rtdetr | Shared static (raw 0.5) | 2 | 28.52 | 41.77 | 49.77 | 1855 | 112025 | 292840 | 71.14 | 48.54 |
| dev40 | internal | yolov8 | Tracker default (raw 0.25) | 0 | 25.25 | 34.06 | 39.75 | 2394 | 52765 | 370212 | 79.03 | 34.94 |
| dev40 | internal | rtdetr | Tracker default (raw 0.25) | 17 | 12.46 | 41.24 | 47.78 | 4837 | 285920 | 207341 | 55.85 | 63.56 |
| dev40 | internal | yolov8 | E41 (rejected V5-TF lock) | 8 | 12.69 | 35.72 | 42.19 | 4285 | 189385 | 303162 | 58.40 | 46.72 |
| dev40 | internal | rtdetr | E41 (rejected V5-TF lock) | 7 | 19.11 | 43.25 | 52.00 | 2646 | 225399 | 232225 | 59.91 | 59.19 |
| dev40 | internal | yolov8 | X5 | 0 | 25.53 | 35.82 | 43.50 | 1272 | 57592 | 364865 | 78.00 | 35.88 |
| dev40 | internal | rtdetr | X5 | 2 | 29.56 | 39.89 | 47.69 | 999 | 58696 | 341150 | 79.52 | 40.05 |
