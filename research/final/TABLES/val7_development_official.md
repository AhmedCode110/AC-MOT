# val-7 (DEVELOPMENT sandbox; V4 in-sample) — official protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val7 | official | yolov8 | V4 (frozen, VisDrone-tuned) | 2 | 10.37 | 29.59 | 30.69 | 190 | 9602 | 54587 | 64.23 | 24.01 |
| val7 | official | rtdetr | V4 (frozen, VisDrone-tuned) | 3 | 10.34 | 32.92 | 33.30 | 186 | 10217 | 54003 | 63.57 | 24.82 |
| val7 | official | yolov8 | Shared static (raw 0.5) | 0 | 12.54 | 27.36 | 26.38 | 110 | 5850 | 56862 | 71.90 | 20.84 |
| val7 | official | rtdetr | Shared static (raw 0.5) | 2 | 10.37 | 34.85 | 37.46 | 375 | 17757 | 46249 | 59.03 | 35.61 |
| val7 | official | yolov8 | E41 (rejected V5-TF lock) | 4 | -3.68 | 29.03 | 30.61 | 762 | 26734 | 46976 | 48.18 | 34.60 |
| val7 | official | rtdetr | E41 (rejected V5-TF lock) | 3 | -25.97 | 32.00 | 33.33 | 1071 | 48722 | 40694 | 38.99 | 43.35 |
