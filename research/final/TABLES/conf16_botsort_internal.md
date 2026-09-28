# tracker transfer: BoT-SORT on confirmation-16 (post-freeze) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conf16 | internal | yolov8 | Tracker default (raw 0.25) + BoT-SORT | 0 | 21.59 | 35.19 | 38.99 | 353 | 15749 | 152061 | 79.85 | 29.09 |
| conf16 | internal | rtdetr | Tracker default (raw 0.25) + BoT-SORT | 8 | 4.94 | 43.68 | 50.52 | 1008 | 118788 | 84073 | 52.33 | 60.80 |
| conf16 | internal | yolov8 | Shared static (raw 0.5) + BoT-SORT | 0 | 17.98 | 31.21 | 31.86 | 92 | 6191 | 169621 | 87.87 | 20.91 |
| conf16 | internal | rtdetr | Shared static (raw 0.5) + BoT-SORT | 3 | 22.38 | 41.70 | 48.67 | 226 | 40237 | 125993 | 68.74 | 41.25 |
| conf16 | internal | yolov8 | V4 (frozen, VisDrone-tuned) + BoT-SORT | 0 | 21.57 | 35.44 | 39.82 | 226 | 15082 | 152889 | 80.32 | 28.71 |
| conf16 | internal | rtdetr | V4 (frozen, VisDrone-tuned) + BoT-SORT | 2 | 24.15 | 41.04 | 46.33 | 160 | 26383 | 136116 | 74.81 | 36.53 |
| conf16 | internal | yolov8 | V6-TF (ours, frozen) + BoT-SORT | 0 | 21.07 | 34.91 | 39.53 | 245 | 17365 | 151659 | 78.34 | 29.28 |
| conf16 | internal | rtdetr | V6-TF (ours, frozen) + BoT-SORT | 1 | 27.49 | 41.66 | 48.53 | 186 | 24993 | 130326 | 77.10 | 39.23 |
