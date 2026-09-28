# Faster R-CNN (unseen detector) on val-7 (post-freeze) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val7 | internal | fasterrcnn | V4 (frozen, VisDrone-tuned) | 0 | 18.14 | 34.78 | 40.16 | 242 | 11796 | 43092 | 67.28 | 36.01 |
| val7 | internal | fasterrcnn | Shared static (raw 0.5) | 3 | 3.31 | 36.96 | 42.56 | 670 | 32611 | 31838 | 52.13 | 52.72 |
| val7 | internal | fasterrcnn | Tracker default (raw 0.25) | 6 | -9.56 | 34.97 | 38.74 | 941 | 42011 | 30830 | 46.50 | 54.22 |
| val7 | internal | fasterrcnn | V6-TF (ours, frozen) | 0 | 20.18 | 36.95 | 44.02 | 305 | 13766 | 39683 | 66.77 | 41.08 |
