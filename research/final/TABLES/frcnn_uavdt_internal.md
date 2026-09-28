# Faster R-CNN on UAVDT (post-freeze) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uavdt | internal | fasterrcnn | V4 (frozen, VisDrone-tuned) | 8 | 18.22 | 35.14 | 44.12 | 342 | 65072 | 213389 | 66.21 | 37.41 |
| uavdt | internal | fasterrcnn | Shared static (raw 0.5) | 7 | 14.15 | 37.17 | 46.90 | 698 | 109364 | 182606 | 59.14 | 46.44 |
| uavdt | internal | fasterrcnn | Tracker default (raw 0.25) | 13 | -5.13 | 34.55 | 41.84 | 1774 | 181930 | 174702 | 47.74 | 48.75 |
| uavdt | internal | fasterrcnn | V6-TF (ours, frozen) | 8 | 18.68 | 34.91 | 44.26 | 381 | 65161 | 211682 | 66.48 | 37.91 |
