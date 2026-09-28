# Faster R-CNN (unseen detector) on test-dev (post-hoc) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| testdev | internal | fasterrcnn | V4 (frozen, VisDrone-tuned) | 1 | 25.99 | 36.09 | 44.53 | 977 | 31243 | 126922 | 73.82 | 40.97 |
| testdev | internal | fasterrcnn | Shared static (raw 0.5) | 5 | 13.35 | 37.16 | 44.72 | 2141 | 89961 | 94209 | 57.32 | 56.18 |
| testdev | internal | fasterrcnn | Tracker default (raw 0.25) | 8 | -1.49 | 35.05 | 41.07 | 3047 | 125283 | 89896 | 49.97 | 58.19 |
| testdev | internal | fasterrcnn | V6-TF (ours, frozen) | 1 | 26.19 | 36.10 | 44.89 | 1028 | 32514 | 125151 | 73.43 | 41.79 |
