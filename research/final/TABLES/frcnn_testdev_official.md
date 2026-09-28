# Faster R-CNN (unseen detector) on test-dev (post-hoc) — official protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| testdev | official | fasterrcnn | V4 (frozen, VisDrone-tuned) | 2 | 21.68 | 32.62 | 39.76 | 1226 | 28583 | 149813 | 73.56 | 34.68 |
| testdev | official | fasterrcnn | Shared static (raw 0.5) | 3 | 17.79 | 35.58 | 42.49 | 3269 | 67570 | 117697 | 62.30 | 48.68 |
| testdev | official | fasterrcnn | Tracker default (raw 0.25) | 6 | 10.38 | 33.97 | 39.55 | 6041 | 87078 | 112421 | 57.31 | 50.98 |
| testdev | official | fasterrcnn | V6-TF (ours, frozen) | 1 | 23.34 | 34.07 | 41.61 | 1006 | 29625 | 145177 | 73.96 | 36.70 |
