# Faster R-CNN (unseen detector) on val-7 (post-freeze) — official protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val7 | official | fasterrcnn | V4 (frozen, VisDrone-tuned) | 3 | 8.54 | 30.15 | 32.75 | 270 | 14519 | 50906 | 59.04 | 29.13 |
| val7 | official | fasterrcnn | Shared static (raw 0.5) | 4 | -4.20 | 32.85 | 36.37 | 947 | 33104 | 40795 | 48.39 | 43.21 |
| val7 | official | fasterrcnn | Tracker default (raw 0.25) | 5 | -14.42 | 31.36 | 33.52 | 1463 | 40965 | 39761 | 43.91 | 44.65 |
| val7 | official | fasterrcnn | V6-TF (ours, frozen) | 3 | 10.64 | 32.67 | 36.83 | 278 | 16301 | 47607 | 59.77 | 33.72 |
