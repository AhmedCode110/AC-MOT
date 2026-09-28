# paired sequence bootstrap testdev (10,000 resamples, seed 42)

| split | protocol | system | baseline | detector | metric | diff | ci_lo | ci_hi | p_le0 | n_boot | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| testdev | internal | V6TF | V4 | yolov8 | HOTA | -0.03 | -0.84 | 0.66 | 0.54 | 10000 | 42 |
| testdev | internal | V6TF | V4 | yolov8 | IDF1 | 0.52 | -0.78 | 1.76 | 0.21 | 10000 | 42 |
| testdev | internal | V6TF | V4 | yolov8 | MOTA | -0.66 | -1.57 | 0.23 | 0.92 | 10000 | 42 |
| testdev | official | V6TF | V4 | yolov8 | HOTA | 0.29 | -0.55 | 1.09 | 0.25 | 10000 | 42 |
| testdev | official | V6TF | V4 | yolov8 | IDF1 | 0.64 | -0.67 | 1.92 | 0.17 | 10000 | 42 |
| testdev | official | V6TF | V4 | yolov8 | MOTA | -0.18 | -0.98 | 0.74 | 0.65 | 10000 | 42 |
| testdev | internal | V6TF | V4 | rtdetr | HOTA | 0.08 | -2.95 | 3.01 | 0.45 | 10000 | 42 |
| testdev | internal | V6TF | V4 | rtdetr | IDF1 | 1.25 | -3.07 | 5.42 | 0.27 | 10000 | 42 |
| testdev | internal | V6TF | V4 | rtdetr | MOTA | 3.98 | 0.01 | 8.69 | 0.03 | 10000 | 42 |
| testdev | official | V6TF | V4 | rtdetr | HOTA | 1.63 | -0.79 | 4.21 | 0.09 | 10000 | 42 |
| testdev | official | V6TF | V4 | rtdetr | IDF1 | 3.00 | -0.34 | 6.81 | 0.04 | 10000 | 42 |
| testdev | official | V6TF | V4 | rtdetr | MOTA | 3.43 | -0.02 | 7.54 | 0.03 | 10000 | 42 |
| testdev | internal | V6TF | shared_static | yolov8 | HOTA | 3.89 | 3.10 | 4.95 | 0.00 | 10000 | 42 |
| testdev | internal | V6TF | shared_static | yolov8 | IDF1 | 6.68 | 5.40 | 8.39 | 0.00 | 10000 | 42 |
| testdev | internal | V6TF | shared_static | yolov8 | MOTA | 1.88 | -0.23 | 3.92 | 0.04 | 10000 | 42 |
| testdev | official | V6TF | shared_static | yolov8 | HOTA | 3.65 | 2.95 | 4.67 | 0.00 | 10000 | 42 |
| testdev | official | V6TF | shared_static | yolov8 | IDF1 | 6.26 | 5.02 | 7.95 | 0.00 | 10000 | 42 |
| testdev | official | V6TF | shared_static | yolov8 | MOTA | 1.15 | -1.19 | 3.17 | 0.15 | 10000 | 42 |
| testdev | internal | V6TF | shared_static | rtdetr | HOTA | -2.41 | -4.42 | -0.64 | 1.00 | 10000 | 42 |
| testdev | internal | V6TF | shared_static | rtdetr | IDF1 | -2.77 | -5.44 | -0.06 | 0.98 | 10000 | 42 |
| testdev | internal | V6TF | shared_static | rtdetr | MOTA | 1.91 | -1.98 | 6.60 | 0.17 | 10000 | 42 |
| testdev | official | V6TF | shared_static | rtdetr | HOTA | -1.85 | -3.41 | -0.27 | 0.99 | 10000 | 42 |
| testdev | official | V6TF | shared_static | rtdetr | IDF1 | -2.25 | -4.73 | 0.46 | 0.94 | 10000 | 42 |
| testdev | official | V6TF | shared_static | rtdetr | MOTA | 1.30 | -2.35 | 5.79 | 0.26 | 10000 | 42 |
