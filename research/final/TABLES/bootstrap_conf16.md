# paired sequence bootstrap conf16 (10,000 resamples, seed 42)

| split | protocol | system | baseline | detector | metric | diff | ci_lo | ci_hi | p_le0 | n_boot | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| conf16 | internal | V6TF | V4 | yolov8 | HOTA | -0.49 | -1.62 | 0.69 | 0.75 | 10000 | 42 |
| conf16 | internal | V6TF | V4 | yolov8 | IDF1 | -0.16 | -2.12 | 1.79 | 0.54 | 10000 | 42 |
| conf16 | internal | V6TF | V4 | yolov8 | MOTA | -0.36 | -1.90 | 1.49 | 0.66 | 10000 | 42 |
| conf16 | official | V6TF | V4 | yolov8 | HOTA | 0.02 | -0.89 | 1.02 | 0.47 | 10000 | 42 |
| conf16 | official | V6TF | V4 | yolov8 | IDF1 | 0.15 | -1.51 | 1.87 | 0.42 | 10000 | 42 |
| conf16 | official | V6TF | V4 | yolov8 | MOTA | -0.28 | -1.77 | 1.31 | 0.64 | 10000 | 42 |
| conf16 | internal | V6TF | V4 | rtdetr | HOTA | 0.31 | -2.73 | 3.56 | 0.41 | 10000 | 42 |
| conf16 | internal | V6TF | V4 | rtdetr | IDF1 | 1.58 | -4.37 | 7.85 | 0.31 | 10000 | 42 |
| conf16 | internal | V6TF | V4 | rtdetr | MOTA | 2.48 | -2.90 | 8.76 | 0.20 | 10000 | 42 |
| conf16 | official | V6TF | V4 | rtdetr | HOTA | 2.78 | 0.67 | 5.37 | 0.00 | 10000 | 42 |
| conf16 | official | V6TF | V4 | rtdetr | IDF1 | 4.24 | -0.29 | 9.47 | 0.04 | 10000 | 42 |
| conf16 | official | V6TF | V4 | rtdetr | MOTA | 4.62 | 0.87 | 9.63 | 0.01 | 10000 | 42 |
| conf16 | internal | V6TF | shared_static | yolov8 | HOTA | 3.68 | 1.30 | 6.45 | 0.00 | 10000 | 42 |
| conf16 | internal | V6TF | shared_static | yolov8 | IDF1 | 7.67 | 3.48 | 11.98 | 0.00 | 10000 | 42 |
| conf16 | internal | V6TF | shared_static | yolov8 | MOTA | 2.83 | -0.00 | 5.59 | 0.03 | 10000 | 42 |
| conf16 | official | V6TF | shared_static | yolov8 | HOTA | 3.45 | 1.30 | 5.84 | 0.00 | 10000 | 42 |
| conf16 | official | V6TF | shared_static | yolov8 | IDF1 | 6.28 | 2.92 | 9.74 | 0.00 | 10000 | 42 |
| conf16 | official | V6TF | shared_static | yolov8 | MOTA | 0.67 | -1.09 | 2.68 | 0.23 | 10000 | 42 |
| conf16 | internal | V6TF | shared_static | rtdetr | HOTA | 0.44 | -3.70 | 6.15 | 0.47 | 10000 | 42 |
| conf16 | internal | V6TF | shared_static | rtdetr | IDF1 | 0.95 | -6.05 | 9.72 | 0.45 | 10000 | 42 |
| conf16 | internal | V6TF | shared_static | rtdetr | MOTA | 4.38 | -1.12 | 12.20 | 0.08 | 10000 | 42 |
| conf16 | official | V6TF | shared_static | rtdetr | HOTA | -0.71 | -2.30 | 0.84 | 0.79 | 10000 | 42 |
| conf16 | official | V6TF | shared_static | rtdetr | IDF1 | -1.41 | -4.47 | 1.21 | 0.82 | 10000 | 42 |
| conf16 | official | V6TF | shared_static | rtdetr | MOTA | -0.57 | -8.89 | 5.88 | 0.52 | 10000 | 42 |
