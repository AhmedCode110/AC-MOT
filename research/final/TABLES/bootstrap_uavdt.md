# paired sequence bootstrap uavdt (10,000 resamples, seed 42)

| split | protocol | system | baseline | detector | metric | diff | ci_lo | ci_hi | p_le0 | n_boot | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uavdt | internal | V6TF | V4 | yolov8 | HOTA | -0.48 | -0.78 | -0.14 | 1.00 | 10000 | 42 |
| uavdt | internal | V6TF | V4 | yolov8 | IDF1 | -0.99 | -1.51 | -0.41 | 1.00 | 10000 | 42 |
| uavdt | internal | V6TF | V4 | yolov8 | MOTA | -0.05 | -0.83 | 0.63 | 0.55 | 10000 | 42 |
| uavdt | internal | V6TF | V4 | rtdetr | HOTA | -0.17 | -2.31 | 1.97 | 0.55 | 10000 | 42 |
| uavdt | internal | V6TF | V4 | rtdetr | IDF1 | 0.66 | -3.07 | 4.18 | 0.36 | 10000 | 42 |
| uavdt | internal | V6TF | V4 | rtdetr | MOTA | 1.40 | -2.84 | 5.92 | 0.25 | 10000 | 42 |
| uavdt | internal | V6TF | shared_static | yolov8 | HOTA | 4.42 | 1.94 | 7.22 | 0.00 | 10000 | 42 |
| uavdt | internal | V6TF | shared_static | yolov8 | IDF1 | 8.12 | 3.80 | 12.61 | 0.00 | 10000 | 42 |
| uavdt | internal | V6TF | shared_static | yolov8 | MOTA | 1.79 | -3.35 | 6.85 | 0.26 | 10000 | 42 |
| uavdt | internal | V6TF | shared_static | rtdetr | HOTA | -0.53 | -2.36 | 1.34 | 0.66 | 10000 | 42 |
| uavdt | internal | V6TF | shared_static | rtdetr | IDF1 | 0.10 | -3.17 | 3.38 | 0.42 | 10000 | 42 |
| uavdt | internal | V6TF | shared_static | rtdetr | MOTA | 2.72 | -1.92 | 9.71 | 0.14 | 10000 | 42 |
