# External transfer — MOT17 val-half, TrackEval MOTChallenge protocol (paper rows as published; Precision/Recall from TrackEval CLEAR)

| System | MOTA ↑ | HOTA ↑ | IDF1 ↑ | IDS ↓ | FP ↓ | FN ↓ | Prec ↑ | Rec ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SparseTrack (IEEE TCSVT 2025) — paper, Table VI | 76.80 | 69.20 | 81.40 | – | – | – | – | – |
| SparseTrack — our faithful execution (official code+ckpt) | 77.85 | 68.88 | 81.97 | 124 | 2231 | 9582 | 95.21 | 82.22 |
| SparseTrack + frozen V6-TF (same detections) | 71.71 | 64.72 | 77.49 | 108 | 601 | 14537 | 98.50 | 73.02 |
|   diagnostic: + V6-TF without duplicate suppression | 75.27 | 67.31 | 80.23 | 106 | 1039 | 12182 | 97.57 | 77.39 |
|   diagnostic: + V6-TF without motion rule | 71.73 | 64.86 | 77.78 | 106 | 594 | 14536 | 98.51 | 73.03 |
| BoostTrack (MVA 2024) — authors' re-reported online (issue #8) | 75.56 | 68.37 | 81.35 | 118 | – | – | – | – |
| BoostTrack — our execution (online) | 75.50 | 68.49 | 81.41 | 113 | 1637 | 11452 | 96.29 | 78.75 |
| BoostTrack + frozen V6-TF (online) | 66.64 | 62.61 | 75.19 | 75 | 286 | 17619 | 99.22 | 67.31 |
| BoostTrack — authors' re-reported + GBI | 80.55 | 71.33 | 83.84 | 106 | – | – | – | – |
| BoostTrack — our execution + GBI | 81.03 | 71.72 | 84.16 | 97 | 2674 | 7451 | 94.56 | 86.17 |
| BoostTrack + frozen V6-TF + GBI | 72.30 | 66.35 | 78.51 | 70 | 676 | 14183 | 98.33 | 73.68 |

## Δ frozen V6-TF (paired sequence bootstrap, 10,000 resamples, seed 42, 95% CI)

| Host system | ΔMOTA | ΔHOTA | ΔIDF1 | ΔIDS | ΔFP | ΔFN |
|---|---|---|---|---|---|---|
| SparseTrack | -6.14 [-8.10, -2.74] | -4.15 [-5.54, -1.66] | -4.49 [-6.19, -1.96] | -16.00 [-47.00, 12.00] | -1630.00 [-2977.15, -490.00] | +4955.00 [1973.00, 9493.00] |
| BoostTrack (online) | -8.87 [-12.37, -7.14] | -5.89 [-7.76, -2.87] | -6.22 [-9.33, -2.85] | -38.00 [-81.00, -1.00] | -1351.00 [-2749.00, -356.98] | +6167.00 [3090.00, 10367.07] |
| BoostTrack + GBI | -8.73 [-12.59, -6.50] | -5.37 [-7.30, -2.75] | -5.65 [-8.96, -2.77] | -27.00 [-73.00, 11.00] | -1998.00 [-3546.00, -668.00] | +6732.00 [3238.00, 10997.00] |
