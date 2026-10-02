# V7 STATISTICS — development matrix of the frozen candidate V7f

Paired sequence bootstrap: 10,000 resamples, numpy default_rng(42), the same
indices for both arms, pooled metrics recomputed per resample, percentile
95% CI of Δ = (host + V7f) − (host alone). Tools: `tools/v7/mot17_eval_v7.py
--boot`, `tools/v7/kitti/kitti_eval.py --boot`. All cells are DEVELOPMENT
evidence (the policy was selected on them); none is external evidence.

## MOT17 val-half (7 sequences; published YOLOX-X detections; TrackEval)
| Host · emission floor | Host HOTA/MOTA/IDF1 (IDS) | ΔHOTA [95% CI] | ΔMOTA [95% CI] | ΔIDF1 [95% CI] | ΔIDS |
|---|---|---|---|---|---|
| ByteTrack official · 0.01 | 67.698/77.604/79.471 (214) | −0.014 [−0.058, +0.009] | +0.058 [−0.148, +0.341] | −0.031 [−0.135, +0.035] | +4 |
| ByteTrack official · 0.1 | 67.698/77.604/79.471 (214) | 0 (identical output) | 0 | 0 | 0 |
| ByteTrack ultralytics · 0.01 | 66.000/74.756/76.417 (424) | 0 (identical) | 0 | 0 | 0 |
| ByteTrack ultralytics · 0.1 | 66.000/74.756/76.417 (424) | 0 (identical) | 0 | 0 | 0 |
| OC-SORT · 0.01 | 66.428/74.672/78.052 (211) | **+0.613 [+0.363, +1.241]** | **+1.267 [+0.173, +3.014]** | +0.656 [−0.248, +1.598] | −8 [−15, −2] |
| OC-SORT · 0.1 | 66.443/74.669/78.046 (213) | **+0.464 [+0.266, +1.086]** | **+1.262 [+0.367, +3.343]** | **+0.287 [+0.003, +0.838]** | −14 [−38, +6] |
| BoostTrack online · 0.1 | 68.492/75.502/81.413 (113) | 0 (identical) | 0 | 0 | 0 |
| BoostTrack + GBI · 0.1 | 71.725/81.032/84.163 (97) | 0 (identical) | 0 | 0 | 0 |

## KITTI tracking training (official KITTI HOTA, car/pedestrian averaged)
| Host · detector | sequences | Host HOTA/MOTA/IDF1 (IDS) | ΔHOTA [95% CI] | ΔMOTA [95% CI] | ΔIDF1 [95% CI] | ΔIDS [95% CI] |
|---|---|---|---|---|---|---|
| ByteTrack · YOLOv8n | 21 | 45.31/46.54/60.92 (561) | +0.06 [−1.19, +1.24] | −1.87 [−6.25, +0.03] | +1.49 [−0.36, +3.27] | −271 [−407, −151] |
| BoT-SORT · YOLOv8n | 20 | 49.21/49.89/64.41 (422) | −1.02 [−2.43, +0.25] | **−2.43 [−7.43, −0.14]** | +0.35 [−2.10, +2.27] | −227 [−343, −123] |
| OC-SORT · YOLOv8n | 21 | 36.27/33.99/50.46 (91) | **+7.91 [+5.47, +9.98]** | **+9.40 [+1.10, +12.30]** | **+9.60 [+5.61, +12.16]** | +79 [+43, +125] |
| ByteTrack · RT-DETR-L | 21 | 50.73/47.27/64.98 (686) | **+0.91 [+0.22, +1.55]** | **+5.98 [+1.16, +9.94]** | **+3.87 [+1.67, +5.06]** | −297 [−530, −122] |
| BoT-SORT · RT-DETR-L | 20 | 53.91/48.25/67.74 (476) | +0.61 [−0.39, +1.61] | **+7.86 [+2.21, +13.44]** | **+3.10 [+1.00, +4.57]** | −226 [−369, −120] |
| OC-SORT · RT-DETR-L | 21 | 52.64/57.59/69.87 (214) | +0.19 [−0.57, +1.19] | −0.57 [−4.36, +1.40] | +0.04 [−1.31, +1.78] | −4 [−31, +25] |

BoT-SORT cells exclude sequence 0020 (host numerical failure, ledger E19).

## Calibration / floor stress (MOT17, labelled; ledger STRESS-L)
V7f restores hosts whose fixed operating point no longer matches the detector
calibration (e.g. scale05: 0 → 66.2 / 66.7 HOTA; pow3: 60.9 → 65.8, 58.3 →
66.6, BoostTrack 61.3 → 64.8); worst cell −0.14 HOTA.

## Summary
- Where the host's own operating point already fits a clean stream (ByteTrack,
  BoostTrack on YOLOX-X MOT17): V7f passes through (Δ = 0 or within ±0.06).
- Where the host lacks a continuation stage (OC-SORT): significant gains on
  both emission floors of MOT17 and on KITTI YOLOv8n.
- Where the detector is noisy (RT-DETR-L on KITTI): significant MOTA/IDF1
  gains for the two-stage hosts, HOTA +0.6–0.9.
- Known regression: under-confident but mostly-correct detector with a
  low-threshold two-stage host (KITTI YOLOv8n with ByteTrack / BoT-SORT):
  the noisy-regime birth restriction costs MOTA (−1.9 / −2.4) while cutting
  IDS by ~50%; HOTA −1.0 for BoT-SORT (CI includes 0).
