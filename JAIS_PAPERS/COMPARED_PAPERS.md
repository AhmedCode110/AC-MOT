# Published systems compared with AC-MOT (with paper links)

All links are the verified DOI or arXiv records (`JAIS_PAPERS/refs/verified.json`). Δ = (system + AC-MOT) − (our reproduction of the system); 95% CI from a paired sequence bootstrap. Paper 2 numbers: frozen V7f, 10,000 resamples, seed 42.

## Trackers (Paper 2 — frozen V7f)
| System (venue) | Link | Role | Data | Reproduction | ΔHOTA [95% CI] | Reading |
|---|---|---|---|---|---|---|
| SparseTrack (IEEE TCSVT 2025) | https://doi.org/10.1109/TCSVT.2024.3524670 | development | MOT17 val-half | 68.88 vs paper 69.2 | +0.055 [−0.011, +0.254] | unchanged (two-stage host) |
| BoostTrack (Machine Vision and Applications 2024) | https://doi.org/10.1007/s00138-024-01531-5 | development | MOT17 val-half | 68.49 online / 71.73 + GBI | 0 (identical output) | pass-through |
| ByteTrack (ECCV 2022) | https://doi.org/10.1007/978-3-031-20047-2_1 | development | MOT17 val-half; KITTI | 67.70 | −0.014 [−0.058, +0.009] (floor 0.01); 0 (floor 0.1) | unchanged; KITTI RT-DETR-L +0.91 [+0.22, +1.55] HOTA, +5.98 MOTA |
| OC-SORT (CVPR 2023) | https://doi.org/10.1109/CVPR52729.2023.00934 | development | MOT17 val-half; KITTI | 66.43 vs paper 66.5 | **+0.613 [+0.363, +1.241]** | improved, 7/7 sequences; KITTI YOLOv8n +7.91 HOTA |
| BoT-SORT (arXiv 2022) | https://arxiv.org/abs/2206.14651 | development | KITTI | — | RT-DETR-L +0.61 [−0.39, +1.61] (MOTA +7.86); YOLOv8n −1.02 [−2.43, +0.25] (MOTA −2.43, significant) | mixed |
| PD-SORT (IEEE TCE 2025) | https://doi.org/10.1109/TCE.2025.3541839 | external, predeclared | MOT17 val-half | 68.011 = authors' released output | **+0.613 [+0.268, +1.646]** (MOTA +1.13, IDF1 +0.82) | improved, 7/7 sequences |
| Hybrid-SORT (AAAI 2024) | https://doi.org/10.1609/aaai.v38i7.28471 | external, predeclared | MOT17 val-half | 66.70 vs README 67.1 | 0 (identical output) | pass-through |
| C-TWiX (Pattern Recognition 2025) | https://doi.org/10.1016/j.patcog.2024.111169 | post-freeze external | MOT17 / KITTIMOT / DanceTrack val | MOT17 CLOSE 77.55 vs 77.8; KITTI car EXACT 89.27 vs 89.3; ped CLOSE 70.77 vs 71.4; DanceTrack CLOSE 59.62 vs 60.4 | MOT17 −0.055 [−0.456, +0.398]; **KITTI car −1.521 [−4.053, −0.134]**; ped −0.103 [−2.163, +0.034]; DanceTrack −1.005 [−2.622, +0.646] | one significant negative (KITTI car, seq 0014) |
| TrackTrack (CVPR 2025) | https://doi.org/10.1109/CVPR52734.2025.01091 | post-freeze external | DanceTrack val | CLOSE 62.96 vs 63.3 | −0.013 [−0.032, −0.001] | near pass-through (−0.01 point) |
| TOPICTrack (IEEE TIP 2025) | https://doi.org/10.1109/TIP.2025.3526066 | post-freeze external | MOT17 val-half | FAILED 67.54 vs 69.6 | −0.016 [−0.031, +0.008] | exploratory only, not in the main table |

## Detectors used as hosts (Paper 2)
| Detector | Link | Use |
|---|---|---|
| YOLOX-X (published MOT17 weights) | https://arxiv.org/abs/2107.08430 | MOT17 hosts |
| RT-DETR-L (CVPR 2024) | https://doi.org/10.1109/CVPR52733.2024.01605 | KITTI, VisDrone caches |
| YOLOv8n (Ultralytics, no paper) | — | KITTI, VisDrone caches, Paper 1 |

## Paper 1 — legacy scene-adaptive AC-MOT
Only one published tracker: ByteTrack (link above), in its default and tuned profiles, with YOLOv8n. Comparisons are against the static default, the hand-designed controller, and the matched static anchor (VisDrone test-dev, UAVDT).

## Prior design (motivation only)
V6-TF on SparseTrack −4.15 [−5.54, −1.66] and BoostTrack −5.89 HOTA (`research/final/EXTERNAL_PAPER_TRANSFER.md`) — the failure that led to V7.
