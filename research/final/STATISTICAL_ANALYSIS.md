# STATISTICAL ANALYSIS

## Protocol (frozen)
- Unit: the sequence. Resampling: with replacement, same indices for both
  systems (paired).
- 10,000 resamples, `numpy.random.default_rng(42)`, percentile 95% CI.
- Statistic: difference of **pooled** metrics (system − baseline). HOTA uses
  TrackEval's own `combine_sequences`; the counts are summed.
- P(Δ ≤ 0) is the share of resamples at or below zero.
- The canonical implementation is `tools/heldout_run.py::bootstrap` (E31).
  `tools/v6/stats.py`, `tools/v6/confirm_report.py` and
  `tools/v6/external/mot17_eval.py` replicate its method (same RNG, seed,
  sampling and percentiles).
- A CI that excludes 0 is called "significant" here. With 7–20 sequences
  per split, CIs are wide; per-sequence variation is reported next to every
  pooled number.

## A. Internal transfers of the frozen V6-TF
(FINAL_RESULTS.md; `outputs/v6/<split>/bootstrap*.json`)

| Setting | Detector | Comparison | ΔHOTA | ΔIDF1 | ΔMOTA | Protocol |
|---|---|---|---|---|---|---|
| confirmation-16 | YOLOv8n | V6-TF − V4 | −0.49 [−1.62, 0.69] | −0.16 [−2.12, 1.79] | −0.36 [−1.90, 1.49] | internal |
| confirmation-16 | RT-DETR-L | V6-TF − V4 | **+2.78 [0.67, 5.37]** | +4.24 [−0.29, 9.47] | **+4.62 [0.87, 9.63]** | official-compatible |
| confirmation-16 | YOLOv8n | V6-TF − shared static | **+3.68 [1.30, 6.45]** | **+7.67 [3.48, 11.98]** | +2.83 [−0.00, 5.59] | internal |
| BoT-SORT, conf-16 | RT-DETR-L | V6-TF − V4 | **+3.09 [0.41, 6.08]** | +4.72 [−0.66, 10.68] | **+5.56 [1.09, 11.36]** | official-compatible |
| Faster R-CNN, val-7 | Faster R-CNN | V6-TF − V4 | **+2.52 [0.25, 5.02]** | **+4.08 [1.20, 7.09]** | +2.10 [−0.37, 4.65] | official-compatible |
| Faster R-CNN, val-7 | Faster R-CNN | V6-TF − shared static | −0.01 [−1.34, 1.66] | +1.46 [−0.37, 3.61] | **+16.88 [12.28, 21.57]** | internal |
| Faster R-CNN, test-dev (post-hoc) | Faster R-CNN | V6-TF − V4 | **+1.46 [0.53, 2.55]** | **+1.85 [0.48, 3.56]** | **+1.66 [0.05, 3.47]** | official-compatible |
| Faster R-CNN, test-dev (post-hoc) | Faster R-CNN | V6-TF − shared static | −1.06 [−3.09, 0.80] | +0.16 [−3.24, 3.13] | **+12.84 [4.18, 22.47]** | internal |
| Faster R-CNN, UAVDT | Faster R-CNN | V6-TF − V4 | −0.23 [−0.81, 0.31] | +0.14 [−1.26, 1.47] | +0.46 [−1.62, 2.09] | internal |
| Faster R-CNN, UAVDT | Faster R-CNN | V6-TF − shared static | **−2.26 [−4.28, −0.04]** | −2.64 [−5.38, 0.84] | +4.53 [−1.24, 13.76] | internal |
| UAVDT | YOLOv8n | V6-TF − V4 | **−0.48 [−0.78, −0.14]** | **−0.99 [−1.51, −0.41]** | −0.05 [−0.83, 0.63] | internal |
| test-dev (post-hoc) | RT-DETR-L | V6-TF − V4 | +0.08 [−2.95, 3.01] | +1.25 [−3.07, 5.42] | **+3.98 [0.01, 8.69]** | internal |
| test-dev (post-hoc) | RT-DETR-L | V6-TF − shared static | **−2.41 [−4.42, −0.64]** | **−2.77 [−5.44, −0.06]** | +1.91 [−1.98, 6.60] | internal |

## B. External published systems (MOT17 val-half, TrackEval MOTChallenge protocol)
| Host | ΔMOTA | ΔHOTA | ΔIDF1 | ΔIDS | ΔFP | ΔFN |
|---|---|---|---|---|---|---|
| SparseTrack (TCSVT 2025) | **−6.14 [−8.10, −2.74]** | **−4.15 [−5.54, −1.66]** | **−4.49 [−6.19, −1.96]** | −16 [−47, 12] | **−1630 [−2977, −490]** | **+4955 [1973, 9493]** |
| BoostTrack (MVA 2024), online | **−8.87 [−12.37, −7.14]** | **−5.89 [−7.76, −2.87]** | **−6.22 [−9.33, −2.85]** | **−38 [−81, −1]** | **−1351 [−2749, −357]** | **+6167 [3090, 10367]** |
| BoostTrack + GBI | **−8.73 [−12.59, −6.50]** | **−5.37 [−7.30, −2.75]** | **−5.65 [−8.96, −2.77]** | −27 [−73, 11] | **−1998 [−3546, −668]** | **+6732 [3238, 10997]** |

## Interpretation
- The frozen layer's effect is **regime-dependent, not universal**.
- It is significantly **positive** where detector score calibration or
  domain differs from the fixed operating point. Examples: RT-DETR-L versus
  the VisDrone-tuned V4 under the official-compatible protocol, the unseen
  Faster R-CNN, and YOLOv8n versus a single shared raw threshold.
- It is **neutral** against V4 on YOLOv8n.
- It is significantly **negative** on in-domain, high-precision pedestrian
  pipelines (SparseTrack, BoostTrack on MOT17). Its FP reduction there is
  outweighed by lost true positives.
- Per-sequence variation is large. For example, among RT-DETR
  confirmation-16 cells, V6-TF scores MOTA −1.5 on uav0000273_00001 and
  +28.1 on uav0000266_03598, and V4 scores −252.4 on uav0000266_04830.
  Pooled means alone would mislead.
