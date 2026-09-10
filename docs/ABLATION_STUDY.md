# Four-way ablation study

This repository publishes the requested conceptual ablation progression:

`Default baseline -> tuned baseline -> adaptive threshold -> adaptive resolution`

## Available evidence

The checked-in table [`ablation_4way_legacy_12seq.csv`](ablation_4way_legacy_12seq.csv) is the preserved legacy result table for 12 VisDrone sequences. It is included as a traceable evidence snapshot, not as a newly run benchmark.

| System | MOTA | IDF1 | HOTA | IDS | FPS |
|---|---:|---:|---:|---:|---:|
| A0 — Default baseline | 0.1709 | 0.2877 | 0.4020 | 464 | 23.0 |
| A1 — Tuned baseline | 0.1840 | 0.3080 | 0.4196 | 311 | 23.7 |
| A2 — Adaptive threshold | 0.1909 | 0.3175 | 0.4276 | 311 | 22.9 |
| A3 — Adaptive resolution | 0.1952 | 0.3238 | 0.4330 | 325 | 23.9 |

These values are copied without scientific changes from the preserved source table in the research workspace:

`First_Seminar/01_EARLY_ABLATIONS/Run_002_2026-06-02_A2_A3/06_Tables_and_CSV/FINAL_table2_ablation_20260603_083146.csv`

## Interpretation

- A1 adds tracker tuning over A0.
- A2 adds scene-adaptive detector threshold/NMS behavior over A1.
- A3 adds adaptive inference resolution over A2 and is the adopted legacy operating point.
- FPS in this legacy table is reported as supplied by the source experiment; it is not interchangeable with the v12 replay FPS protocol.

## Reproducibility boundary

The legacy table covers 12 sequences. The current portable v12 runner and Drive cache target a separate 17-sequence workflow, use pinned Ultralytics/TrackEval components, and intentionally preserve different implementation details. The 12-sequence rows must not be presented as the final 17-sequence live benchmark.

For v12 experiments, use a development cache and evaluate candidates with `experiment.py replay`, then freeze a development selection before any test-data run. Do not use the test split to search thresholds or select a winner. The Drive cache is external to GitHub and is never committed to this repository.

## Source and protocol notes

- Detector: YOLOv8n; tracker: ByteTrack.
- A4/ReID is outside this four-way comparison.
- No new metric is claimed by this document.
- For the authoritative final live result, use the separately preserved 17-sequence report and its exact provenance rather than this legacy ablation table.
