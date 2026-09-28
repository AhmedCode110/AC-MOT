# V7 recent external systems — blocks, incompatibilities, failed reproductions

Each entry is recorded once, with the evidence, and is never silently dropped.

| System | Status | Evidence | Date |
|---|---|---|---|
| LG-MOT (TCSVT 2025) | pending audit | offline graph tracker (SUSHI family) run with `--cuda`; the README reports test-set results only, so a val reproduction has no official reference | 2026-09-28 |
| MOTIP (CVPR 2025) | pending audit | end-to-end DETR + ID decoder; candidate boundary = detections passed to the ID decoder after the detection threshold | 2026-09-28 |
| CAMELTrack | exploratory only | arXiv 2505.01257; peer-reviewed venue not verified | 2026-09-28 |
| LA-MOTR (ICCV 2025), CO-MOT (ICLR 2025), SambaMOTR (ICLR 2025) | not audited in depth | end-to-end query-based trackers; no separate detector-output stage expected (to be confirmed if time allows) | 2026-09-28 |
| TOPICTrack (IEEE TIP 2025), MOT17 val-half | FAILED reproduction (section 4: |ΔHOTA| > 1.0) | official loop on CPU fp32, detector `topictrack_ablation.pth.tar`, FastReID `mot17_sbs_S50.pth` (run 36483670872, repo 2d1a504): interpolated output HOTA 67.54 / MOTA 79.07 / IDF1 78.63 (FP 1,841, FN 9,272) vs README 69.6 / 79.8 / 81.2 (FP 3,028, FN 7,612); raw output HOTA 65.81. Cause not diagnosed (the run emits fewer detections: FP −1,187, FN +1,660 vs README). The V7f arm ran in the same job and is recorded as exploratory only, not in the main table: raw ΔHOTA +0.006 [−0.016, +0.025], interpolated −0.016 [−0.031, +0.008] (10,000 resamples, seed 42), W/T/L 2/1/4 on raw (|Δ| per sequence ≤ 0.07); the predeclared pass-through prediction (|ΔHOTA| ≤ 0.1, CI including 0) held. MOT20 val-half not run. Files: `research/final/recent/topictrack/` | 2026-09-28 |
