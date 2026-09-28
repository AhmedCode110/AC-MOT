# V7 recent external systems — blocks, incompatibilities, failed reproductions

Each entry is recorded once, with the evidence, and is never silently dropped.

| System | Status | Evidence | Date |
|---|---|---|---|
| LG-MOT (TCSVT 2025) | pending audit | offline graph tracker (SUSHI family) run with `--cuda`; the README reports test-set results only, so a val reproduction has no official reference | 2026-09-28 |
| MOTIP (CVPR 2025) | pending audit | end-to-end DETR + ID decoder; candidate boundary = detections passed to the ID decoder after the detection threshold | 2026-09-28 |
| CAMELTrack | exploratory only | arXiv 2505.01257; peer-reviewed venue not verified | 2026-09-28 |
| LA-MOTR (ICCV 2025), CO-MOT (ICLR 2025), SambaMOTR (ICLR 2025) | not audited in depth | end-to-end query-based trackers; no separate detector-output stage expected (to be confirmed if time allows) | 2026-09-28 |
