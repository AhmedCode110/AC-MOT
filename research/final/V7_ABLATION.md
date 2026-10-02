# V7 ablation — from V6-TF to the frozen V7f (development data)

HOTA unless stated; MOT17 val-half on the published YOLOX-X detections
(floor 0.01 / 0.1); KITTI = training split, official HOTA (car/ped avg).
Source: ledger E0–E20, `V7_STATISTICS.md`.

## Step by step (one mechanism at a time)
| Step | Change | Key effect (labelled) |
|---|---|---|
| V6EMU | V6-TF inside the V7 framework (always noisy bands, IoU dedup, ECDF remap) | ByteTrack floor 0.1: 56.41 vs host 67.70; OC-SORT floor 0.1: 45.72 vs 66.44 |
| V7a–V7d (Mac) | ρ regime, host-anchored clean regime (clean=upper), track-context duplicates only in noisy frames, cross-class duplicates in clean frames, motion rule only in noisy frames | SparseTrack 64.72 → 68.93 (host 68.88); BoostTrack 62.61 → 68.37 (host 68.49) |
| + regime from the whole stream (rho_frames 0) | removes mid-stream flips on transient confidence dips (MOT17-10) | ByteTrack floor 0.1: 67.561 → 67.702; OC floor 0.1: 65.697 → 66.071; BoostTrack 68.373 → 68.465 |
| + foreground track-consistent rescue | supplies a continuation stage to single-stage hosts; no-op for two-stage hosts | OC floor 0.01: 66.405 → 66.619; OC floor 0.1: 66.071 → 66.469; KITTI YOLOv8n OC: 41.14 (V7d) → 44.15 |
| + split interpretability check (= V7f) | a stream without background mode gives no evidence against the host; rescue from the stream floor for single-stage hosts | ByteTrack/BoostTrack floor 0.1 back to identical-to-host; OC floor 0.01: 67.041 (+0.61 vs host); OC floor 0.1: 66.907 (+0.46 vs host) |

## Variants tested and not adopted
| Variant | Evidence | Verdict |
|---|---|---|
| cold = none (nothing admitted at frame 1) | −0.3 to −0.5 HOTA on every MOT17 host | rejected |
| pool = raw (statistics on all emitted candidates) | within ±0.1 on MOT17; KITTI YOLOv8n +0.17 HOTA, −IDF1 | not adopted |
| cold_dup = noisy | within ±0.06 | not adopted |
| rho_ref = host | no gain once the regime uses the whole stream | not adopted |
| rescue below a two-stage host's low stage | FP +700–1000, −0.4 to −0.5 HOTA | rejected |
| ECDF scores in clean frames | ByteTrack −0.2 / −1.6 HOTA | rejected |
| rescue band (host.low, assoc) with a misdeclared single-stage low | OC +0.25 but ByteTrack −0.04 | redesigned as rescue fg |
| noisy primary = clip(host, t1, t2) | KITTI YOLOv8n ByteTrack +1.4 HOTA but VisDrone RT-DETR explosion uncontrolled (26.9 boxes/frame vs 14.9) | rejected |
| motion rule always / off (E11) | effects ≤ 0.12 HOTA, sign follows CMC presence | kept "noisy frames only"; no CMC flag |
