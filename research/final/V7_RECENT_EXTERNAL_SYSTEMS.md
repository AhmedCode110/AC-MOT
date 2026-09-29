# V7 recent external systems — provenance and results

Protocol: `V7_RECENT_EXTERNAL_PROTOCOL.md` (preregistered). Machine-readable:
`V7_RECENT_EXTERNAL_RESULTS.json`. Blocks and exclusions:
`V7_RECENT_EXTERNAL_FAILURES.md`. Policy: V7f @ 488df9a, unchanged.

## Official reference numbers (recorded before any run)
| System | Source | Split | HOTA | MOTA | IDF1 | other |
|---|---|---|---|---|---|---|
| TOPICTrack | official README, "MOT17-half-val" | MOT17 val-half | 69.6 | 79.8 | 81.2 | FP 3,028, FN 7,612 |
| TOPICTrack | official README, "MOT20-half-val" | MOT20 val-half | 57.5 | 73.0 | 73.6 | FP 28,583, FN 135,945 |
| C-TWiX | official README, "HOTA on the validation sets" | MOT17 val-half (CenterTrack) | 77.8 | – | – | detections: YOLOX from ByteTrack |
| C-TWiX | same | DanceTrack val | 60.4 | – | – | detections: YOLOX from ByteTrack |
| C-TWiX | same | KITTIMOT val car / ped | 89.3 / 71.4 | – | – | detections: Permatrack |
| TrackTrack | CVPR 2025 paper, Tables 4–7 (full method, TPA + TAI, D_del) | MOT17 val-half | 69.1 | – | – | AssA 72.7; post-processed output (GBI), as `run.py` evaluates `_post` |
| TrackTrack | same | DanceTrack val | 63.3 | – | – | AssA 49.7; post-processed output (AFLink) |

## Systems
Filled per system as runs complete: citation, venue, year, repository and
commit, checkpoint / detection hashes, ReID source, dataset, split, evaluator,
configuration, reference metrics, reproduced metrics, classification, then
baseline vs + V7f with CIs and per-sequence wins / ties / losses.

### TOPICTrack — MOT17 val-half (run 36483670872)
Reproduced (interpolated, as the README evaluates): HOTA 67.538, MOTA 79.070, IDF1 78.634, IDS 166, FP 1,841, FN 9,272 vs reference 69.6 / 79.8 / 81.2 → **FAILED** (ΔHOTA −2.06 > 1.0). Not in the main table; details and the exploratory V7f arm in `V7_RECENT_EXTERNAL_FAILURES.md`.

### TrackTrack — DanceTrack val (features: run 36487286878; tracking/evaluation: run 36506599893)
TrackTrack @ ee7f1c5 (CVPR 2025), released detections (NMS 0.80 / 0.95 views) and DanceTrack FastReID weights (sha256 in
`recent/assets/SHA256SUMS_tracktrack.txt`); features re-extracted on CPU float32; official `run.py` loop, per-sequence
parameters, seed 10000; `_post` = AFLink as in the paper. Policy lock 10/10.
Reproduced (post-processed): HOTA 62.961, AssA 49.111, MOTA 92.517, IDF1 66.550, IDS 1,336 vs paper 63.3 / AssA 49.7 → **CLOSE**
(ΔHOTA −0.34; documented cause: CPU float32 ReID features; nothing tuned).
Host + V7f (10,000 resamples, seed 42):
- raw: HOTA 62.730 → 62.718, ΔHOTA −0.012 [−0.032, −0.001], ΔMOTA −0.073 [−0.208, −0.004], ΔIDF1 +0.010 [+0.000, +0.028], W/T/L 0/21/4
- post: HOTA 62.961 → 62.948, ΔHOTA −0.013 [−0.032, −0.001], ΔMOTA −0.073 [−0.208, −0.004], ΔIDF1 +0.009 [−0.001, +0.028], W/T/L 0/21/4
- regimes over 25,508 frames: clean 25,453, noisy 30, cold 25 (`recent/tracktrack/audit_DanceTrack.json`).
Reading: near pass-through on a clean two-view host; the HOTA interval excludes 0 but the effect is −0.01 HOTA points
(below the 0.01 tie threshold per sequence on 21 of 25 sequences). No improvement claim.
