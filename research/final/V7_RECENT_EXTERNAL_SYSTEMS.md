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
