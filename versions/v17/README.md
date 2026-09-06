# AC-MOT v17 — Presentation-Aligned Scientific Evaluation

## Status
UNVERIFIED CANDIDATE. Do not claim measured improvements until a saved v17 result exists.

## Starting point
v15 verified decoded/live-frame processing at 34.06 FPS for its adaptive runtime system. v16 introduced a paper-evaluation harness, but its proposed ablations emphasized recovery/feedback/adaptive-birth implementation details rather than the core SmartCalibrator story shown in the seminar. The attempted v16 run on the second Colab account did not start because GitHub authentication returned HTTP 401.

## Problem discovered
The current `core.py` v15 implementation is not identical to the seminar SmartCalibrator description. With `recovery=True`, detector confidence is fixed at the low recovery value and adaptive NMS is disabled, so the v15 SCI primarily controls resolution plus tracker-side details. The seminar and historical v10 notebook describe SCI driving detector confidence, NMS IoU and resolution.

## Exact v17 change
v17 does not overwrite v15/v16. It adds `core_v17.py`, which restores the historical v10 SmartCalibrator mapping recovered from `AC_MOT_v10.ipynb`:

- `conf = 0.245 - 0.050 * SCI`, with a -0.012 scene nudge for crowded/tiny/night, clipped to [0.19, 0.28].
- `NMS IoU = 0.490 - 0.050 * SCI`, with a -0.012 blur nudge, clipped to [0.40, 0.52].
- `imgsz = 832` when SCI > 0.60 or tiny ratio > 0.50; `736` when SCI > 0.35 or scene is crowded/tiny; otherwise `640`.
- SCI uses the seminar five cues, seven-reading smoothing, and every-10-frame analysis by default.

## Systems
1. A0_DEFAULT_FIXED640 — stock-style ByteTrack thresholds; fixed detector settings.
2. A1_TUNED_FIXED640 — tuned ByteTrack; fixed detector settings.
3. A2_THRESHOLD_ONLY — tuned tracker + adaptive confidence/NMS IoU; fixed 640 resolution.
4. A2R_RESOLUTION_ONLY — tuned tracker + adaptive resolution; fixed confidence/NMS IoU.
5. A3_FULL_ACMOT_PRESENTATION — tuned tracker + adaptive confidence/NMS IoU/resolution.
6. A3_NO_SMOOTHING — full A3 but SCI smoothing window=1.
7. A3_EVERY_FRAME_SCI — full A3 but analyze every frame instead of every 10 frames.

## Metrics
Pinned TrackEval outputs HOTA, DetA, AssA, MOTA, IDF1, IDS, FN and FP using the existing custom research GT filter. Speed reporting uses decoded/live-frame processing time: analysis/controller + one fresh YOLOv8n FP16 inference + ByteTrack. JPEG decode is reported separately and Drive-to-local copy is excluded.

## Realtime reporting
- >=25 FPS: STRICT_REALTIME
- 20.00–24.99 FPS: ACCEPTABLE_REALTIME / near-real-time
- <20 FPS: BELOW_REALTIME

25 FPS is retained as the strict camera-rate target. v17 does not silently relabel 20 FPS as 25 FPS realtime. Automatic ranking chooses the highest-HOTA system among systems at or above 20 FPS; IDF1 then MOTA break ties. It also separately reports the highest-quality system overall.

## Progress output
Each system processes all 17 videos / 6635 frames. During processing the console reports the current sequence/video name, frame index, SCI, confidence, NMS IoU, image size, measured processing FPS, progress bar and ETA. Before the run it prints the exact per-video pipeline.

## Research integrity
v17 results are an evaluation of these predeclared variants. They must not be used to repeatedly retune parameters on the same held-out test split and then described as an unbiased final test result. If a variant is selected/tuned from these results for a new final system, that is a new version and should be validated on an independent split/protocol.

## Expected output
`paper_comparison_v17_FINAL.csv`, `paper_comparison_v17_FINAL.md`, `BEST_RESULT_FINAL.json`, raw timing, per-sequence recordings and pinned TrackEval results.
