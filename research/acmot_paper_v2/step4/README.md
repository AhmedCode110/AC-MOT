# Step 4 — operating point on top of ByteTrack and OATrack (development, val-7, IN-SAMPLE)

`tools/oatrack/op_sweep.py`; detections from the sweep cache (COCO YOLOv8n and RT-DETR-L, not the paper's
YOLO11m); internal protocol + TrackEval HOTA. Every row was evaluated on the same 7 sequences the best value
is read from, so the "best" columns are an in-sample upper bound, not a result. Raw numbers:
`sweep_yolov8.json`, `sweep_rtdetr.json`.

| detector, px | ByteTrack host | ByteTrack, best conf filter | OATrack (paper, min_conf 0.40) | OATrack, best min_conf |
|---|---|---|---|---|
| YOLOv8n, 960 | 35.70 | 35.70 (none) | 34.73 | 36.31 (0.30) |
| YOLOv8n, 1344 | 37.20 | 37.28 (0.25) | 37.96 | 40.46 (0.25) |
| RT-DETR-L, 960 | 38.03 | 40.97 (0.40) | 42.65 | 43.23 (0.35) |
| RT-DETR-L, 1344 | 32.27 | 32.41 (0.35) | 31.88 | 33.71 (0.25) |

Reading:
1. OATrack's fixed min_conf = 0.40 is not the best operating point for either detector; the best value
   (0.25–0.35) and the best resolution depend on the detector — the gap AC-MOT is meant to close.
2. A calibrated confidence filter in front of ByteTrack does not reach OATrack (RT-DETR-L 40.97 vs 42.65).
3. The candidate claim is therefore "OATrack + AC-MOT operating-point calibration > OATrack", to be shown
   with a label-free or calibration-split selection, then on held-out data with the VisDrone-trained detector.
