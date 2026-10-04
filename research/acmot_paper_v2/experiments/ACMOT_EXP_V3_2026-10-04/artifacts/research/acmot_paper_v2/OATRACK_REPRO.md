# OATrack re-implementation — mechanism check (development, val-7)

Paper: Qi, Wang, Jiang, "OATrack: A Quality-Gated Progressive Association Framework with a YOLO Detection
Cache for UAV Small-Object Multi-Object Tracking", Sensors 26(16):5222, 2026 (PMC13517331). No code released.
Implementation: `adapters/trackers/oatrack.py` (all equations and hyperparameters of the paper's main
configuration; three unstated choices listed in the module docstring). Tests: `tests/test_oatrack.py`.

Check (`tools/oatrack/val_check.py`): byte-identical cached detections (COCO detectors, not the paper's
YOLO11m-smallobj), 960 px, val-7, internal protocol + TrackEval HOTA. Purpose: does the re-implementation
reproduce the paper's *pattern* against Ultralytics ByteTrack (fewer ID switches and false positives, more
false negatives, AssA up, DetA down)? The absolute numbers are not comparable with the paper (other detector).

| detector | tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDS | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | ByteTrack | 35.70 | 26.81 | 49.22 | 18.95 | 39.67 | 465 | 13,886 | 40,233 |
| YOLOv8n | OATrack | 34.73 | 21.63 | 56.80 | 21.89 | 37.19 | 85 | 4,013 | 48,505 |
| RT-DETR-L | ByteTrack | 38.03 | 30.16 | 49.82 | −8.71 | 41.94 | 854 | 47,798 | 24,556 |
| RT-DETR-L | OATrack | 42.65 | 32.00 | 58.05 | 29.11 | 49.51 | 282 | 11,067 | 36,392 |

Paper, VisDrone val (YOLO11m-smallobj): IDSW −69.2 %, FP −68.8 %, FN +20 %, DetA 32.41 → 27.33,
AssA 47.18 → 55.36, HOTA 38.48 → 38.58. Re-implementation with YOLOv8n: IDS −81.7 %, FP −71.1 %,
FN +20.6 %, DetA down, AssA up, HOTA −0.97 — same pattern. With an over-confident detector (RT-DETR-L,
ByteTrack MOTA −8.71) OATrack gains strongly, as on the paper's UAVDT row (ByteTrack MOTA 7.52).
