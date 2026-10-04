# AC-MOT vs OATrack — experimental design (fixed before the detector is trained)

## Detector (paper-aligned re-implementation)
YOLO11m, COCO initialisation, trained on all 56 VisDrone2019-MOT-train sequences, every frame, 1536 px, last
epoch kept (`notebooks/YOLO11m_VisDrone_train_and_cache.ipynb`). OATrack's epochs, augmentation, class mapping
and "smallobj" data configuration are not published, so this is not an exact detector reproduction. Class
set: `eval5` (pedestrian, car, van, truck, bus) unless the authors' mapping is obtained; until then the
published OATrack numbers are an external reference, not an apples-to-apples comparison.

Note on data protection: the 16 train sequences that were protected for the universal-layer line (V6/V7f
confirmation) are used here to train this detector. That line evaluates COCO detectors only; any future use
of those 16 sequences with this detector must be labelled as training data.

## Splits
- AC-MOT calibration: 8 train sequences (`DETECTOR_SPLIT.json` → `calibration`), seen by the detector in
  training, never by any evaluation.
- Final benchmark: VisDrone2019-MOT-val (7 sequences) — the split OATrack reports. Not used for any choice.
- Transfer: UAVDT test (20 sequences), once, after the freeze (owner's authorization required, P4).

## Systems (same detector weights, classes, evaluator TrackEval 12c8791, tracker configs)
1. YOLO11m + Ultralytics ByteTrack (0.25 / 0.10 / 0.25 / 30 / 0.80, score fusion)
2. YOLO11m + OATrack (all parameters frozen as in the paper, min_conf 0.40)
3. AC-MOT + YOLO11m + ByteTrack
4. AC-MOT + YOLO11m + OATrack
AC-MOT acts on the detector only (input size, confidence floor, NMS IoU); the trackers are unchanged. With
OATrack's internal min_conf of 0.40, a detector confidence floor below 0.40 has no effect on OATrack, so
on that host AC-MOT effectively chooses resolution, NMS and a confidence floor ≥ 0.40.
Controls: the best static detector operating point chosen on the calibration split (matched static), and a
shuffled AC-MOT schedule at equal compute.

## Caches
Per split, resolution ∈ {1088, 1280, 1536} and NMS IoU ∈ {0.45, 0.60, 0.70}, score floor 0.01. One forward
pass per frame and resolution; the Ultralytics postprocess is applied once per NMS value. Checked: identical
to `model.predict(iou=τ)` for every box on 2 images × 2 sizes × 3 NMS values (`tools/oatrack/multi_nms.py`).
Re-applying NMS to an already-suppressed cache is not exact, which is why every NMS value is cached.

## Reading of Step 4 (development, COCO detectors)
The best operating point is strongly detector- and resolution-dependent: with OATrack frozen at min_conf 0.40,
YOLOv8n is better at 1344 than at 960 px (37.96 vs 34.73 HOTA) and RT-DETR-L is better at 960 than at 1344 px
(42.65 vs 31.88). This motivates detector-side operating-point control; it says nothing about the paper's own
detector.
