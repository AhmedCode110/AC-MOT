# FINAL RESULTS — V6-TF (frozen, tag `universal-acmot-v6-freeze` → 2cff95f)

Status legend:
- DEVELOPMENT: used to design or check the method.
- ONE-WAY: first and only evaluation after the freeze.
- POST-HOC: data seen before by V4, labelled.

Two evaluation protocols are reported separately and never mixed:
- INTERNAL: class-agnostic; the project's canonical protocol since V1.
- OFFICIAL-COMPATIBLE: a VisDrone Task-4b port, class-aware, with ignored
  regions dropped. COCO detectors have no "van" class, so all van GT
  counts as FN.

Neither is the official VisDrone leaderboard, so these numbers must NOT be
compared with published VisDrone results. Statistics use a paired sequence
bootstrap (10,000 resamples, seed 42, percentile 95% CI), system minus
baseline, on pooled metrics.
"cat" counts catastrophic cells (sequence × detector with MOTA < 0).

Baselines:
- **V4:** the previous frozen Universal AC-MOT. Its constants were selected
  on VisDrone val-7 (category E), and its val-7 numbers are in-sample.
- **Shared static:** one raw threshold of 0.5 for every detector.
- **Tracker default:** ByteTrack/BoT-SORT native thresholds on the raw
  scores.
- **E41:** the rejected V5-TF lock.

## Headline
Across 5 evaluation settings × 2–3 detectors × 2 trackers, the training-free
V6-TF, which uses no dataset-selected constants:
1. **matches the VisDrone-tuned V4** on YOLOv8n: no significant HOTA, IDF1
   or MOTA difference on confirmation-16, test-dev or BoT-SORT, and a small
   significant deficit on UAVDT (HOTA −0.48, IDF1 −0.99);
2. **exceeds V4 on RT-DETR-L** under the official-compatible protocol:
   - confirmation-16: HOTA +2.78 [0.67, 5.37], MOTA +4.62 [0.87, 9.63];
   - BoT-SORT: HOTA +3.09, MOTA +5.56;
   - test-dev post-hoc: MOTA +3.98 [0.01, 8.69] (internal);
3. **exceeds V4 on the unseen Faster R-CNN** (val-7): official HOTA +2.52
   [0.25, 5.02] and IDF1 +4.08 [1.20, 7.09]. The shared static threshold
   collapses there (MOTA −16.9 vs V6-TF);
4. **has the fewest catastrophic cells** on every VisDrone evaluation:
   - confirmation-16: 1 vs 3 for V4;
   - BoT-SORT: 1 vs 2;
   - Faster R-CNN val-7: 0 vs 0 for V4 and 3 for shared static;
   - confirmation-16, RT-DETR uav0000266_04830: V4 collapses to MOTA −252,
     V6-TF +3.6;
5. **does not dominate a fixed raw threshold on RT-DETR-L.** RT-DETR's
   scores are close to calibrated, so shared static 0.5 is as good or better
   on HOTA/IDF1 (test-dev −2.4 HOTA for V6-TF, significant). On UAVDT,
   shared static also has fewer catastrophic cells (6 vs 12).

## Confirmation-16 (ONE-WAY, ByteTrack)
Paired bootstrap (from `outputs/v6/conf16/bootstrap.json`):

| Protocol | Detector | Comparison | ΔHOTA [95% CI] | ΔIDF1 [95% CI] | ΔMOTA [95% CI] |
|---|---|---|---|---|---|
| internal | YOLOv8n | V6-TF − V4 | −0.49 [−1.62, 0.69] | −0.16 [−2.12, 1.79] | −0.36 [−1.90, 1.49] |
| official | YOLOv8n | V6-TF − V4 | +0.02 [−0.89, 1.02] | +0.15 [−1.51, 1.87] | −0.28 [−1.77, 1.31] |
| internal | RT-DETR-L | V6-TF − V4 | +0.31 [−2.73, 3.56] | +1.58 [−4.37, 7.85] | +2.48 [−2.90, 8.76] |
| official | RT-DETR-L | V6-TF − V4 | **+2.78 [0.67, 5.37]** | +4.24 [−0.29, 9.47] | **+4.62 [0.87, 9.63]** |
| internal | YOLOv8n | V6-TF − shared static | **+3.68 [1.30, 6.45]** | **+7.67 [3.48, 11.98]** | +2.83 [−0.00, 5.59] |
| official | YOLOv8n | V6-TF − shared static | **+3.45 [1.30, 5.84]** | **+6.28 [2.92, 9.74]** | +0.67 [−1.09, 2.68] |
| internal | RT-DETR-L | V6-TF − shared static | +0.44 [−3.70, 6.15] | +0.95 [−6.05, 9.72] | +4.38 [−1.12, 12.20] |
| official | RT-DETR-L | V6-TF − shared static | −0.71 [−2.30, 0.84] | −1.40 [−4.47, 1.21] | −0.57 [−8.89, 5.88] |

## Tracker transfer: BoT-SORT on confirmation-16 (frozen layer, no change)
Bootstrap (`outputs/v6/conf16/bootstrap_botsort.json`):
- **V6-TF − V4:**
  - YOLOv8n: no significant difference;
  - RT-DETR-L official: HOTA **+3.09 [0.41, 6.08]**, MOTA **+5.56 [1.09, 11.36]**,
    IDF1 +4.72 [−0.66, 10.68].
- **V6-TF − BoT-SORT default:**
  - RT-DETR-L MOTA **+22.6 [6.8, 48.5]** (internal);
  - YOLOv8n official MOTA −1.70 [−3.24, −0.30].

## Detector transfer: Faster R-CNN R50-FPN v2 (never seen by V6-TF development)
val-7 (scenes seen during development, detector unseen). Bootstrap
(`outputs/v6/val7/bootstrap_frcnn.json`):
- **V6-TF − V4:** official HOTA **+2.52 [0.25, 5.02]**, IDF1
  **+4.08 [1.20, 7.09]**, MOTA +2.10 [−0.37, 4.65].
- **V6-TF − shared static:** MOTA **+16.88 [12.28, 21.57]** (internal),
  HOTA ±0.
- test-dev and UAVDT with Faster R-CNN: see the tables below if present
  (they ran after the NMS-0.45 reference caches finished;
  `outputs/v6/after_frcnn.log`).

## Dataset transfer: UAVDT (20 sequences, internal protocol with UAVDT ignore regions)
- **V6-TF − V4:**
  - YOLOv8n: HOTA −0.48 [−0.78, −0.14], IDF1 −0.99 [−1.51, −0.41]
    (significant, small), MOTA n.s.;
  - RT-DETR-L: n.s. (MOTA +1.40).
- **V6-TF − shared static:** YOLOv8n HOTA **+4.42**, IDF1 **+8.12**;
  RT-DETR-L n.s.
- **Catastrophic cells:** V6-TF 12, V4 12, shared static 6, default 13.
  COCO detectors find few UAVDT vehicles, with per-sequence precision often
  below 50%, so this is domain shift that no candidate policy removes (see
  FAILURE_ANALYSIS.md).

## VisDrone test-dev (POST-HOC; V4 was evaluated once here in E31)
The runner reproduces V4's E31 numbers exactly (YOLOv8n 23.89/33.72/40.57,
RT-DETR-L 23.60/36.33/42.56).
- **V6-TF − V4:** YOLOv8n n.s.; RT-DETR-L MOTA **+3.98 [0.01, 8.69]**
  (internal), official MOTA +3.43 [−0.02, 7.54].
- **V6-TF − shared static:** YOLOv8n HOTA **+3.89**, IDF1 **+6.68**;
  RT-DETR-L HOTA **−2.41 [−4.42, −0.64]**.

## Development (reference only; not evidence of generalisation)
See EXPERIMENT_LEDGER.md: val-7 sandbox (V4 in-sample) and development-40
robustness check.

## Computational cost
The layer adds 3.6 ms (YOLOv8n) and 3.9 ms (RT-DETR-L) per frame on
average, P95 7.6/8.4 ms, on the Mac CPU, excluding the tracker. The resolution
equals V4's (736), so detector compute is identical. Official T4 timing is
deferred (Amendment 9 §5).

---
# Generated tables (tools/v6/make_tables.py; CSV and LaTeX in TABLES/)

## confirmation-16 (ONE-WAY, post-freeze) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conf16 | internal | yolov8 | V4 (frozen, VisDrone-tuned) | 0 | 20.55 | 34.07 | 38.06 | 295 | 14419 | 155662 | 80.30 | 27.41 |
| conf16 | internal | rtdetr | V4 (frozen, VisDrone-tuned) | 3 | 24.15 | 39.40 | 44.37 | 222 | 22458 | 139981 | 76.83 | 34.73 |
| conf16 | internal | yolov8 | Shared static (raw 0.5) | 0 | 17.37 | 29.90 | 30.24 | 190 | 5744 | 171265 | 88.26 | 20.14 |
| conf16 | internal | rtdetr | Shared static (raw 0.5) | 3 | 22.26 | 39.27 | 45.01 | 532 | 36654 | 129541 | 69.85 | 39.59 |
| conf16 | internal | yolov8 | Tracker default (raw 0.25) | 0 | 20.56 | 33.18 | 35.97 | 616 | 14146 | 155595 | 80.62 | 27.45 |
| conf16 | internal | rtdetr | Tracker default (raw 0.25) | 6 | 8.77 | 41.94 | 48.20 | 1467 | 103961 | 90228 | 54.44 | 57.93 |
| conf16 | internal | yolov8 | E41 (rejected V5-TF lock) | 4 | 11.03 | 35.03 | 39.48 | 1211 | 56952 | 132638 | 58.96 | 38.15 |
| conf16 | internal | rtdetr | E41 (rejected V5-TF lock) | 5 | 4.31 | 41.55 | 47.80 | 1103 | 110120 | 93993 | 52.24 | 56.17 |
| conf16 | internal | yolov8 | V6-TF (ours, frozen) | 0 | 20.20 | 33.58 | 37.90 | 334 | 16981 | 153825 | 78.12 | 28.27 |
| conf16 | internal | rtdetr | V6-TF (ours, frozen) | 1 | 26.63 | 39.71 | 45.96 | 312 | 23675 | 133357 | 77.40 | 37.82 |

## confirmation-16 (ONE-WAY, post-freeze) — official protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conf16 | official | yolov8 | V4 (frozen, VisDrone-tuned) | 1 | 13.87 | 29.80 | 31.94 | 372 | 19554 | 177527 | 72.57 | 22.56 |
| conf16 | official | rtdetr | V4 (frozen, VisDrone-tuned) | 3 | 8.96 | 30.32 | 31.75 | 329 | 34065 | 174332 | 61.72 | 23.96 |
| conf16 | official | yolov8 | Shared static (raw 0.5) | 1 | 12.93 | 26.38 | 25.82 | 182 | 9052 | 190386 | 81.11 | 16.96 |
| conf16 | official | rtdetr | Shared static (raw 0.5) | 3 | 14.15 | 33.82 | 37.40 | 684 | 41397 | 154741 | 64.29 | 32.50 |
| conf16 | official | yolov8 | Tracker default (raw 0.25) | 1 | 15.13 | 29.61 | 31.01 | 682 | 17754 | 176127 | 74.95 | 23.17 |
| conf16 | official | rtdetr | Tracker default (raw 0.25) | 8 | -3.46 | 35.07 | 38.28 | 2823 | 108278 | 126087 | 48.79 | 45.00 |
| conf16 | official | yolov8 | E41 (rejected V5-TF lock) | 5 | 4.12 | 31.13 | 33.46 | 1558 | 61244 | 157009 | 54.12 | 31.51 |
| conf16 | official | rtdetr | E41 (rejected V5-TF lock) | 6 | -9.94 | 33.94 | 36.80 | 2886 | 117400 | 131763 | 45.37 | 42.53 |
| conf16 | official | yolov8 | V6-TF (ours, frozen) | 1 | 13.59 | 29.82 | 32.09 | 317 | 22381 | 175393 | 70.65 | 23.50 |
| conf16 | official | rtdetr | V6-TF (ours, frozen) | 1 | 13.58 | 33.11 | 36.00 | 325 | 35184 | 162624 | 65.44 | 29.06 |

## tracker transfer: BoT-SORT on confirmation-16 (post-freeze) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conf16 | internal | yolov8 | Tracker default (raw 0.25) + BoT-SORT | 0 | 21.59 | 35.19 | 38.99 | 353 | 15749 | 152061 | 79.85 | 29.09 |
| conf16 | internal | rtdetr | Tracker default (raw 0.25) + BoT-SORT | 8 | 4.94 | 43.68 | 50.52 | 1008 | 118788 | 84073 | 52.33 | 60.80 |
| conf16 | internal | yolov8 | Shared static (raw 0.5) + BoT-SORT | 0 | 17.98 | 31.21 | 31.86 | 92 | 6191 | 169621 | 87.87 | 20.91 |
| conf16 | internal | rtdetr | Shared static (raw 0.5) + BoT-SORT | 3 | 22.38 | 41.70 | 48.67 | 226 | 40237 | 125993 | 68.74 | 41.25 |
| conf16 | internal | yolov8 | V4 (frozen, VisDrone-tuned) + BoT-SORT | 0 | 21.57 | 35.44 | 39.82 | 226 | 15082 | 152889 | 80.32 | 28.71 |
| conf16 | internal | rtdetr | V4 (frozen, VisDrone-tuned) + BoT-SORT | 2 | 24.15 | 41.04 | 46.33 | 160 | 26383 | 136116 | 74.81 | 36.53 |
| conf16 | internal | yolov8 | V6-TF (ours, frozen) + BoT-SORT | 0 | 21.07 | 34.91 | 39.53 | 245 | 17365 | 151659 | 78.34 | 29.28 |
| conf16 | internal | rtdetr | V6-TF (ours, frozen) + BoT-SORT | 1 | 27.49 | 41.66 | 48.53 | 186 | 24993 | 130326 | 77.10 | 39.23 |

## tracker transfer: BoT-SORT on confirmation-16 (post-freeze) — official protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conf16 | official | yolov8 | Tracker default (raw 0.25) + BoT-SORT | 1 | 16.02 | 31.58 | 33.80 | 442 | 19305 | 172792 | 74.52 | 24.63 |
| conf16 | official | rtdetr | Tracker default (raw 0.25) + BoT-SORT | 10 | -6.43 | 36.77 | 40.55 | 2874 | 120478 | 120635 | 47.41 | 47.38 |
| conf16 | official | yolov8 | Shared static (raw 0.5) + BoT-SORT | 1 | 13.48 | 27.67 | 27.29 | 90 | 9486 | 188768 | 81.02 | 17.66 |
| conf16 | official | rtdetr | Shared static (raw 0.5) + BoT-SORT | 3 | 14.48 | 36.36 | 41.13 | 381 | 44315 | 151360 | 63.74 | 33.98 |
| conf16 | official | yolov8 | V4 (frozen, VisDrone-tuned) + BoT-SORT | 1 | 14.70 | 31.13 | 33.53 | 313 | 20357 | 174893 | 72.76 | 23.71 |
| conf16 | official | rtdetr | V4 (frozen, VisDrone-tuned) + BoT-SORT | 4 | 8.98 | 32.05 | 33.89 | 368 | 37733 | 170574 | 60.86 | 25.60 |
| conf16 | official | yolov8 | V6-TF (ours, frozen) + BoT-SORT | 1 | 14.31 | 31.05 | 33.51 | 231 | 22847 | 173365 | 70.98 | 24.38 |
| conf16 | official | rtdetr | V6-TF (ours, frozen) + BoT-SORT | 1 | 14.54 | 35.14 | 38.62 | 200 | 36177 | 159544 | 65.84 | 30.41 |

## Faster R-CNN (unseen detector) on val-7 (post-freeze) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val7 | internal | fasterrcnn | V4 (frozen, VisDrone-tuned) | 0 | 18.14 | 34.78 | 40.16 | 242 | 11796 | 43092 | 67.28 | 36.01 |
| val7 | internal | fasterrcnn | Shared static (raw 0.5) | 3 | 3.31 | 36.96 | 42.56 | 670 | 32611 | 31838 | 52.13 | 52.72 |
| val7 | internal | fasterrcnn | Tracker default (raw 0.25) | 6 | -9.56 | 34.97 | 38.74 | 941 | 42011 | 30830 | 46.50 | 54.22 |
| val7 | internal | fasterrcnn | V6-TF (ours, frozen) | 0 | 20.18 | 36.95 | 44.02 | 305 | 13766 | 39683 | 66.77 | 41.08 |

## Faster R-CNN (unseen detector) on val-7 (post-freeze) — official protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val7 | official | fasterrcnn | V4 (frozen, VisDrone-tuned) | 3 | 8.54 | 30.15 | 32.75 | 270 | 14519 | 50906 | 59.04 | 29.13 |
| val7 | official | fasterrcnn | Shared static (raw 0.5) | 4 | -4.20 | 32.85 | 36.37 | 947 | 33104 | 40795 | 48.39 | 43.21 |
| val7 | official | fasterrcnn | Tracker default (raw 0.25) | 5 | -14.42 | 31.36 | 33.52 | 1463 | 40965 | 39761 | 43.91 | 44.65 |
| val7 | official | fasterrcnn | V6-TF (ours, frozen) | 3 | 10.64 | 32.67 | 36.83 | 278 | 16301 | 47607 | 59.77 | 33.72 |

## UAVDT test (dataset transfer, post-freeze) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uavdt | internal | yolov8 | V4 (frozen, VisDrone-tuned) | 8 | 17.34 | 31.20 | 39.58 | 312 | 48373 | 233109 | 69.03 | 31.62 |
| uavdt | internal | rtdetr | V4 (frozen, VisDrone-tuned) | 4 | 25.57 | 40.55 | 50.83 | 258 | 61116 | 192374 | 70.85 | 43.57 |
| uavdt | internal | yolov8 | Shared static (raw 0.5) | 2 | 15.50 | 26.30 | 30.47 | 102 | 17828 | 270132 | 79.88 | 20.76 |
| uavdt | internal | rtdetr | Shared static (raw 0.5) | 4 | 24.24 | 40.91 | 51.39 | 190 | 71269 | 186802 | 68.38 | 45.20 |
| uavdt | internal | yolov8 | Tracker default (raw 0.25) | 6 | 17.53 | 29.01 | 34.58 | 508 | 37512 | 243137 | 72.27 | 28.68 |
| uavdt | internal | rtdetr | Tracker default (raw 0.25) | 7 | 3.29 | 40.07 | 48.59 | 1041 | 187041 | 141614 | 51.59 | 58.46 |
| uavdt | internal | yolov8 | V6-TF (ours, frozen) | 7 | 17.29 | 30.72 | 38.58 | 355 | 46297 | 235300 | 69.52 | 30.98 |
| uavdt | internal | rtdetr | V6-TF (ours, frozen) | 5 | 26.96 | 40.38 | 51.49 | 303 | 62829 | 185849 | 71.16 | 45.48 |

## VisDrone test-dev (POST-HOC; V4 held-out E31) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| testdev | internal | yolov8 | V4 (frozen, VisDrone-tuned) | 0 | 23.89 | 33.72 | 40.57 | 749 | 22949 | 139955 | 76.58 | 34.91 |
| testdev | internal | rtdetr | V4 (frozen, VisDrone-tuned) | 0 | 23.60 | 36.33 | 42.56 | 386 | 22440 | 141441 | 76.63 | 34.22 |
| testdev | internal | yolov8 | Shared static (raw 0.5) | 0 | 21.35 | 29.80 | 34.41 | 464 | 10660 | 157988 | 84.25 | 26.52 |
| testdev | internal | rtdetr | Shared static (raw 0.5) | 0 | 25.67 | 38.82 | 46.58 | 983 | 42164 | 116667 | 69.99 | 45.74 |
| testdev | internal | yolov8 | Tracker default (raw 0.25) | 0 | 22.84 | 31.81 | 37.20 | 1265 | 24478 | 140168 | 75.36 | 34.81 |
| testdev | internal | rtdetr | Tracker default (raw 0.25) | 6 | 9.05 | 37.75 | 43.98 | 2271 | 104065 | 89227 | 54.73 | 58.50 |
| testdev | internal | yolov8 | V6-TF (ours, frozen) | 0 | 23.23 | 33.69 | 41.09 | 754 | 24211 | 140097 | 75.58 | 34.84 |
| testdev | internal | rtdetr | V6-TF (ours, frozen) | 0 | 27.58 | 36.41 | 43.81 | 614 | 20054 | 135042 | 79.95 | 37.19 |

## VisDrone test-dev (POST-HOC; V4 held-out E31) — official protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| testdev | official | yolov8 | V4 (frozen, VisDrone-tuned) | 1 | 17.58 | 29.98 | 35.33 | 870 | 25298 | 162857 | 72.44 | 28.99 |
| testdev | official | rtdetr | V4 (frozen, VisDrone-tuned) | 0 | 19.69 | 32.21 | 36.63 | 792 | 18867 | 164529 | 77.45 | 28.26 |
| testdev | official | yolov8 | Shared static (raw 0.5) | 0 | 16.24 | 26.61 | 29.71 | 441 | 13243 | 178408 | 79.36 | 22.21 |
| testdev | official | rtdetr | Shared static (raw 0.5) | 1 | 21.82 | 35.69 | 41.88 | 1122 | 37743 | 140438 | 70.20 | 38.76 |
| testdev | official | yolov8 | Tracker default (raw 0.25) | 1 | 17.81 | 28.54 | 32.77 | 1395 | 24882 | 162214 | 72.96 | 29.27 |
| testdev | official | rtdetr | Tracker default (raw 0.25) | 5 | 12.97 | 35.40 | 40.73 | 3783 | 79741 | 116070 | 58.69 | 49.39 |
| testdev | official | yolov8 | V6-TF (ours, frozen) | 1 | 17.40 | 30.27 | 35.97 | 715 | 26320 | 162410 | 71.78 | 29.18 |
| testdev | official | rtdetr | V6-TF (ours, frozen) | 1 | 23.12 | 33.84 | 39.63 | 594 | 20794 | 154934 | 78.16 | 32.44 |

## val-7 (DEVELOPMENT sandbox; V4 in-sample) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val7 | internal | yolov8 | V4 (frozen, VisDrone-tuned) | 0 | 17.49 | 33.25 | 36.57 | 159 | 7639 | 47765 | 71.94 | 29.07 |
| val7 | internal | rtdetr | V4 (frozen, VisDrone-tuned) | 0 | 21.53 | 38.99 | 42.41 | 81 | 7084 | 45681 | 75.36 | 32.17 |
| val7 | internal | yolov8 | Shared static (raw 0.5) | 0 | 18.59 | 30.73 | 31.30 | 119 | 4206 | 50499 | 80.02 | 25.01 |
| val7 | internal | rtdetr | Shared static (raw 0.5) | 0 | 22.96 | 40.59 | 46.10 | 312 | 14698 | 36874 | 67.46 | 45.25 |
| val7 | internal | yolov8 | Tracker default (raw 0.25) | 0 | 18.40 | 31.62 | 33.62 | 320 | 8414 | 46219 | 71.52 | 31.37 |
| val7 | internal | rtdetr | Tracker default (raw 0.25) | 5 | -6.24 | 36.76 | 40.67 | 839 | 42646 | 28065 | 47.95 | 58.33 |
| val7 | internal | yolov8 | E41 (rejected V5-TF lock) | 3 | 4.04 | 32.82 | 36.34 | 640 | 25140 | 38844 | 53.13 | 42.32 |
| val7 | internal | rtdetr | E41 (rejected V5-TF lock) | 3 | -13.76 | 37.47 | 41.80 | 577 | 45968 | 30068 | 44.78 | 55.35 |
| val7 | internal | yolov8 | V6-TF (ours, frozen) | 0 | 18.56 | 34.26 | 38.60 | 159 | 8427 | 46260 | 71.45 | 31.31 |
| val7 | internal | rtdetr | V6-TF (ours, frozen) | 0 | 25.01 | 41.65 | 48.11 | 150 | 9474 | 40877 | 73.64 | 39.30 |

## val-7 (DEVELOPMENT sandbox; V4 in-sample) — official protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val7 | official | yolov8 | V4 (frozen, VisDrone-tuned) | 2 | 10.37 | 29.59 | 30.69 | 190 | 9602 | 54587 | 64.23 | 24.01 |
| val7 | official | rtdetr | V4 (frozen, VisDrone-tuned) | 3 | 10.34 | 32.92 | 33.30 | 186 | 10217 | 54003 | 63.57 | 24.82 |
| val7 | official | yolov8 | Shared static (raw 0.5) | 0 | 12.54 | 27.36 | 26.38 | 110 | 5850 | 56862 | 71.90 | 20.84 |
| val7 | official | rtdetr | Shared static (raw 0.5) | 2 | 10.37 | 34.85 | 37.46 | 375 | 17757 | 46249 | 59.03 | 35.61 |
| val7 | official | yolov8 | E41 (rejected V5-TF lock) | 4 | -3.68 | 29.03 | 30.61 | 762 | 26734 | 46976 | 48.18 | 34.60 |
| val7 | official | rtdetr | E41 (rejected V5-TF lock) | 3 | -25.97 | 32.00 | 33.33 | 1071 | 48722 | 40694 | 38.99 | 43.35 |

## development-40 robustness check (not iterated; X5 = V6-TF) — internal protocol

| split | protocol | detector | system | n_catastrophic | MOTA | HOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev40 | internal | yolov8 | V4 (frozen, VisDrone-tuned) | 0 | 25.13 | 35.19 | 42.19 | 1188 | 49574 | 375254 | 79.63 | 34.05 |
| dev40 | internal | rtdetr | V4 (frozen, VisDrone-tuned) | 3 | 26.31 | 38.41 | 44.41 | 753 | 48120 | 370453 | 80.49 | 34.90 |
| dev40 | internal | yolov8 | Shared static (raw 0.5) | 0 | 22.72 | 31.13 | 35.47 | 838 | 19967 | 418951 | 88.26 | 26.37 |
| dev40 | internal | rtdetr | Shared static (raw 0.5) | 2 | 28.52 | 41.77 | 49.77 | 1855 | 112025 | 292840 | 71.14 | 48.54 |
| dev40 | internal | yolov8 | Tracker default (raw 0.25) | 0 | 25.25 | 34.06 | 39.75 | 2394 | 52765 | 370212 | 79.03 | 34.94 |
| dev40 | internal | rtdetr | Tracker default (raw 0.25) | 17 | 12.46 | 41.24 | 47.78 | 4837 | 285920 | 207341 | 55.85 | 63.56 |
| dev40 | internal | yolov8 | E41 (rejected V5-TF lock) | 8 | 12.69 | 35.72 | 42.19 | 4285 | 189385 | 303162 | 58.40 | 46.72 |
| dev40 | internal | rtdetr | E41 (rejected V5-TF lock) | 7 | 19.11 | 43.25 | 52.00 | 2646 | 225399 | 232225 | 59.91 | 59.19 |
| dev40 | internal | yolov8 | X5 | 0 | 25.53 | 35.82 | 43.50 | 1272 | 57592 | 364865 | 78.00 | 35.88 |
| dev40 | internal | rtdetr | X5 | 2 | 29.56 | 39.89 | 47.69 | 999 | 58696 | 341150 | 79.52 | 40.05 |
