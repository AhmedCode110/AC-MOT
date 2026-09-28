# FAILURE ANALYSIS — V6-TF and its predecessors

Tools:
- `tools/v6/det_audit.py` labels each candidate or output box:
  - evaluated object;
  - unevaluated object or region (ignored region, excluded class, heavy
    occlusion);
  - clutter.
- The ledger scripts attribute clutter boxes to ghost tracks (tracks that
  never touch GT) or to real tracks.

The internal protocol is used unless stated otherwise.

## 1. Failure taxonomy
| Class | Mechanism | Where observed | Addressed by V6-TF? |
|---|---|---|---|
| F-A Candidate inflation | a rank- or Otsu-relative threshold always promotes a fixed share of candidates, including in frames without confident objects | E41 (dev-40: primaries 2–3× GT; 15 catastrophic cells), F1–F3 | yes, by the nested split (precision safeguard): 0 cat val-7, 2 dev-40, 1 conf-16 |
| F-B Duplicate tracks | detector-native suppression (YOLO 0.7, RT-DETR none) leaves overlapping boxes that each start a track | E41: 19% duplicate output boxes | yes, by IoU-0.5 suppression (0.0–0.1% duplicates) |
| F-C Ghost persistence | ghost tracks survive through a broad extension band | E41/X1: ghost tracks median 16 frames | partly: the extension band is the foreground's lower class only; ghost boxes fall from 19.1k (E41) to 2.7k (V6-TF) on val-7 RT-DETR |
| F-D Leader-relative collapse in near-empty scenes | when a scene is almost empty, the leader of the candidate stream is itself clutter; a leader-relative gate (V4) admits clutter | V4, conf-16 RT-DETR uav0000266_04830: 918 ghost-clutter boxes, MOTA −252 | yes: V6-TF emits 99 ghost boxes, MOTA +3.6 |
| F-E Unevaluated real objects counted as FP | the internal protocol drops heavily occluded objects and non-target classes from GT but not from predictions | dev-40 RT-DETR uav0000263: V6-TF 2,224 boxes on unevaluated objects vs 898 for V4, clutter only +282 | evaluation artefact, partly removed by the official-compatible protocol (ignored regions); reported, not "fixed" |
| F-F Dense clutter with more true positives | in dense scenes V6-TF finds many more objects but also more clutter | conf-16 RT-DETR uav0000273: TP 4,786 vs 2,856 (V4), but +1,456 unevaluated and +2,487 clutter boxes → MOTA −1.5 (V4 12.5) | NOT fixed: the only V6-TF catastrophic cell on confirmation-16 |
| F-G Domain shift (detector blind to the target domain) | COCO detectors on UAVDT's tiny, high-altitude vehicles: per-sequence precision 19–53%, recall low | UAVDT: 12 catastrophic cells for V6-TF and V4; shared static 6 (it outputs almost nothing: recall 0.0 and MOTA 0.0 on 4 YOLOv8n and 2 RT-DETR-L sequences) | NOT fixed: a candidate-control layer cannot create detections; relative thresholds still admit the detector's best (wrong) candidates |
| F-H Calibrated-detector ceiling | when a detector's raw scores are already close to calibrated (RT-DETR-L), one fixed raw threshold is a strong operating point | test-dev RT-DETR: shared static HOTA +2.4 over V6-TF (significant) | NOT fixed: the price of calibration invariance; V6-TF wins where calibration differs (YOLOv8n, Faster R-CNN) |
| F-I Non-affine recalibration | the nested Otsu is equivariant to affine logit maps (Platt) only | val-7 stress s³ / 0.5·s: 2–4 cat | NOT fixed (limitation) |
| F-J Adapter emission contract | raising the adapter's emission floor removes the background mass the split relies on | val-7 floor 0.05/0.1: YOLO HOTA −4.9 / −8.9 | documented interface requirement |

## 2. Representative cases
1. **RT-DETR uav0000266_04830 (confirmation-16, 116 frames, 393 GT boxes).**
   V4 collapses to MOTA −252.4 with 918 ghost-clutter boxes; shared static
   gets −49.6; V6-TF gets +3.6 with 99 ghost boxes. In a near-empty scene
   the pooled history holds almost no foreground mass, so the nested split
   admits few primaries (F-D).
2. **RT-DETR uav0000273_00001 (confirmation-16, dense).** V6-TF MOTA −1.5
   vs V4 12.5, but V6-TF has 68% more true positives, and HOTA is higher for
   V6-TF. The extra FP are partly unevaluated objects (F-E) and partly dense
   clutter (F-F).
3. **RT-DETR uav0000263_03289 (development-40).** V6-TF −46.7, V4 −8.3,
   shared static −50.9. Mostly F-E: 2,224 boxes on ignored regions,
   occluded cars and excluded classes.
4. **E41 on RT-DETR uav0000263 / 264 / 218 (development-40).** MOTA −207 /
   −164 / −110 through F-A + F-B + F-C together. This is the audit that
   rejected E41 (FX-18).
5. **UAVDT M1007 / M1101 (YOLOv8n).** V6-TF −28.4 / −25.8, V4 −26.5 / −19.7,
   shared static +2.6 / +19.4. Detector precision is ≈35–43%, so the
   conservative fixed threshold wins by outputting little (F-G).

## 3. Limitations (to be stated in the paper)
- Aerial evaluation uses COCO-pretrained detectors, not VisDrone-trained
  ones. Absolute numbers are far below VisDrone-trained trackers and must not
  be compared with them.
- The official-compatible evaluator is a Python port of the Task-4b MATLAB
  toolkit rules. It is not the official server, and HOTA is not part of the
  official toolkit.
- All quality results come from Mac-MPS caches. The T4 fidelity gate is
  deferred; prior E05 evidence showed that MPS caches reproduce CUDA metrics.
- Pre-detector (resolution) scene adaptation failed (FX-17). The only
  scene-state control kept is motion-conditioned association, a small effect.
- V6-TF was designed on val-7, which V4 was also tuned on. The generalisation
  claims rest on confirmation-16 (one-way), BoT-SORT, Faster R-CNN, UAVDT and
  test-dev (post-hoc).
