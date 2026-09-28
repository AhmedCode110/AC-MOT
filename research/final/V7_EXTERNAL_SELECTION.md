# V7 EXTERNAL SELECTION — PREDECLARATION (closed in the V7 freeze commit)

Frozen policy: `configs/universal_acmot_policy_v7.json` (system V7f), lock
`research/V7_POLICY_LOCK.json`, tag `universal-acmot-v7-freeze`. This file is
part of the freeze commit: the systems, artefacts, protocol and reported
quantities below are fixed BEFORE any AC-MOT run on them. No AC-MOT parameter
is changed after seeing their results, and every outcome is reported.

Question: what happens when the SAME frozen Universal AC-MOT layer is added,
without retuning, to an already-strong published MOT pipeline?

## Selection rules
- recent strong publication (2025/2026 preferred; Q1/Q2 journal or top venue);
- official code + the tracker-input artefacts reachable from this environment
  (GitHub, PyPI, official KITTI S3; motchallenge.net, Google Drive, arXiv,
  Hugging Face, Zenodo and several author sites are unreachable here);
- architecture where AC-MOT sits between the published detector output and the
  published tracker input, the published tracker/detector/thresholds/evaluator/
  split untouched;
- not a V7 development system; not selected because AC-MOT helps it.

## Predeclared systems
### S1 — PD-SORT (Wang et al., IEEE Transactions on Consumer Electronics, 2025)
- Code: github.com/Wangyc2000/PD_SORT @ af21db6 (OC-SORT-based; pseudo-depth
  DVIoU + QPDM; CMC from the shipped BoT-SORT GMC files; no ReID).
- Tracker input: the authors' released MOT17 val-half detections
  (`res_mot/MOT17-val/yolox_x_ablation_results/<run>/MOT17-*_detections.txt`,
  YOLOX-X `ocsort_mot17_ablation`, conf 0.1, NMS 0.7, test size 800×1440;
  identical across the released runs).
- Baseline: the repository's code with its README MOT17-val arguments
  (track_thresh 0.6, iou_thresh 0.3, delta_t 3, inertia 0.2, asso iou,
  use_byte False, min_box_area 100, aspect 1.6, CMC method "file"), frames
  replaced by their shape only (the tracker reads pixels only through CMC,
  which is served from the shipped files — verified by reproducing the
  authors' released outputs).
- Reference "paper final result": the paper PDF host is unreachable; the
  authors' own released MOT17-val outputs + TrackEval summaries in the
  repository are used as the published reference, and the run folder whose
  outputs our reproduction matches is reported.
- Host contract (documented tracker properties): single-stage, assoc = birth =
  low = 0.6, match = 1 − 0.3 = 0.7. Controls driven: det_thresh, iou_threshold.
- Motion cue: unavailable (frames unreachable) → the V7 motion rule is inactive.

### S2 — Hybrid-SORT (Yang et al., AAAI 2024) — secondary
Included because the 2025/2026 alternatives found are not reproducible here
(TrackTrack, CAMELTrack, TOPICTrack need frames/ReID; C-TWiX artefacts are on
a blocked host; WM-SORT / KalmanFormer have no official code).
- Code: github.com/ymzis69/HybridSORT @ 396f8d3, configuration
  `exps/example/mot/yolox_x_ablation_hybrid_sort.py` (no ReID; use_byte True;
  det_thresh 0.6; iou_thresh 0.25; Height-Modulated IoU; TCM both steps;
  inertia 0.05).
- Tracker input: the same YOLOX-X `ocsort_mot17_ablation` MOT17 val-half
  detections (conf 0.1, NMS 0.7, 800×1440) as released with S1 (Hybrid-SORT's
  README uses fp32, the released detections are fp16: any deviation from the
  reported numbers is reported).
- Reported (README, MOT17-half-val): HOTA 67.1, IDF1 78.0, MOTA 75.8.
- Host contract: two-stage, assoc = birth = 0.6, low = 0.1, match = 0.75.

### Reserved / infeasible
- TOPICTrack (IEEE TIP 2025): stays untouched; infeasible here (YOLOX +
  FastReID on frames).

## Protocol (both systems)
Evaluator TrackEval @12c8791, MOT17 val-half GT (BoostTrack `results/gt`).
Report for PAPER REFERENCE vs OUR REPRODUCTION vs REPRODUCTION + FROZEN V7:
HOTA, MOTA, IDF1, IDS, FP, FN, Precision, Recall, AssA, DetA; per-sequence
results; Δ; 10,000-sample paired sequence bootstrap (seed 42) with 95% CIs;
sequences improved/degraded; failure cases. The MOT17 frames are unreachable,
so the V7 image motion cue is absent in both systems (stated with every
number).
