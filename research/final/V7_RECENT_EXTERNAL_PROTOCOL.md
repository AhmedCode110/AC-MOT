# V7 recent external systems — protocol (preregistered before any run)

Written and committed BEFORE any baseline or AC-MOT run on the systems below.
Nothing in this file may be changed after the first result except by
appending a dated amendment that states what changed and why; the original
text stays.

## 1. Policy under test
- `universal-acmot-v7-freeze` → 488df9a (V7f), `configs/universal_acmot_policy_v7.json`.
- Every run first checks `research/V7_POLICY_LOCK.json` by sha256 (10 files);
  a mismatch aborts the run.
- No change to thresholds, ρ logic, Otsu logic, history windows, motion rule,
  duplicate policy, host anchoring, score calibration. No tracker, paper or
  dataset names reach the layer. Only format adapters are written.

## 2. Selection criteria (fixed before results)
A system enters the main external table only if all hold:
1. peer-reviewed publication 2024–2026 (journal or CVPR/ICCV/ECCV/AAAI/ICLR level);
2. official public code and official weights or released detections;
3. evaluation on a benchmark with public GT (MOT17 val-half, MOT20 val-half,
   DanceTrack val, KITTI tracking training/val);
4. a detector-output boundary where a candidate filter can sit without
   retraining or changing the learned/association logic;
5. never used in AC-MOT development (excluded: SparseTrack, BoostTrack,
   ByteTrack, OC-SORT, PD-SORT, Hybrid-SORT, BoT-SORT);
6. baseline reproduction classified EXACT or CLOSE (section 4).
Systems are not chosen or dropped because of their AC-MOT outcome. Every
system that passes 1–6 is reported, whatever its delta.

## 3. Candidates and order (audit of the official repositories, 2026-09-28)
| Order | System | Venue | Repository @ audited commit | Boundary | Assets | Datasets planned |
|---|---|---|---|---|---|---|
| 1 | TOPICTrack | IEEE TIP 2025 | holmescao/TOPICTrack @ e7b260f | YOLOX output → `OCSort.update` (Deep-OC-SORT family, embeddings from FastReID) | detector `topictrack_ablation.pth.tar`, `topictrack_mot17.pth.tar`, ReID `mot17_sbs_S50.pth`, `mot20_sbs_S50.pth` (Google Drive, fetched by GitHub Actions) | MOT17 val-half, MOT20 val-half |
| 2 | C-TWiX | Pattern Recognition 160 (2025) 111169 | Guepardow/TWiX @ 3cff9cc | released detection files → `Cascade_TWiX` per frame | detections `results.zip`, weights `weights.zip` (mehdimiah.com, fetched by GitHub Actions) | MOT17 val-half (CenterTrack split), KITTIMOT val, DanceTrack val |
| 3 | TrackTrack | CVPR 2025 | kamkyu94/TrackTrack @ ee7f1c5 | detection pickles → tracker | released detection pickles, ReID weights (Drive); GMC files in repo | MOT17 val-half, DanceTrack val |
| 4 | LG-MOT | IEEE TCSVT 2025 | WesLee88524/LG-MOT @ b5ee9d4 | offline graph (SUSHI family), CUDA; README reports test sets only | checkpoints (Drive) | audit only unless a verifiable val reproduction exists |
| 5 | MOTIP | CVPR 2025 | MCG-NJU/MOTIP @ ffc0e90 | DETR detections → ID decoder (det/newborn thresholds) | checkpoints | architecture audit first |
| – | CAMELTrack | arXiv 2505.01257 (peer review not verified) | TrackingLaboratory/CAMELTrack @ 46a74bb | – | HF weights, states | exploratory only, never in the main table |

## 4. Reproduction classification
Reference = the authors' reported number for the same split and setting
(paper or official README), recorded before running.
- **EXACT**: |ΔHOTA| ≤ 0.2 and |ΔMOTA|, |ΔIDF1| ≤ 0.3, or the authors' released
  outputs reproduced byte- or metric-identically.
- **CLOSE**: |ΔHOTA| ≤ 1.0 with a documented, non-tuned cause (e.g. CPU fp32
  instead of GPU fp16, library version).
- **BLOCKED**: an asset or step cannot be obtained or run.
- **INCOMPATIBLE**: no scientifically clean detector-output boundary.
- anything else: **FAILED** (reported in `V7_RECENT_EXTERNAL_FAILURES.md`, not
  in the main table).
AC-MOT is run only after the baseline is classified EXACT or CLOSE. Both arms
always come from the same environment and the same detections.

## 5. Host contracts (derived from the published configuration, not from results)
Same rule as for every earlier host: assoc = first-association score threshold;
birth = lowest score from which a new track can start; low = lowest score of
any association stage (low ≥ min(assoc, birth) marks a single-stage host);
match = 1 − IoU gate where the host has an IoU gate.

| System · setting | assoc | birth | low | match | Mapping of the layer's decision onto the host |
|---|---|---|---|---|---|
| TOPICTrack MOT17 / MOT20 val (`run/mot17_val.sh`, `run/mot20_val.sh`; `args.py` track_thresh 0.6, iou_thresh 0.2; `extract_detections` second band `scores > 0.4`; new tracks from unmatched first AND second-band detections) | 0.6 | 0.4 | 0.4 | 0.8 | kept candidates and their scores → `update`; `det_thresh ← decision.assoc`; `iou_threshold ← 1 − decision.match`; the 0.4 band is hard-coded and is not changed |
| C-TWiX MOT17 (`script/c-twix_MOT17.sh`: min_score 0.50, min_score_new 0.70) | 0.5 | 0.7 | 0.5 | 0.8 (host has no IoU gate; not mapped) | kept candidates and scores → frame's ObsCollection; `min_score ← decision.assoc`, `min_score_new ← decision.birth` |
| C-TWiX KITTIMOT (`c-twix_KT.sh`: 0.50 / 0.50) | 0.5 | 0.5 | 0.5 | 0.8 (not mapped) | same |
| C-TWiX DanceTrack (`c-twix_DT.sh`: 0.50 / 0.90) | 0.5 | 0.9 | 0.5 | 0.8 (not mapped) | same |

Host-side filters that follow the layer stay as published (TOPICTrack
aspect-ratio 1.6 and min area 10 on outputs, interpolation for `_post`;
C-TWiX min_area 128 and class filter).

## 6. Controller inputs
- Candidates: every detection the host's detector emits for the frame, in
  original image coordinates, with the detector score.
- Image cue: the V7 motion cue is computed from the frame only when the host
  pipeline itself reads frames (TOPICTrack). Coordinate-only hosts (C-TWiX)
  pass no image, as for PD-SORT.
- Track feedback: the host's confirmed outputs of the frame (`observe`).

## 7. Evaluation
- TrackEval 12c8791; MotChallenge2DBox (MOT17, MOT20, DanceTrack) and
  Kitti2DBox (KITTIMOT) with the split and GT each system's own protocol
  defines (TOPICTrack ships MOT17-val GT; C-TWiX uses the CenterTrack half
  split and the KITTIMOTS val split).
- Metrics: HOTA, DetA, AssA, MOTA, IDF1, IDS, FP, FN, Precision, Recall;
  pooled and per sequence.
- Paired sequence bootstrap of pooled metrics: 10,000 resamples, seed 42,
  percentile 95% CI of Δ = (host + V7f) − host. A difference is called
  significant only when the CI excludes 0.
- Sequence wins / ties / losses on HOTA: tie when |ΔHOTA| < 0.01.

## 8. Predictions (written before any run)
From the mechanism documented in `V7_METHOD.md`:
- TOPICTrack is a two-stage host on a clean YOLOX-X stream → V7f passes
  through: expected |ΔHOTA| ≤ 0.1, CI including 0.
- C-TWiX is a single-stage host → the foreground track-consistent rescue is
  active → expected ΔHOTA ≥ 0 on MOT17 (no numeric claim); no prediction for
  DanceTrack or KITTIMOT beyond "no collapse" (|ΔHOTA| < 1).
These predictions are scored after the runs, whatever the outcome.

## 9. Contamination rule
A system used later to develop V8 becomes DEVELOPMENT for V8 permanently and
cannot count as untouched validation for V8. The V7f results recorded here
are never rewritten.

## 10. Compute
GitHub Actions (ubuntu-24.04, 4 vCPU, no GPU), because the Claude container
cannot reach motchallenge.net, Google Drive or the authors' hosts. Every
workflow records the repository commit, asset sha256, library versions and
CPU, and commits its results to `research/final/recent/<system>/`.

## Amendment 1 (2026-09-28, before any TrackTrack run): TrackTrack contract and adapter
Source: kamkyu94/TrackTrack @ ee7f1c5, `3. Tracker/utils/etc.py::set_parameters`,
`trackers/tracker.py::update`, `1. YOLOX/detect.py` (conf 0.1).

| Setting | assoc | birth | low | match | Mapping |
|---|---|---|---|---|---|
| MOT17 val-half (sequences 02/04/05/09/10/11/13: det_thr 0.60, init_thr 0.70, match_thr 0.70) | 0.6 | 0.7 | 0.1 | 0.7 (mixed IoU/ReID cost; not mapped) | `det_thr ← decision.assoc`, `init_thr ← decision.birth` |
| DanceTrack val (det_thr 0.60, init_thr 0.60, match_thr 0.80) | 0.6 | 0.6 | 0.1 | 0.8 (not mapped) | same |

TrackTrack takes two views of the same detector output per frame (NMS 0.80 =
the candidates, NMS 0.95 = the source of its "deleted detections"). The layer
runs on the NMS-0.80 candidates. In the NMS-0.95 view, a row that matches a
candidate (the tracker's own IoU ≥ 0.97 rule) takes that candidate's fate:
removed if the layer removed it, rescored if the layer rescored it; rows the
tracker would treat as deleted detections are left as they are. The tracker
reads no image (CMC from its shipped GMC files, ReID features precomputed), so
no image cue is passed. Reproduction reference: the paper's MOT17 / DanceTrack
validation numbers, recorded in `V7_RECENT_EXTERNAL_SYSTEMS.md` from the CVF
paper text before the first run.

## Amendment 2 (2026-09-28, before any TrackTrack run): TrackTrack MOT17 split
Amendment 1 named the TrackTrack MOT17 setting "val-half (02/04/05/09/10/11/13)".
The official TrackTrack MOT17 validation protocol is different: sequences
MOT17-04, -05, -09 in full (`3. Tracker/trackeval/seqmap/mot17/val.txt`,
`mot17_val.json`, 2,412 frames), detector `mot17_half.pth.tar`, GT = the
official `train/<seq>/gt/gt.txt`. The contract values are unchanged (these
sequences use det_thr 0.60, init_thr 0.70, match_thr 0.70). Only three
sequences exist, so the MOT17 bootstrap has three resampling units; it is
reported but not used to call significance on its own. The released MOT17
NMS-0.80 detections are absent from the official folder (only NMS 0.95 is
released), so the official detector is re-run for both views and its NMS-0.95
output is compared with the released file as the detector reproduction check.
