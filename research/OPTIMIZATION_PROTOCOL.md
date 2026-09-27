# Global parameter selection protocol (declared BEFORE running, 2026-09-27)

## Question
Which shared global setting of the Universal AC-MOT candidate controller
gives the most robust tracking across detector families, without any
detector-specific parameter?

## Fixed before running
* Development data: VisDrone2019-MOT-val, 7 sequences. Test-dev untouched.
* Detectors seen jointly by the optimizer: YOLOv8n, RT-DETR-L.
* Tracker: ByteTrack (legacy AC-MOT settings buffer 45, match 0.86).
* Evaluator: reference protocol (HOTA TrackEval; rest motmetrics).
* Search space (28 configs, full grid — low-dimensional and cheap, so a
  full grid gives the complete response surface; no adaptive search needed):
  * architecture ∈ {gate-only (V1 base), gate + density Top-K (V2cA base)}
  * leader-relative gate ρ ∈ {0 (off), 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}
  * density budget multiplier κ ∈ {0.5, 1, 2} (scales 12/36/12/48)
* Constraint (catastrophe guard): overall MOTA > 0 for EVERY detector.
* Primary objective J2: maximize the worst detector's ½(HOTA + IDF1).
  Rationale: the claim is universality, so no detector may be sacrificed;
  HOTA and IDF1 are the tracking-centric metrics; MOTA enters as a guard
  because it is dominated by raw FP/FN counts.
* Comparison objectives (reported, not used for the final choice):
  J1 = mean over detectors of ½(HOTA + IDF1);
  J3 = mean relative gain of ½(HOTA + IDF1) over V1 (ρ=0, gate-only).
* Validation: leave-one-sequence-out (7 folds). Each fold selects on 6
  sequences × 2 detectors and scores the held-out sequence. Pooled
  held-out metrics = CV estimate of the whole selection procedure.
  Final parameters = the same procedure applied to all 7 sequences.
  Stability = agreement of fold choices with the final choice.
* Frames are never split; sequences are the unit.

## Not tuned here (documented as inherited AC-MOT defaults; sensitivity
analysed separately): normalizer bins/decay/update period, generic
sensitivity formula, resolution controller, ByteTrack buffer/match,
leader EMA decay 0.9.

## Amendment 1 (2026-09-27, after the grid, BEFORE computing J4)
Observed: J2 (worst-detector absolute ½(HOTA+IDF1)) selected gate_r0.4,
whose RT-DETR MOTA is only 3.3. Structural reason: YOLOv8n is uniformly the
weaker detector in absolute terms, so min() is always attained by YOLO and
RT-DETR never enters the objective — max-min over unnormalized metrics
degenerates to single-detector optimization when detector capacities differ.
Declared now (before computing it): J4 = min over detectors of the relative
gain of ½(HOTA+IDF1) over each detector's own V1 on the same sequences
(scale-normalized max-min). Decision rule: if the two scale-normalized
objectives (J3 mean relative gain, J4 worst relative gain) select the same
config, that config is final; otherwise J4 (more conservative) is used.
J2 results are reported unchanged as the original primary objective.

## Amendment 2 (2026-09-27, after viewing the response surface)
Disclosure: the full 7-sequence response surface had been printed before
this amendment. Reason: the user's requirements (stated before this
protocol) say a solution that collapses on one detector/sequence is
unacceptable; the original constraint (overall MOTA > 0 per detector) did
not encode that. J4's choice (gate_r0.4) has RT-DETR MOTA −45/−25/−9 on
three sequences. Definition with no new tuned number: a (detector,
sequence) cell is catastrophic if MOTA < 0 (worse than an empty tracker,
whose MOTA is exactly 0). Selection becomes lexicographic: (1) minimise
the number of catastrophic cells on the selection sequences, (2) maximise
J4. Applied identically inside every LOSO fold (training sequences only).
Both the Amendment-1 result and this result are reported.

## Amendment 3 (2026-09-27) — V3 calibration-invariant family (before any test-dev metric)
Trigger: score-calibration stress test on val (E20) showed the v1 freeze
(hist normalizer + raw ratio gate) is not invariant to temperature
recalibration (YOLO HOTA 32.1→25.7 at T=0.5; RT-DETR MOTA 24.1→7.9 at T=2),
while a shared static raw threshold collapses under scale ×0.5.
Design requirement declared: decisions must be exactly invariant to the
Platt/temperature family logit' = a·logit + b (a>0). V3 = order-only ECDF
normaliser + zlogit gate (demote) + no raw floor in the policy; exact
invariance verified (byte-identical tracks under T=2 and T=0.5).
Selection family (only these are selectable): V3 gate-only,
τ ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}. Same lexicographic rule (fewest
catastrophic cells, then J4; J3 reported), same LOSO folds.
Ablations (not selectable): ECDF + ratio ρ ∈ {0.5, 0.6, 0.7}.
The v1 tag (universal-acmot-v1-freeze) is superseded; kept as history.

## Amendment 4 (2026-09-27) — whole-controller generalization audit → V4 (before any test-dev metric)
Evidence (val only): E24 controller ablation — the legacy SCI resolution
rule is no better than a uniform 640/736 mix at equal pixel cost, and the
SCI→sensitivity mapping equals a constant 0.4. E25 per-frame audit — no
causal scene cue predicts the benefit of 832 over 640 consistently (|ρ|<0.3
within sequences, signs flip across sequences and detectors). E26 matched-
compute end-to-end test — no single cue (crowd, tiny, area, edges,
darkness, blurriness) nor the legacy SCI beats RANDOM 50% allocation on both
detectors (selection frequency 0/2 for every cue). Decision: remove the
scene controller from the decision path. Resolution is a compute-budget
input (fixed per deployment; latency-feedback controller for live use).
V4 search space (shared by both detectors, resolution 736 for selection):
  τ ∈ {0.5, 0.75, 1.0, 1.25, 1.5}; sensitivity s ∈ {0.3, 0.4, 0.5, 0.6};
  association offset ∈ {0.10, 0.18, 0.26}; birth offset ∈ {0.0, 0.05, 0.10};
  tracker retention/match ∈ {legacy AC (45, 0.86), tracker-native (30, 0.8)}.
Selection: nested sequence-level LOSO; in each outer fold the config is
chosen on the 6 training sequences × 2 detectors by the lexicographic rule
(1) fewest catastrophic (MOTA<0) cells, (2) max J4 (worst-detector relative
gain of ½(HOTA+IDF1) over V1). Outer held-out sequences give the CV
estimate. Stability: per-parameter fold choices, median, range, frequency;
objective sensitivity around the optimum. A parameter whose fold choices
are unstable is not averaged — the mechanism is redesigned or the
parameter is fixed at the value with the flattest objective, and reported.

## Post-freeze audit rule for the NMS request (declared 2026-09-27, before any Faster R-CNN result; V4 is NOT changed by it)
E27 selected 0.45 on YOLO only (RT-DETR is NMS-free). 0.45 may be described
as a shared robust choice only if, for EVERY NMS-bearing detector family,
V4 with 0.45 is not worse than V4 with that family's API-native NMS
(YOLO 0.7, torchvision Faster R-CNN 0.5) on ½(HOTA+IDF1) in ≥ 4/7 val
sequences. RT-DETR counts as "not applicable", not as support. If the rule
fails, NMS 0.45 is reported as a YOLO-derived limitation and adapter-native
suppression is recorded as the recommended design for a future version
(V5), which would require its own development selection and a new held-out set.

## Amendment 5 (2026-09-27) — V5 Generalized Scene-Adaptive Universal AC-MOT
Direction set by the project owner: the contribution is scene-adaptive control
BEFORE and around the detector–tracker chain. E24–E26 only show that the
LEGACY cues/SCI do not predict when higher resolution helps; they do not show
that scene adaptation is useless. V4 (compute-budget only) is kept as an
ablation/alternative branch; V5 is developed separately.

Data status (fixed now, before any V5 work):
* VisDrone test-dev was evaluated once for V4 (E31). It is NOT a clean
  held-out set for V5; any V5 test-dev number will be labelled post-hoc.
* Clean for V5 final evidence (never evaluated by any system): UAVDT test
  split, Faster R-CNN (unseen detector, any dataset). Tracker transfer:
  BoT-SORT (val results for V3 exist; V5 not yet seen).
* V5 development uses ONLY VisDrone val (7 sequences) with YOLOv8n +
  RT-DETR-L jointly. Faster R-CNN, UAVDT and test-dev are never used to
  select cues, weights, controller form, targets or thresholds.

V5 development stages:
 S1 adaptation-value (headroom) analysis per control target (resolution,
    sensitivity, gate τ, association strictness, retention; NMS if pre-NMS
    caches are added): per-window oracle vs best global value, GT offline.
 S2 cue utility per target: image / detector-output / tracker-state cue
    families, LOSO-cross-validated predictive value, detector consistency,
    redundancy, overhead.
 S3 controller: scene-state vector → per-target mapping; cue subset,
    weights, thresholds by Optuna (TPE, fixed seed, SQLite) inside nested
    LOSO with both detectors; lexicographic objective (catastrophic cells;
    robust cross-detector HOTA/IDF1/MOTA; IDS; recall; detector variance;
    complexity). J2 (unnormalised max-min) is not reused.
 S4 formal comparison at matched compute: static / legacy SCI / V4 compute
    only / V5 and V5 family ablations (visual only, detector-feedback only,
    tracker-feedback only, minus each family, adaptive vs fixed per target).
 S5 freeze V5 (tag universal-acmot-v5-scene-generalized-freeze), then
    Faster R-CNN, BoT-SORT, UAVDT transfer; test-dev post-hoc only.

### Amendment 5a — S3 nested procedure and V5 adoption rule (declared before S3 runs)
S1 result: headroom exists for gate τ, association offset, resolution and
sensitivity (per-sequence 0.4–2.3 MOTA pts); retention ≈ 0 → stays global.
S2 result (all 7 sequences, diagnostic only): single-cue signal beyond the
permutation null and positive for BOTH detectors only for detector-output
(det_gap, det_count), tracker-state (trk_survival, trk_match) and motion
(img_motion, img_motion_resp) cues; edges/brightness/blur negative everywhere.
Because S2 saw all 7 sequences, cue choice is REDONE inside every outer fold:
 outer LOSO fold (held-out h): on the 6 training sequences × 2 detectors,
   for each target: inner-LOSO cv_gain and permutation null (100) per cue;
   eligible = cv_gain > null95 and gain > 0 for both detectors; pick the
   eligible cue with max inner cv_gain, fit the cost-sensitive stump on the
   6 sequences; no eligible cue → target stays at its V4 value.
   Cost = window errors FP+FN+IDSW (S1 runs); non-adapted targets at V4
   values; resolution fixed at 736 for the primary (exactly compute-matched)
   comparison; a budgeted resolution variant is secondary.
 End-to-end replay of the fold controller on h (both detectors).
V5 is adopted over V4 iff on the pooled 7 outer held-out sequences:
 (1) catastrophic (MOTA<0) cells V5 ≤ V4, and
 (2) ½(HOTA+IDF1) V5 ≥ V4 for BOTH detectors;
 a paired sequence bootstrap of the difference is reported (not a gate).
Stability: cue chosen per target per outer fold is reported; a target whose
cue differs across most folds is not adapted in the final V5.
