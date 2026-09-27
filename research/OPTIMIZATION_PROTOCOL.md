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

### Amendment 5b — S3 attempt 1 failed the adoption rule; attempt 2 (disclosed)
Attempt 1 (window cost FP+FN+IDSW, MOTA-aligned): outer-CV ½(HOTA+IDF1)
vs V4: YOLO −2.42 [−4.86, −0.61], RT-DETR −2.64 [−8.30, +0.57]; MOTA +0.1/+0.7,
IDS −38%/−36%, recall −3.8/−3.6. NOT adopted. Diagnosis: the learning
cost was MOTA-aligned while the adoption criterion is HOTA/IDF1.
Gate τ cue was unstable across folds (4 different answers) → τ is not
adapted in any later attempt.
Attempt 2 (the only further attempt on these outer folds): identical nested
procedure, window cost = (FP+FN) + (IDFP+IDFN), the latter from the global
optimal identity mapping (same definition as motmetrics IDF1; verified).
Targets: sensitivity, association offset (τ fixed at 0.75). Same adoption
rule. Because the outer folds were already seen once, attempt 2's outer-CV
result is reported with this disclosure; the clean confirmation of any
adopted V5 is the untouched Faster R-CNN / UAVDT / BoT-SORT transfer.

### Amendment 5c — attempt 2 not adopted; V5 development moves to VisDrone-MOT-train
Attempt 2 outer-CV ½(HOTA+IDF1) vs V4: YOLO −2.26 [−4.77, −0.38]; RT-DETR
−0.60 [−2.15, +1.10] (MOTA +2.8). NOT adopted; no further attempt on the val
outer folds. Diagnosis: S1 headroom is real and the sensitivity cue family
is stable (detector-output cues 6/7 folds), but 7 sequences (~190 windows)
are too few to learn the mapping — S2 stumps beat the global value on the
training folds 7/7 yet lose on held-out sequences (overfitting by scarcity).
Decision: V5 is developed on VisDrone2019-MOT-train (56 sequences; never
used by any system; detectors are COCO-trained, so these videos are unseen
data for the policy). VisDrone val becomes V5's CONFIRMATION set against V4
(V4 was selected on val, i.e. the comparison favours V4). UAVDT test,
Faster R-CNN and BoT-SORT remain untouched transfer tests; VisDrone test-dev
is post-hoc only for V5.
The S1/S2/S3 procedure is re-run unchanged on train (outer LOSO over train
sequences, both detectors, cost (FP+FN)+(IDFP+IDFN)); the adoption rule of
Amendment 5a is applied on the val confirmation set.

### Amendment 5d — V5 protocol on VisDrone-MOT-train (declared before any V5 result on train)
Correction: VisDrone val is NOT an independent confirmation set for V5 (two
V5 attempts were evaluated on it and influenced this redesign). Val is a
SECONDARY check only. Clean tests: Faster R-CNN, BoT-SORT, UAVDT.

Split (research/TRAIN_SPLIT_V5.json, tools/make_train_split.py, seed
20260927, stratified by sequence-length quartiles, metadata only):
  development = 40 sequences (17,167 frames) — all fitting/selection;
  confirmation = 16 sequences (7,034 frames) — untouched until V5 freeze.
Frames are never split.

On the development subset only (YOLOv8n + RT-DETR-L jointly, one shared
controller, no detector/tracker-specific values):
 S1 fixed-value runs around V4 (736): sensitivity, association offset,
    gate τ, retention; per-frame cost (FP+FN)+(IDFP+IDFN). A target is a
    candidate for adaptation only if its per-sequence headroom is ≥ 0.5
    points of #GT for both detectors (else it stays at the V4 value).
 S2 cue utility with sequence-level 5-fold CV inside development (seeded),
    permutation null (200); eligible cue = gain > null95 and > 0 for both
    detectors.
 S3 controller families (declared): C1 single-cue stump per target;
    C2 depth-2 tree per target over eligible cues (min leaf = 10% of the
    training windows); C3 Optuna-TPE linear scene score (non-negative
    weights on development-quantile-normalised eligible cues, thresholds to
    value levels; seed 0; SQLite outputs/v5/optuna_v5.db; 150 trials per
    fold) — C3 is included ONLY if Optuna is installed before S3 starts.
    Family choice and stability: outer 5-fold sequence CV inside
    development; inner selection on 4/5; end-to-end replays of held-out
    folds; lexicographic: (1) fewest catastrophic (MOTA<0) cells,
    (2) worst-detector relative gain of ½(HOTA+IDF1) over V4 on the same
    sequences, (3) lower complexity. A target whose selected cue differs
    across the majority of outer folds is not adapted.
    Resolution: a secondary budgeted variant (V5-R) allocating 832/640 by
    the learned demand with mean pixel cost ≤ that of V4_736 (0.78), compared
    at matched compute; the primary V5 runs at 736 (exactly V4's compute).
 Final V5 = the selected procedure fitted on all 40 development sequences;
 committed and tagged universal-acmot-v5-scene-generalized-freeze BEFORE the
 confirmation subset is evaluated.
Confirmation (once): 16 sequences, V5 vs V4 at 736; adopted iff
 (1) catastrophic cells V5 ≤ V4 and (2) ½(HOTA+IDF1) V5 ≥ V4 for both
 detectors (pooled); paired sequence bootstrap reported.
If not adopted, V4 remains the final system and V5 is reported as a
documented negative result. After the decision: secondary check on val,
then Faster R-CNN, BoT-SORT, UAVDT transfer and official T4 timing for the
final system; no tuning is reopened after any of these.

### Amendment 5e — C3 specification (Optuna installed; declared before S3 runs)
C3 per adapted target: z = Σ w_c·u_c / Σ w_c over the inner-fold eligible
cues, u_c = cue mapped through 21 quantile knots of the TRAINING windows
(optionally inverted, 1−u_c); decision = high value if z > θ else low value.
Search space: w_c ∈ [0,1], orientation ∈ {+,−}, θ ∈ [0,1], low/high ∈ the
target's value grid. Objective: total training-window regret with the same
cost as C1/C2 ((FP+FN)+(IDFP+IDFN)). Optuna 5.0.0, TPESampler(seed=0),
150 trials per study, persistent storage sqlite outputs/v5/optuna_v5.db,
one study per (target, fold) named v5_C3_<target>_<fold-seed>; the search
space is stored in each study's user attributes. The best trial of each
study is used as-is (no manual selection). Family choice among C1/C2/C3
follows Amendment 5d (outer-fold lexicographic rule).

### Amendment 5f — cross-hardware cache-fidelity gate (declared before any V5 result)
Development caches for V5 are built on Mac/MPS. Before the V5 freeze, a
gate compares Mac-MPS and Colab-T4-CUDA caches of the SAME frozen models and
settings on a fixed representative subset: 4 development sequences, one per
length quartile (seed 5): uav0000020_00406_v, uav0000315_00000_v,
uav0000316_01288_v, uav0000342_04692_v; YOLOv8n and RT-DETR-L at 736.
(Confirmation sequences are never used for the gate.)
Measured: per-frame detection counts; one-to-one IoU matching (Hungarian)
of detections — match rate, median/5th-percentile IoU, |Δscore|, rank
(Spearman) agreement of scores, class agreement; replayed tracks and
MOTA/HOTA/IDF1/IDS for V4 and for the candidate V5 controller; agreement of
V5 per-frame decisions.
PASS iff, for both detectors: ≥ 99% of detections matched with IoU ≥ 0.95,
class agreement ≥ 99.5%, median |Δscore| ≤ 0.01; and for V4 and candidate
V5 pooled over the subset |ΔHOTA|, |ΔIDF1|, |ΔMOTA| ≤ 0.5 and |ΔIDS| ≤ 5%;
V5 per-frame decision agreement ≥ 95%.
If FAIL: the development caches are rebuilt on T4 and S1–S3 re-run on them
before any freeze. Prior evidence (E05): MPS-cache replay reproduced the
Colab/CUDA V1/V2b/V2c val metrics exactly (RT-DETR one row within ~1 FP).
Official timing (detector, tracker, AC overhead, total, P95, FPS, GPU memory)
is T4 only.

### Amendment 5g — status of gate thresholds and pre-freeze use of transfer components
* The Amendment-5f fidelity-gate thresholds are PROTOCOL CONSTANTS, not model
  parameters. They are applied exactly as declared and are never relaxed or
  re-interpreted after the gate result is observed.
* Before the V5 freeze, Faster R-CNN and BoT-SORT may be used ONLY for
  (a) latency/throughput measurement (tools/t4_benchmark.py computes no
  tracking metrics) and (b) detection-level cache fidelity (tools/
  fidelity_gate.py compare_det: counts, IoU, scores, classes). No tracking-
  quality metric (MOTA/HOTA/IDF1/IDS/FP/FN/precision/recall) is computed for
  them, and no design or tuning decision may use their timing or fidelity
  output other than the declared gate PASS/FAIL. Their performance
  evaluation is strictly post-freeze, under the transfer locks.

## Amendment 6 (2026-09-27) — HARD REQUIREMENT: training-free online self-calibrating AC (V5-TF)
Owner requirement: the deployed AC layer must be training-free, online
self-calibrating, causal, plug-and-play, detector- and tracker-agnostic and
real-time. No offline-fitted controller (stumps, trees, regressions, Optuna
best-trial weights), no dataset/detector/tracker-specific tuning, no labels,
no GT at deployment. Allowed constants: A structural/mathematical,
B online data-derived, C generic engineering safety bounds, D API-required.
"Worked best on VisDrone" constants (E) are not allowed in V5-TF.
Consequences for V4's VisDrone-selected values (τ 0.75, s 0.4, association
offset 0.10, NMS 0.45, tracker 45/0.86): they are category E under this rule
and are NOT used by V5-TF. V4 remains a reference/ablation.
S1/S2/S3, C1–C3 and Optuna remain DISCOVERY tools only (headroom, cue
utility, rule stability, sensitivity). No final fit on the 40 development
sequences; the S3 learned controller is only a research upper bound (D).
No V5 result on train existed when this amendment was written (only
YOLOv8n train caches 35/56).

V5-TF core (declared before any result):
 * Normalisation: order-only causal ECDF (V3/V4, invariant).
 * Candidate handling by 3-class Otsu on candidate LOGITS pooled over the
   causal window (the same 10-frame memory as the other stream statistics;
   frames < t): thresholds T1 < T2 maximise between-class variance; bin
   edges span the window's own logit range (affine-equivariant ⇒ exactly
   invariant to Platt/temperature recalibration). logit ≥ T2 → primary
   (association + birth); T1 ≤ logit < T2 → extend-only (secondary
   association); < T1 → discarded. Three classes mirror the tracker's
   high/low/discard structure (structural, A). First frames (empty window):
   within-frame Otsu.
 * Tracker: native defaults (ByteTrack/BoT-SORT: buffer 30, match 0.8,
   low 0.1); class bands passed as scores: primary 0.5+0.5u, secondary
   0.1+0.4u (u = ECDF rank), association = birth threshold 0.5 (band
   boundary, A).
 * Detector suppression: detector-native (YOLO 0.7, Faster R-CNN 0.5,
   RT-DETR none) (D).
 * Resolution: compute budget (V4 latency controller); evaluated at 736.
 * Online scene-state vector (robust z vs. own causal history — rolling
   median/MAD): density (primary count), size (median log area fraction),
   score reliability (Otsu separability η), motion (global motion ratio to
   its rolling median), association (track survival). Logged for all
   families; used by rules only where declared.
Candidate rule families (validated, not fitted, on the 40 development
sequences):
 F1  Otsu-3 bands (window) — core only.
 F2  F1 with within-frame Otsu (frame t's own candidates) instead of window.
 F3  F1 + motion-aware association: match_t = min(0.95, 1 − (1 − m0)/max(1, r_t)),
     r_t = motion / rolling median motion (m0 = tracker-native 0.8; 0.95 =
     safety bound C).
 F4  F1 + leader-relative demotion kept (z-logit gate with τ derived online
     as the Otsu T2 expressed in z units — no fixed τ).
Family choice on the 40 development sequences (both detectors, 736):
 lexicographic — (1) fewest catastrophic (MOTA<0) cells, (2) worst-detector
 relative ½(HOTA+IDF1) gain vs V4, (3) simplicity (F1 < F2 < F3 < F4).
Comparison set at matched compute (736): A static (default & shared-static),
B legacy SCI (V3), C V4 compute-only, D S3 learned controller (research
upper bound only), E V5-TF.
Freeze (tag universal-acmot-v5tf-freeze) only after: rule design fixed and
the Amendment-5f T4 fidelity gate passed. Then the 16 confirmation
sequences once: V5-TF vs V4 with the Amendment-5a rule. If V5-TF does not
beat V4, that is reported as the result ("training-free adaptation matches/
does not match tuned V4"); no switch to a trained controller. Then val
(secondary), Faster R-CNN, BoT-SORT, UAVDT, official T4 timing — no
retuning after any of them.

## Amendment 7 (2026-09-27) — V5-TF is THE final target; scene-state control; constant audit (declared before any V5-TF result)
Owner direction: the final research target and contribution is V5-TF
(training-free, online self-calibrating, scene/state adaptive, detector- and
tracker-agnostic, plug-and-play, real-time, causal). V4 is historical
evidence, an ablation and a comparison baseline ONLY — never a fallback final
system. Weaknesses of V5-TF are fixed in V5-TF itself, scientifically.
State at declaration: no V5-TF result exists (outputs/v5tf_dev absent; train
caches YOLOv8n 0.45 56/56, RT-DETR-L 26/56, YOLOv8n native 0/56).

1. Replaces the system-selection clauses of Amendments 5a/5d/6 ("adopted iff …
   otherwise V4 remains final"): the confirmation-16 run (once, after freeze)
   REPORTS V5-TF vs V4 (catastrophic cells, ½(HOTA+IDF1) per detector, paired
   sequence bootstrap). It does not choose the final system. If V5-TF is worse,
   that is reported as a limitation of V5-TF; any post-confirmation fix is a new
   declared revision whose evidence needs new clean data (never confirmation-16
   again).
2. F4 is removed from the Amendment-6 families (a z-gate with τ = Otsu T2 in z
   units is identical to F1; commit 71faf44). Simplicity order F1 < F2 < F3 < F5.
3. Scene-state vector (all causal: state for frame t from frames < t, except the
   image motion of frame t vs t−1, which is available before detection):
   size s = median log(box area / image area) of primary-band candidates;
   density = log(1 + #primary); reliability η = Otsu separability; motion
   ratio r = motion / rolling median; association = track survival
   |ids_t ∩ ids_{t−1}| / |ids_{t−1}|. Computed in the V5-TF path
   (online_calibration.py), NOT by scene_state.py (so Z_REF and DECAY are not
   part of V5-TF). Components without a declared rule are logged only.
   Retention is not adapted (S1: retention headroom ≈ 0, E32).
4. New declared rule R-res (control BEFORE the detector, compute budget B =
   the deployment level, 736 in experiments; levels {640, 736, 832}, D):
   ℓ = the level used at frame t−1; s̃ = median of s over the last 10 frames
   processed AT LEVEL ℓ (the Otsu memory); p_t = causal ECDF rank of s̃ among
   the stored history of s̃ observed AT LEVEL ℓ (RobustHistory window).
   Conditioning on the level removes the self-induced bias of the action
   (higher resolution detects smaller objects, lowering s; found by reasoning
   and a metric-free crash test on one val cache, before any V5-TF result).
   p_t < 1/3 → 832 (objects small relative to this stream's own history),
   p_t > 2/3 → 640, else B; history of ℓ not warm → B. Tertiles = three equiprobable bands for
   three levels (A).
   The decision is taken at the start of each block of 10 frames (the Otsu
   memory; same block length as the F5R control) and held within the block —
   a per-frame version flickered (≈125 switches / 200 frames in the crash
   test), which would confound tracking and make F5R an unfair control.
   Budget guard (C, safety): at each block start, if the running mean pixel
   cost (level²) so far ≥ B², a request for 832 is served at B. Warm-up (< warm-up
   samples) → B.
   Families added: F5 = F3 + R-res. Control (ablation, not selectable):
   F5R = F3 + random level (uniform over the three levels, redrawn every 10
   frames, fixed seed) with the same budget guard — tests whether the size
   state beats random allocation at matched compute (E26 design).
5. Final-family requirement: the frozen V5-TF must contain scene-state control,
   so the selectable families are F3 and F5; F1/F2 are reported as ablations
   (candidate handling without scene control). Choice between F3 and F5:
   (1) fewest catastrophic cells, (2) worst-detector relative ½(HOTA+IDF1)
   vs V4 on the same sequences, (3) simplicity. Mean pixel cost is reported per
   detector; F5 must satisfy mean cost ≤ 1.01·B² to be selectable.
6. Constant audit on the chosen family (development-40, both detectors,
   ByteTrack): OTSU_BINS {32, 64, 128}; Otsu window {5, 10, 20}; RobustHistory
   window {50, 100, 200}; warm-up {3, 5, 10}; one-at-a-time around the declared
   defaults (64, 10, 100, 5). Criterion (E28 precedent): pooled |ΔHOTA| ≤ 0.4 per
   detector and no additional catastrophic cell → value reclassified as
   structural with insensitivity evidence (A). Otherwise the declared default is
   KEPT and reported as a sensitive category-E limitation — never replaced by
   the best-scoring value.
7. Further rule revisions on development-40 are allowed only as new families
   committed to this protocol before they are run; every attempt is reported;
   no parameter search. Val, confirmation-16, Faster R-CNN, BoT-SORT, UAVDT and
   test-dev remain as in Amendments 5d/5g/6.
