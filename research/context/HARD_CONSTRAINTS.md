# HARD CONSTRAINTS — non-negotiable research identity

Source of authority: `research/OPTIMIZATION_PROTOCOL.md` Amendment 6 (commit
3684684) and Amendments 5d–5g. A constraint here may only be changed by a new
protocol amendment committed BEFORE the result it would affect, plus a new
superseding entry in `DECISIONS.md`. Chat instructions alone do not change it.

## C1 — Final AC layer is training-free (Version: V5-TF)
The deployed AC layer must NOT require, at any point:
training on VisDrone2019-MOT · fitting on the development-40 sequences ·
learned decision stumps / trees · regression or logistic models · learned cue
weights · Optuna best-trial weights · dataset-specific calibration ·
detector-specific tuning · tracker-specific tuning · labeled calibration data ·
GT at deployment. It must start on a new video directly and self-calibrate
online. Detector and tracker stay frozen.

## C2 — Required deployment properties
training-free · online self-calibrating · causal · scene/state adaptive ·
plug-and-play · detector-agnostic · tracker-agnostic · lightweight/real-time ·
no GT at runtime · no dataset-specific deployment calibration.

## C3 — Causality
Decision for frame t uses only frames ≤ t; stream statistics used to decide
frame t prefer frames < t (avoid circular dependence). Never future frames,
never GT, never full-sequence statistics.

## C4 — Allowed final forms (explicit rule logic only)
Percentile bands · monotonic thresholds on online-normalised states · simple
deterministic rule tables · bounded proportional adjustment · hysteresis ·
causal state machines · order/rank statistics (ECDF, Otsu on logits,
rolling median/MAD, robust z, ratios to rolling median).
FORBIDDEN final forms: fitted decision tree, learned coefficients, trained
classifier, neural network, regression fitted on development labels,
"Optuna best trial" as the deployed controller.

## C5 — Constants
Allowed only if category A structural/mathematical, B online data-derived,
C generic engineering safety bound, D API-required. Category E ("worked best
on VisDrone") is not allowed in V5-TF. V4's VisDrone-selected values
(τ 0.75, s 0.4, association offset 0.10, NMS 0.45, tracker 45/0.86) are
category E under this rule and are NOT used by V5-TF. Open V5-TF items still
unjustified (RobustHistory window 100 and warm-up 5, Otsu window 10, OTSU_BINS
64, Z_REF 0.75 / DECAY 0.9 if used by a rule) are listed in PARAMETER_STATUS.md
and block the freeze.

## C6 — Architecture: AC starts BEFORE the detector
Frame → Generic Scene/State Analyzer → Online Self-Calibration → Generalized
Scene State / SCI → Universal AC Controller → Compute/Latency Constraint →
Detector Adapter → Detector → Universal Score/Candidate Handling → Tracker
Adapter → Tracker → Tracks → Causal feedback. Do not reduce the contribution to
a post-detector score filter. V4 compute-only is an ABLATION, not permission
to silently delete scene adaptation.

## C7 — Research tools are discovery-only
S1/S2/S3, C1/C2/C3, Optuna, decision stumps, cue selection and the 40/16 split
may answer: which cues help / are useless, which targets have headroom, which
rules are stable, which ranges are insensitive, which failure modes exist.
They must NOT produce the deployed controller. No final fit on the
development-40. The S3 learned controller is a research upper bound only
(comparison system D), never the method.

## C8 — Protected evaluations (details: PROTECTED_EVALUATIONS.md)
No quality metric (HOTA/IDF1/MOTA/IDS/FP/FN/precision/recall) on the
confirmation-16, Faster R-CNN, BoT-SORT (V5-TF) or UAVDT before the V5-TF
freeze. Caching, detection-level fidelity and timing are allowed where the
protocol says so. After freeze: evaluate once, no retuning.

## C9 — Hardware
Mac/MPS = development only; never report Mac timing. Official timing is Colab
T4 only. Amendment 5f fidelity-gate thresholds are PROTOCOL CONSTANTS; never
relax them after seeing the gate. If the gate fails materially: rebuild
reference development caches on T4; never tune V5-TF to compensate.

## C10 — Runtime cost
Online self-calibration = simple incremental statistics. No Optuna, model
fitting, per-frame optimisation or batch reprocessing at runtime. AC overhead
must be much smaller than detector latency (measured on T4).

## C11 — Honest reporting
If V5-TF does not reliably beat V4 compute-only at matched compute, report
"scene-aware adaptation adds insufficient benefit under the tested
training-free constraints". Never silently switch to a trained controller.
Never claim "works with every detector/tracker": state exactly what was tested
(Detector: YOLOv8n, RT-DETR-L, Faster R-CNN ResNet50-FPN v2; Tracker:
ByteTrack, BoT-SORT).

## C12 — History is immutable
Never overwrite V1, V2*, V3, V4, V5 artifacts, frozen tags, locks, old
protocols, failed-experiment evidence or old results. New work = new named
revision (V5-TF). Changes to decisions = new superseding decision.

## C13 — Anti-hallucination
Never invent metrics, experiment status, commit SHAs, dataset leakage status,
parameter justifications or frozen state. Check git/code → canonical result
files → research/context → Graphify → detailed logs. If still ambiguous, write
UNKNOWN or NEEDS VERIFICATION.
