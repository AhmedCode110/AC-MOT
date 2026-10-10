# HARD CONSTRAINTS — non-negotiable research identity

Source of authority: `research/OPTIMIZATION_PROTOCOL.md` Amendment 6 (commit
3684684) and Amendments 5d–5g. A constraint here may only be changed by a new
protocol amendment committed BEFORE the result it would affect, plus a new
superseding entry in `DECISIONS.md`. Chat instructions alone do not change it.

## C0 — FINAL TARGET = V6-TF (Amendment 9; supersedes the V5-TF/E41 target of Amendment 7)
The final research target and contribution is Version: V6-TF — the
training-free line's successor to V5-TF after the E41 audit (Amendment 9):
training-free, online self-calibrating, causal, detector-agnostic,
tracker-agnostic, plug-and-play, real-time, with scene-state
(motion-conditioned) association control.
Version: V4 is historical evidence, an ablation and a comparison baseline
ONLY. It is NEVER a fallback final system. Do not revert to V4, and do not
present V4 and V6-TF as alternative final systems. V5-TF families (F1–F5,
E41) are recorded development history; E41 is rejected (FX-18), never frozen.
Development of V6-TF iterated on val-7 (Amendment 9). The confirmation-16
comparison against V4 is REPORTED, not a system-selection gate.

## C1 — Final AC layer is training-free (Version: V6-TF)
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

## C3 — Causality (tightened by Amendment 9 §3)
Every control decision for frame t (thresholds, bands, association
tolerance) uses only image cues of frame t and statistics of frames < t;
frame-t detections update state only for frames > t. Never future frames,
never GT, never full-sequence statistics. Asserted by
tests/test_v6_adaptive_layer.py (future-perturbation test).

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
on VisDrone") is not allowed in V6-TF (nor V5-TF). V4's VisDrone-selected values
(τ 0.75, s 0.4, association offset 0.10, NMS 0.45, tracker 45/0.86) are
category E under this rule and are NOT used by V6-TF. Every V6-TF constant
is classified in PARAMETER_STATUS.md (V6-TF table); the Otsu memory window,
RobustHistory window/warm-up and ECDF memory passed the E28 insensitivity
criterion on val-7 (category A). OTSU_BINS is not used by V6-TF.

## C6 — Architecture: AC starts BEFORE the detector
Frame → Generic Scene/State Analyzer → Online Self-Calibration → Generalized
Scene State / SCI → Universal AC Controller → Compute/Latency Constraint →
Detector Adapter → Detector → Universal Score/Candidate Handling → Tracker
Adapter → Tracker → Tracks → Causal feedback. Do not reduce the contribution to
a post-detector score filter. V4 compute-only is an ABLATION, not permission
to silently delete scene adaptation. V6-TF contains scene-state control:
motion-conditioned association (F3 rule, image statistics of frame t vs
their own causal history). Pre-detector resolution adaptation (F5, R-res)
was tested and REJECTED (FX-17); resolution is the compute-budget input.
This limitation is reported, not hidden (C11).

## C7 — Research tools are discovery-only
S1/S2/S3, C1/C2/C3, Optuna, decision stumps, cue selection and the 40/16 split
may answer: which cues help / are useless, which targets have headroom, which
rules are stable, which ranges are insensitive, which failure modes exist.
They must NOT produce the deployed controller. No final fit on the
development-40. The S3 learned controller is a research upper bound only
(comparison system D), never the method.

## C8 — Protected evaluations (details: PROTECTED_EVALUATIONS.md)
No quality metric (HOTA/IDF1/MOTA/IDS/FP/FN/precision/recall) on the
confirmation-16, Faster R-CNN, BoT-SORT or UAVDT (or test-dev for design)
before the V6-TF freeze (tag universal-acmot-v6-freeze). Caching, detection-level fidelity and timing are allowed where the
protocol says so. After freeze: evaluate once, no retuning.

## C9 — Hardware
Mac/MPS = development only; never report Mac timing. Official timing is Colab
T4 only. Amendment 5f fidelity-gate thresholds are PROTOCOL CONSTANTS; never
relax them after seeing the gate. If the gate fails materially: rebuild
reference development caches on T4; never tune the layer to compensate.
Amendment 9 §5: the owner deferred the gate and official timing to the final
pre-paper step; the V6-TF freeze is not conditioned on it (declared
limitation: all quality results come from Mac-MPS caches).

## C10 — Runtime cost
Online self-calibration = simple incremental statistics. No Optuna, model
fitting, per-frame optimisation or batch reprocessing at runtime. AC overhead
must be much smaller than detector latency (measured on T4).

## C11 — Honest reporting (V6-TF stays the final system)
Report every comparison truthfully. If V6-TF does not beat V4 compute-only at
matched compute, report it as a limitation of V6-TF (and, for scene components,
"scene-aware adaptation adds insufficient benefit under the tested
training-free constraints") — V6-TF remains the final system; never switch to
V4 or to a trained controller. Pre-freeze weaknesses are fixed in V6-TF;
post-confirmation fixes are a new revision needing new clean data.
Never claim "works with every detector/tracker": state exactly what was tested
(Detector: YOLOv8n, RT-DETR-L, Faster R-CNN ResNet50-FPN v2; Tracker:
ByteTrack, BoT-SORT).

## C12 — History is immutable
Never overwrite V1, V2*, V3, V4, V5 artifacts, frozen tags, locks, old
protocols, failed-experiment evidence or old results. New work = new named
revision (V5-TF, V6-TF). Changes to decisions = new superseding decision.

## C13 — Anti-hallucination
Never invent metrics, experiment status, commit SHAs, dataset leakage status,
parameter justifications or frozen state. Check git/code → canonical result
files → research/context → Graphify → detailed logs. If still ambiguous, write
UNKNOWN or NEEDS VERIFICATION.
