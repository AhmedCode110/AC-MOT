# PARAMETER STATUS — current constants (V5-TF focus)

Categories: A mathematical/structural · B online-derived · C research-only
optimisation variable / generic safety bound (see note) · D API-required ·
E unjustified/manual ("worked on VisDrone"). Historical audit (V1–V4):
`research/PARAMETER_AUDIT.md` — still authoritative for V4.
Note on C: Amendment 6 uses C = "generic engineering safety bound"; the V4
audit used C = "globally optimised". Below, C(safety) and C(opt) are kept apart.
C(opt) values are NOT allowed in V5-TF.

## ⚠ Category E / unverified items that touch V5-TF (must be resolved before freeze)
| Parameter | Component | Value | Problem | Required action |
|---|---|---|---|---|
| RobustHistory window | `online_calibration.py` | 100 samples | memory of the motion-ratio / scene-state history (NOT the Otsu window, which is 10 frames as in Amendment 6); Amendment 6 gives no length → no evidence yet | justify as structural memory + sensitivity sweep on development-40, or derive from stream |
| RobustHistory warm-up | `online_calibration.py` | 5 samples | minimum sample count, no evidence | treat as structural warm-up (A) with rationale, or sweep |
| Otsu window | `universal_policy_pipeline.py` (`gate_window`) | 10 frames | inherited from V4 gate (E28 covers V4 gate, not Otsu) | sensitivity on development-40 |
| OTSU_BINS | `online_calibration.py` | 64 | code comment claims "sensitivity-checked"; no registry evidence found | NEEDS VERIFICATION |
| Z_REF | `scene_state.py` | 0.75 (= V4 τ) | category E if any V5-TF rule consumes det_gap | currently logging only; must not enter a V5-TF rule |
| DECAY | `scene_state.py` | 0.9 | EMA memory; E28 evidence is for V4 leader EMA | logging only in V5-TF; justify if used |

## V5-TF parameters
| Parameter | Component | Value | Cat. | Origin | Evidence | Sensitivity | Status |
|---|---|---|---|---|---|---|---|
| ECDF normaliser | score handling | order-only, frames < t | A/B | V3 | E21–E23 exact invariance | — | kept |
| 3-class Otsu thresholds T1, T2 | candidate handling | derived per window | B | Amendment 6 | affine-equivariant on logits | — | experimental |
| Number of classes | candidate handling | 3 | A | mirrors tracker high/low/discard | structural | — | experimental |
| Band scores | candidate→tracker | primary 0.5+0.5u, secondary 0.1+0.4u | A | band boundaries | structural mapping | — | experimental |
| Tracker thresholds | tracker adapter | association = birth 0.5, low 0.1 | A/D | band boundaries; native low | — | — | experimental |
| First-frame fallback | candidate handling | within-frame Otsu | A | Amendment 6 | — | — | experimental |
| Logit clip | `online_calibration.logits` | 1e-9 | A | numerical | — | none | kept |
| MAD scale | RobustHistory | 1.4826 | A | Gaussian consistency constant | — | none | kept |
| F3 cap | association tolerance | 0.95 | C(safety) | Amendment 6 | keeps match < 1 | — | experimental |
| m0 | association tolerance | tracker-native 0.8 | D | tracker default | — | — | experimental |
| Tracker retention / match | ByteTrack, BoT-SORT | native 30 / 0.8 | D | tracker defaults | — | — | experimental |
| Detector suppression | detector adapters | native: YOLOv8n 0.7, Faster R-CNN 0.5, RT-DETR-L none | D | API defaults | — | — | experimental |
| Score floor / max det | detector adapters | 0.01 / 1000 | D | emission floor | same all adapters | — | kept |
| Resolution | compute constraint | budget input; 736 in experiments | D (deployment input) | V4 | E24, E28 | monotone | kept |
| Motion cue | `tools/visual_cues.py` | phase-correlation shift / image diagonal | A | definition | — | — | experimental |

## V4 values — category E under Amendment 6 (V4 only, never V5-TF)
τ 0.75 · s 0.4 · association offset 0.10 · NMS request 0.45 · tracker 45 / 0.86
(C(opt) in V4's own audit; "worked best on VisDrone" under Amendment 6).

## Protocol constants (not model parameters; never tuned)
Fidelity gate (Amendment 5f): ≥99% detections matched IoU≥0.95, class
agreement ≥99.5%, median |Δscore| ≤0.01, |ΔHOTA|,|ΔIDF1|,|ΔMOTA| ≤0.5,
|ΔIDS| ≤5%, V5 decision agreement ≥95%. Bootstrap 10,000 resamples seed 42.
Catastrophic cell: MOTA < 0. Split seed 20260927.
