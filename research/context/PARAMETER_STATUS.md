# PARAMETER STATUS — current constants (V5-TF focus)

Categories: A mathematical/structural · B online-derived · C research-only
optimisation variable / generic safety bound (see note) · D API-required ·
E unjustified/manual ("worked on VisDrone"). Historical audit (V1–V4):
`research/PARAMETER_AUDIT.md` — still authoritative for V4.
Note on C: Amendment 6 uses C = "generic engineering safety bound"; the V4
audit used C = "globally optimised". Below, C(safety) and C(opt) are kept apart.
C(opt) values are NOT allowed in V5-TF.

## ⚠ Category E items that block the V5-TF freeze
| Parameter | Component | Value | E39 evidence | Required action |
|---|---|---|---|---|
| Otsu window | `universal_policy_pipeline.py` (`gate_window`) | 10 frames | sensitive: 5 adds a catastrophic cell; 20 passes | keep only for recorded F3; replace dependency in a new predeclared family, never choose 20 post-hoc |
| OTSU_BINS | `online_calibration.py` | 64 | sensitive: 32 changes RT-DETR HOTA −0.63 and adds catastrophic cells; 128 passes | keep only for recorded F3; replace fixed-bin dependency, never choose 128 post-hoc |

## Category E values outside the V5-TF decision path

These values do not block the freeze while they remain excluded from V5-TF.

| Parameter | Component | Value | Evidence | Required action |
|---|---|---|---|---|
| Z_REF | `scene_state.py` | 0.75 (= V4 τ) | outside V5-TF (Amendment 7 §3: V5-TF scene state is computed in online_calibration.py) | none while scene_state.py stays out of V5-TF decisions |
| DECAY | `scene_state.py` | 0.9 | outside V5-TF (same) | none |

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
| RobustHistory window | `online_calibration.py` | 100 samples | A | declared default | E39: 50 and 200 pass both detectors with no extra catastrophic cell | insensitive | kept |
| RobustHistory warm-up | `online_calibration.py` | 5 samples | A | declared default | E39: 3 and 10 pass; effectively identical | insensitive | kept |
| Otsu window | candidate handling | 10 frames | E | declared default, retained not selected | E39 sensitive | 5 sensitive, 20 passes | freeze blocker |
| OTSU_BINS | candidate handling | 64 | E | declared default, retained not selected | E39 sensitive | 32 sensitive, 128 passes | freeze blocker |
| F3 cap | association tolerance | 0.95 | C(safety) | Amendment 6 | keeps match < 1 | — | experimental |
| m0 | association tolerance | tracker-native 0.8 | D | tracker default | — | — | experimental |
| Tracker retention / match | ByteTrack, BoT-SORT | native 30 / 0.8 | D | tracker defaults | — | — | experimental |
| Detector suppression | detector adapters | native: YOLOv8n 0.7, Faster R-CNN 0.5, RT-DETR-L none | D | API defaults | — | — | experimental |
| Score floor / max det | detector adapters | 0.01 / 1000 | D | emission floor | same all adapters | — | kept |
| Resolution | compute constraint | budget input; 736 in experiments | D (deployment input) | V4 | E24, E28 | monotone | kept |
| R-res levels | compute constraint | {640, B, 832}, B = 736 | D | deployment/API levels (stride-32 multiples) | E24 accuracy monotone in level | — | declared (Amendment 7) |
| R-res bands | compute constraint | ECDF-rank tertiles (1/3, 2/3) | A | three equiprobable bands for three levels | structural | — | declared |
| R-res smoothing / decision block | compute constraint | 10-frame median per level; decision held per 10-frame block (= Otsu memory) | A/B | shares the Otsu window (audited in §6) | — | via Otsu-window audit | declared |
| R-res state conditioning | compute constraint | size history kept per level | A | removes self-induced bias of the action | structural | — | declared |
| Budget guard | compute constraint | mean pixel cost ≤ B² | C(safety) | compute budget | matched compute | — | declared |
| F5R block length | control only | 10 frames, fixed seed | A | same memory as the rule (E26 design) | — | — | control, not deployed |
| Motion cue | `tools/visual_cues.py` | phase-correlation shift / image diagonal | A | definition | — | — | experimental |

E39 authority: `outputs/v5tf_dev/constant_audit.json`; per-variant PKLs under
`outputs/v5tf_dev/S:F3:*`. Amendment 7 says sensitive defaults are kept and
reported, but this does not waive HARD_CONSTRAINTS C5 for the frozen system.

## V4 values — category E under Amendment 6 (V4 only, never V5-TF)
τ 0.75 · s 0.4 · association offset 0.10 · NMS request 0.45 · tracker 45 / 0.86
(C(opt) in V4's own audit; "worked best on VisDrone" under Amendment 6).

## Protocol constants (not model parameters; never tuned)
Fidelity gate (Amendment 5f): ≥99% detections matched IoU≥0.95, class
agreement ≥99.5%, median |Δscore| ≤0.01, |ΔHOTA|,|ΔIDF1|,|ΔMOTA| ≤0.5,
|ΔIDS| ≤5%, V5 decision agreement ≥95%. Bootstrap 10,000 resamples seed 42.
Catastrophic cell: MOTA < 0. Split seed 20260927.
