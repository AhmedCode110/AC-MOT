# FAILED EXPERIMENTS — negative results are project knowledge

Retry = allowed only if the listed precondition changes AND the retry is
declared in the protocol before running.

| ID | Version / commit | Hypothesis | Result (evidence) | Why it failed | Lesson | Retry? |
|---|---|---|---|---|---|---|
| FX-01 | V1 (pre-freeze, Colab) | online percentile normalisation alone is enough | E01: RT-DETR MOTA −112 | equal percentage ≠ equal count → candidate explosion | need a detector-relative gate | no |
| FX-02 | V2b | density budget + Top-K | E02: HOTA/recall destroyed | budget applied twice | avoid count budgets | no |
| FX-03 | V2c A/B/C | additive-offset density thresholds | E03: each variant fails a different sequence | additive offsets over-suppress | structural, not tunable | no |
| FX-04 | V2d | reliability-scaled budget / birth / SCI feedback | E07: RT-0268 MOTA −91.5 | budget floor 12, births in dead zone | — | no |
| FX-05 | V2 diagnostics | temporal persistence separates FP clutter | E10, E14, E17: persistence high for clutter (0.92–0.96), AUC 0.53–0.67 | scene property, not failure signal | Cue: temporal persistence rejected | only with a new signal definition |
| FX-06 | V2 diagnostics | per-track continuity separates false tracks | E09: gating cuts 24–30% TP | not a clean separator | — | no |
| FX-07 | V2f | closed-loop trust controller (r* 0.80/0.85/0.90) | E13: 2 tracks/frame on RT-0268; YOLO HOTA 33.5→27 | reliability conflates hard scene with false candidates (windup) | avoid integrating a reliability error | no |
| FX-08 | V1 selection | J2 = max-min absolute ½(HOTA+IDF1) | E18 / Amendment 1: always attained by YOLOv8n | unnormalised max-min degenerates when capacities differ | use relative gains (J4) + catastrophic-cell rule | no |
| FX-09 | V1 | histogram normaliser + ratio gate is robust | E20: YOLO HOTA 32.1→25.7 at T=0.5; RT MOTA 24.1→7.9 at T=2 | not invariant to temperature | require exact Platt invariance | no |
| FX-10 | legacy SCI in V3 | SCI (crowd/tiny/edges/darkness/blur) predicts when to spend compute | E24–E26: ≤ RANDOM allocation, sign flips | handcrafted absolute thresholds, no generalisation | never restore legacy SCI blindly | only as online-normalised re-definitions (C4) |
| FX-11 | shared-static baseline | a static raw threshold is enough | E19 competitive, but E20: 0 tracks under score ×0.5 | depends on raw score scale | baseline fragility, not our method | n/a |
| FX-12 | V5 S3 attempt 1 | MOTA-aligned learned stump controller | E34: ½(HOTA+IDF1) −2.4/−2.6 vs V4 | objective mismatch (MOTA cost vs HOTA/IDF1 criterion) | align cost with adoption metric | no (val folds exhausted) |
| FX-13 | V5 S3 attempt 2 | identity-aligned learned controller | E35: YOLO −2.26 | overfitting by scarcity (7 sequences, ~190 windows) | learned mapping does not generalise; motivates training-free (D-015) | no on val |
| FX-14 | V5 S3 | adapt gate τ by a cue | 4 different cues across folds | unstable cue mapping | τ not adapted | no |
| FX-15 | V4 on test-dev (E31) | V4 ≥ shared static at matched compute (H2) | RT-DETR −2.49 HOTA, −4.02 IDF1 (sig.) | recall-limited: τ, s at catastrophe-constraint edge | V4's VisDrone constants not universally good → category E | not for V4 (frozen) |
| FX-16 | V5-TF F4 | z-gate with online τ adds something to F1 | identical to F1 by construction (71faf44) | redundancy | — | no |

Gate ablation note (not a failure of the method): V4 without the gate
collapses like V1 (test-dev RT-DETR 12 catastrophic sequences, E31).
