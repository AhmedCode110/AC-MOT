# PROJECT COMPLETION — definition of done for the final system (V5-TF)

FINAL TARGET = Version: V5-TF. V4 = historical baseline / ablation only
(HARD_CONSTRAINTS C0). The project is complete when every item below is DONE,
with evidence recorded in DECISIONS / EXPERIMENT_REGISTRY / RESULTS_CANONICAL.
Status values: DONE · IN PROGRESS · BLOCKED · TODO. Update in the same commit as
the evidence.

## A. Design (pre-freeze, development-40 only)
| # | Item | Status | Evidence / pointer |
|---|---|---|---|
| A1 | Training-free requirement declared | DONE | Amendment 6 (3684684), C1 |
| A2 | Final target = V5-TF, V4 baseline only | DONE | Amendment 7, C0, D-018 |
| A3 | Candidate handling (ECDF + Otsu-3 bands) implemented | DONE | 71faf44 |
| A4 | Motion-aware association (F3) implemented | DONE | 71faf44 |
| A5 | Scene-adaptive resolution R-res + random control F5R declared | DONE | Amendment 7 §4 |
| A6 | R-res, scene-state vector logging implemented | DONE (crash-tested, no metrics) | online_calibration.py, universal_policy_pipeline.py, tools/v5tf_dev.py |
| A7 | Multi-resolution (640/832) development caches | IN PROGRESS | tools/mac_cache_queue_v5tf_res.sh → tools/merge_cache_levels.py → outputs/det_cache_train_res/ |
| A8 | E36 family validation (F1, F2, F3, F5, F5R + static, V4) | TODO | tools/v5tf_dev.py run/report |
| A9 | Family choice among selectable F3/F5 (declared rule) | TODO | outputs/v5tf_dev/family_choice.json |
| A10 | Constant audit (OTSU_BINS, Otsu window, RobustHistory window, warm-up) | TODO | Amendment 7 §6, tools/v5tf_dev.py sens |
| A11 | Parameter audit: every V5-TF constant in A/B/C/D or reported E | TODO | PARAMETER_STATUS.md |
| A12 | Optional research upper bound D (S3 on development-40, never final) | TODO (optional) | E37 |

## B. Freeze gate
| # | Item | Status | Evidence |
|---|---|---|---|
| B1 | Amendment-5f T4 fidelity gate PASS (fixed thresholds) | BLOCKED (needs T4) | E38, tools/fidelity_gate.py |
| B2 | Freeze tag universal-acmot-v5tf-freeze + V5-TF policy file + lock | TODO | FROZEN_VERSIONS.md |

## C. Post-freeze evaluation (once each, no retuning)
| # | Item | Status |
|---|---|---|
| C1 | Confirmation-16: V5-TF vs V4 (reported, not a gate) | TODO |
| C2 | VisDrone val secondary check | TODO |
| C3 | Unseen detector: Faster R-CNN ResNet50-FPN v2 | TODO |
| C4 | Tracker transfer: BoT-SORT | TODO |
| C5 | Unseen dataset: UAVDT test | TODO |
| C6 | Official T4 timing: scene analyzer, normaliser, AC decision, detector, tracker, total, P95, FPS, GPU memory, AC overhead % | TODO (tools/t4_benchmark.py needs per-component AC timers) |
| C7 | VisDrone test-dev post-hoc (labelled post-hoc) | TODO |

## D. Final report answers (must all be evidence-backed)
Does the AC layer require training? (NO) · labeled calibration data? (NO) ·
detector-specific tuning? (NO) · tracker-specific tuning? (NO) · self-calibrates
online? (YES) · causal? (YES) · real-time? (measured, C6) · transfers to Faster
R-CNN / BoT-SORT / UAVDT without tuning? (C3–C5) · improvement over V4
compute-only at matched compute? (C1, reported either way) · which scene/state
cues help? (A8 ablations: F1 vs F3 vs F5 vs F5R) · which old cues were rejected?
(D-005, D-010) · remaining failure cases? (FAILED_EXPERIMENTS.md).
