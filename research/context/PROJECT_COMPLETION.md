# PROJECT COMPLETION — definition of done for the final system (V6-TF)

FINAL TARGET = Version: V6-TF (Amendment 9; HARD_CONSTRAINTS C0). V4 =
historical baseline / ablation only. V5-TF/E41 = rejected development
history (FX-18). The project is complete when every item below is DONE, with
evidence in DECISIONS / EXPERIMENT_REGISTRY / RESULTS_CANONICAL and
`research/final/`. Status values: DONE · IN PROGRESS · BLOCKED · TODO ·
DEFERRED (owner).

## A. Design (pre-freeze, development only: val-7 sandbox, dev-40 check)
| # | Item | Status | Evidence / pointer |
|---|---|---|---|
| A1 | Training-free requirement declared | DONE | Amendment 6, C1 |
| A2 | E41 audit; E41 not freeze-worthy | DONE | E42, FX-18, D-026 |
| A3 | V6 development iterations with ledger | DONE | E43, research/final/EXPERIMENT_LEDGER.md |
| A4 | Candidate selected by owner priorities (no catastrophic, FP, precision, MOTA, …) | DONE — X5 | D-027 |
| A5 | Robustness: Platt/monotone recalibration, emission floor, memory constants, ablations | DONE | E44 |
| A6 | Development-40 robustness check (not iterated) | DONE | E45 |
| A7 | Parameter audit: every V6-TF constant A/B/C(safety)/D | DONE | PARAMETER_STATUS.md V6-TF table |
| A8 | Causality audit (future-perturbation test) + no detector branching | DONE | tests/test_v6_adaptive_layer.py |
| A9 | Live == replay parity for the actual candidate | DONE | E46, outputs/v6/live_replay_parity.json |
| A10 | Functional tests pass | DONE | tests/ (V6 9 + legacy 3) |
| A11 | Official-compatible VisDrone evaluator implemented and sanity-checked | DONE | tools/v6/eval_official.py, E47 |

## B. Freeze
| # | Item | Status | Evidence |
|---|---|---|---|
| B1 | Clean tree; unknown files preserved outside the freeze | DONE | D-028 |
| B2 | Policy file + lock with file hashes | DONE | research/V6TF_POLICY_LOCK.json |
| B3 | Freeze tag universal-acmot-v6-freeze + freeze record | DONE (2cff95f) | research/final/FREEZE_RECORD_V6TF.md |
| B4 | Amendment-5f T4 fidelity gate | DEFERRED (owner, Amendment 9 §5) | run before publication; report either way |

## C. Post-freeze evaluation (once each, no retuning)
| # | Item | Status |
|---|---|---|
| C1 | Confirmation-16: V6-TF vs V4 vs shared-static (internal + official-compatible, paired bootstrap) | DONE (E48) |
| C2 | VisDrone val official-compatible table (development, labelled) | DONE (E47, TABLES/val7_development_official) |
| C3 | Unseen detector: Faster R-CNN ResNet50-FPN v2 | val-7 DONE; test-dev/UAVDT IN PROGRESS (E52) |
| C4 | Tracker transfer: BoT-SORT | DONE (E51) |
| C5 | Unseen dataset: UAVDT | DONE (E50) |
| C6 | Official T4 timing | DEFERRED (owner) |
| C7 | VisDrone test-dev post-hoc (labelled post-hoc) | DONE (E49) |
| C8 | External published MOT system(s) + frozen layer, reproduced baseline first | BLOCKED: owner permission for downloads (MOT17, weights, repos); survey done (research/final/EXTERNAL_PAPER_TRANSFER.md) |

## D. Paper package (research/final/) — drafted; FRCNN test-dev/UAVDT and external transfer sections pending
FINAL_METHOD · EXPERIMENT_LEDGER · FINAL_RESULTS · FAILURE_ANALYSIS · ABLATION ·
REPRODUCIBILITY · PAPER_CLAIMS · TABLES/ · FIGURES/ · FINAL_SUMMARY ·
EXTERNAL_PAPER_TRANSFER — each evidence-backed; claims split into supported /
partially supported / not supported.
