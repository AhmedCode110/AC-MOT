# AC-MOT (scene-adaptive, Paper 1) — gap audit for a comparable system paper

Scope: the original AC-MOT line (tag `v1.0.0-acmot-frozen` → a6c1fa4) and the work around it. V7f, G1 and
G2 are parked for the next paper (V7f lock: 10 files identical to tag `universal-acmot-v7-freeze`, checked
2026-10-03). Every entry names the file it rests on; anything not confirmed from a file is marked
NEEDS VERIFICATION.

Legend: ✓ exists and is usable as is · ✗ missing · ↻ exists but must be redone or re-scored under the
official/comparable protocol.

## 1. Target of the paper
"AC-MOT improves strong, published tracking pipelines on VisDrone2019-MOT and UAVDT, under the protocol
those papers use, at real-time cost." Two kinds of comparison are needed:
- same detector, same tracker, same protocol: host vs host + AC-MOT (the claim itself);
- published state of the art on the same splits: a table of reported numbers with the detector of each row.

## 2. Audit

### 2.1 Method and code
| item | status | evidence / note |
|---|---|---|
| Frozen AC-MOT controller (cues, ECDF calibration, laws, K=10, W=7) | ✓ | tag `v1.0.0-acmot-frozen`; `scripts/run_final_test_3workers_locked.py::FrozenSCIController` |
| Validation-only selection chain (sweeps, TPE search, rules fixed before search) | ✓ | `EV/FROZEN_DEFENSIBLE_ACMOT_CONFIG.json` |
| Hand-designed controller (baseline) | ✓ | `core_v17.py::PresentationController` at a6c1fa4 |
| Plug-in form that wraps any detector + tracker (one interface) | ↻ | old code is wired to YOLOv8n + Ultralytics ByteTrack; a recalibrated variant ran inside U2MOT (`acmot_full_policy_calibrated.py`, Drive); the adapter layer of this repository (`adapters/`) can host it but has not been used for AC-MOT |
| Causality / no-GT unit tests for the AC-MOT controller | ✗ | tests exist for V7f and the SCI port (`tests/test_sci_v7.py`), not for the frozen AC-MOT code path |
| EV/ = `research/paper_split/evidence/legacy/` | | |

### 2.2 Evaluation protocol
| item | status | evidence / note |
|---|---|---|
| Internal protocol (class-agnostic, occ/trunc < 2) | ✓ but not comparable | all Paper 1 numbers; `tools/eval_local.py` |
| Official-compatible VisDrone Task-4b evaluator (class-aware, ignored regions) | ✓ code / ✗ validated | `tools/v6/eval_official.py` (Python port of the MATLAB toolkit); never checked against a published paper's numbers |
| Evaluator validation against published results | ✗ | needed before any SOTA table |
| UAVDT standard CLEAR-MOT protocol | ↻ | Paper 1 used a frozen adapter (ignore regions, IoU 0.5); whether it matches what UAVDT papers report: NEEDS VERIFICATION |
| Paired sequence bootstrap | ✓ | `tools/v7/bootstrap.py` (10,000, seed 42); Paper 1 used 5,000 |

### 2.3 Detector / tracker pipelines
| pipeline | split | status | result |
|---|---|---|---|
| YOLOv8n (COCO, not fine-tuned) + ByteTrack, AC-MOT Q | test-dev, UAVDT | ✓ internal / ↻ official | Q vs default +5.41 HOTA [+4.13, +6.90]; vs hand-designed +1.14 [+0.14, +2.27]; **vs matched static −0.09 [−0.25, +0.06]** (`EV/MATCHED_STATIC_A0_TESTDEV.json`); UAVDT +4.31 vs default |
| YOLOX-X fine-tuned on VisDrone (U2MOT release) + U2MOT, recalibrated AC-MOT | test-dev | ✓ (U2MOT repo protocol) | HOTA 54.94 vs 55.00 baseline, Δ −0.06 [−0.31, +0.22], MOTA 53.77 vs 53.87 (`research/transfer_legacy/rescore/u2mot/u2mot_testdev_rescore.json`) |
| SparseTrack + YOLOX (MOT17 val-half) | in-sample | ✓ | +0.13 vs static 0.75, −0.07 vs default 0.70 |
| Strong detector + ByteTrack / OC-SORT / BoT-SORT with AC-MOT on VisDrone | ✗ | — |
| Raw tracker outputs of the YOLOv8n test-dev / UAVDT runs (needed to re-score officially) | NEEDS VERIFICATION | per-sequence metrics are archived (`EV/V1_PER_SEQUENCE_METRICS.csv`); the track files themselves are not in the repository (Drive?) |
| Raw U2MOT tracker outputs | ✓ (Drive) | used for the rescore |

### 2.4 Baselines that a reviewer will ask for
| baseline | status |
|---|---|
| Static default of each host | ✓ (YOLOv8n, U2MOT) |
| Matched / tuned static operating point | ✓ (YOLOv8n test-dev, post hoc) — this is the comparison AC-MOT currently does not win |
| Other adaptive methods on the same pipeline (adaptive confidence threshold, adaptive resolution, content-adaptive configuration) | ✗ |
| Random / shuffled switching at equal compute | ✗ for AC-MOT (exists for the SCI port on val-7, `research/final/SCI_V7F_EXPERIMENT_LEDGER.md`) |
| Published SOTA trackers on VisDrone2019-MOT test-dev and UAVDT (reported numbers) | ✗ table not built; numbers must come from the PDFs |

### 2.5 Runtime
| item | status |
|---|---|
| T4 processing FPS, YOLOv8n pipeline | ✓ 38.98 (decode excluded); 21.90 with decoding (anchor run) |
| End-to-end FPS with a strong detector | ✗ (U2MOT matched timing exists for validation only) |
| Embedded device (Jetson) | ✗ |

### 2.6 Data
| split | local here | status |
|---|---|---|
| VisDrone2019-MOT-val | ✓ (`/root/acmot_work/VisDrone2019-MOT-val`) | development |
| VisDrone2019-MOT-train | Drive | needed only if a detector is fine-tuned |
| VisDrone2019-MOT-test-dev | Drive | PROTECTED (P5): post hoc use only, needs the owner's authorization |
| UAVDT test | Drive | PROTECTED (P4) for the universal line; legacy AC-MOT already ran on it |

## 3. What the audit says about the claim
1. Against weak defaults AC-MOT wins clearly (+5.41 / +4.31 HOTA), but both existing tests against a
   calibrated static setting are null: YOLOv8n matched static −0.09 [−0.25, +0.06] and U2MOT −0.06
   [−0.31, +0.22]. The second is the only strong-detector result and it is held-out.
2. No number of the paper is comparable with published VisDrone/UAVDT results yet (protocol and detector).
3. A paper that "beats" strong recent trackers therefore needs, first, evidence on validation that AC-MOT
   adds something over a tuned static point on a strong detector. Without that, a held-out run would
   most likely reproduce the U2MOT null result.

## 4. The experiments (7), in order; each ends with a decision

| # | experiment | data | status | decision it makes |
|---|---|---|---|---|
| X1 | Evaluator validation: score released tracker outputs of one published VisDrone tracker (U2MOT outputs on Drive, or a public result file) with `tools/v6/eval_official.py` and compare with the numbers printed in that paper | test-dev outputs that already exist (scoring only) | needs authorization (P5, post hoc) | if the reported MOTA/IDF1 are not reproduced within rounding, fix the evaluator before anything else |
| X2 | Literature table: 10–15 VisDrone2019-MOT / UAVDT papers 2022–2026 with detector, training data, protocol, MOTA/IDF1/HOTA/FPS, code availability; numbers transcribed from the PDFs with table numbers | none | ✗ | defines which rows are comparable (same detector or released detections) |
| X3 | Strong common pipeline on val-7: U2MOT's released YOLOX-X VisDrone detector (no training needed) feeding ByteTrack, OC-SORT, BoT-SORT; cache detections at the AC-MOT operating points | val-7 | ✗ (needs GPU or CI CPU time; YOLOX-X is ~105 M params) | provides the host baselines |
| X4 | Decisive validation comparison on X3: host default, tuned static (selected on val by the same chain), AC-MOT recalibrated by the frozen selection chain, shuffled schedule at equal compute, and two re-implemented adaptive baselines (adaptive confidence threshold, adaptive resolution); leave-one-sequence-out selection so the comparison is not in-sample | val-7 (LOSO) | ✗ | **gate**: AC-MOT must beat the tuned static point with a CI above 0 on at least two hosts; otherwise no held-out run and the claim is revised |
| X5 | Freeze (config hashes, lock) and one held-out run: test-dev (official protocol) and UAVDT, all systems of X4 | test-dev, UAVDT | needs authorization | the headline table |
| X6 | Runtime: end-to-end FPS and P95 with decoding on a T4 (and Jetson if available) for host and host + AC-MOT | timing only | ✗ | real-time claim |
| X7 | Re-score the existing YOLOv8n test-dev / UAVDT runs with the official protocol (if the raw track files are found) so the original result is reported on the same protocol | existing outputs | NEEDS VERIFICATION (raw files) + authorization | keeps Paper 1's held-out evidence usable |

Not redone: the selection chain, the ablations and the temporal grid of Paper 1 (validation, internal
protocol, they explain the method and are labelled as such).
