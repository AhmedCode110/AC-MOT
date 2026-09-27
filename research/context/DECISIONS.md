# DECISIONS — append-only log

Rules: never delete or edit the substance of an entry. A changed decision gets
a NEW entry with "Supersedes: D-xxx"; the old entry gets only a trailing
"Superseded by D-yyy" marker. Commit = the commit that recorded the decision
(verify with `git show <sha>`). Evidence IDs E01–E31: research/EXPERIMENT_REGISTRY.md;
E32+: research/context/EXPERIMENT_REGISTRY.md.

---
### D-001 — Original AC-MOT: handcrafted SCI controller around YOLOv8n + ByteTrack
Date 2026-09-11 · Commit a6c1fa4 (tag v1.0.0-acmot-frozen)
Question: can a lightweight SCI adapt detector operation online? Evidence:
legacy 17-sequence runs (README_ACMOT_*_2026-09-11.md). Decision: freeze legacy
AC-MOT. Reason: historical baseline. Alternatives rejected: none recorded.
Revisit: no (historical). Status: SUPERSEDED by D-002 as research line.

### D-002 — Universal AC-MOT: detector/tracker adapters + online normalisation
Date 2026-09-26 · Commits 6ebc6df…3f894d9
Question: make AC detector/tracker agnostic. Decision: adapter interfaces
(`adapters/detectors`, `adapters/trackers`), behaviour-preserving pipeline,
online score normalisation. Reason: plug-and-play requirement. Revisit: no.

### D-003 — V1 freeze via lexicographic cross-detector selection
Date 2026-09-27 · Commit e56c2f3 (tag universal-acmot-v1-freeze)
Evidence: E18; Amendments 1–2 (J2 max-min degenerates to weakest detector;
catastrophic-cell rule). Decision: gate ρ 0.6. Alternatives rejected: J2 choice
gate_r0.4 (3 RT-DETR cells MOTA<0). Superseded by D-004.

### D-004 — Calibration invariance is required → V3 (ECDF + z-logit gate)
Date 2026-09-27 · Commit c1e799d (tag universal-acmot-v3-freeze) · Amendment 3
Evidence: E20 (V1 breaks under temperature), E21–E23. Decision: order-only
ECDF + z-logit leader gate (demote), τ 0.75 (7/7 folds). Alternatives
rejected: histogram normaliser, ratio gate (not invariant). Revisit: only with
evidence that exact Platt invariance costs large accuracy on multiple detectors.

### D-005 — Legacy SCI questioned and removed from the V4 decision path
Date 2026-09-27 · Commit fc003bf (tag universal-acmot-v4-freeze) · Amendment 4
Evidence: E24 (SCI resolution ≈ uniform mix; SCI sensitivity ≈ constant 0.4),
E25 (no causal cue predicts 832-over-640 benefit), E26 (no cue nor SCI beats
RANDOM allocation at matched compute). Decision: V4 = compute-budget only.
Rejected cues: Cue: crowd n/30, Cue: tiny area, Cue: edge density,
Cue: brightness/darkness, Cue: blur, SCI weights. Revisit: only for cues
re-defined as online-normalised signals (done under D-009/D-015).

### D-006 — VisDrone test-dev evaluated exactly once, for V4
Date 2026-09-27 · Commits ac65d18 (lock), 9d9ebeb (results)
Evidence: E31, research/RESULTS_TESTDEV_V4.md. Decision: no change to V4
after test-dev. Consequence: test-dev is post-hoc only for all later versions.
Revisit: never.

### D-007 — Transfer tests locked before evaluation
Date 2026-09-27 · Commit a55b138 · research/TRANSFER_LOCK_FASTERRCNN.json,
TRANSFER_LOCK_UAVDT.json, TRANSFER_LOCK_UAVDT_FRCNN.json (V4).
Decision: Faster R-CNN (unseen detector) and UAVDT (unseen dataset) are
protected transfer tests. Revisit: no.

### D-008 — NMS 0.45 audit rule for NMS-bearing detectors
Date 2026-09-27 · Commit c1491ba. Evidence: E27 (YOLO only).
Decision: 0.45 is "shared robust" only if not worse than native NMS in ≥4/7
val sequences for every NMS-bearing family. Status: not run for Faster R-CNN
(blocked pre-V5-TF-freeze by D-014). Superseded for V5-TF by D-015
(detector-native suppression, category D).

### D-009 — Deleting scene adaptation rejected as final direction; V4 = ablation
Date 2026-09-27 · Commit 469e931 · Amendment 5
Question: does E24–E26 prove scene adaptation is useless? Decision: no — they
only show the LEGACY cues/SCI fail. V4 kept as ablation/alternative; V5 scene-
adaptive developed separately. Data status fixed: test-dev not clean for V5.
Revisit: if V5-TF fails to beat V4 at matched compute, report that honestly
(C11) — do not re-delete scene adaptation silently.
Refined by D-018 (V4 is baseline/ablation only, never final).

### D-010 — S2 cue status (val, diagnostic)
Date 2026-09-27 · Commit 57a7e72 · Amendment 5a · E33
Candidates: Cue: det_gap, Cue: det_count, Cue: trk_survival, Cue: trk_match,
Cue: img_motion, Cue: img_motion_resp (signal beyond permutation null, both
detectors). Rejected: Cue: img_edges, Cue: img_brightness, Cue: img_blur
(negative everywhere). Revisit: candidates must be re-validated as online-
normalised signals on development-40.

### D-011 — Learned V5 attempts on val not adopted; move development to train
Date 2026-09-27 · Commit 62111e8 · Amendments 5b/5c · E34, E35
Evidence: attempt 1 (MOTA-aligned cost) and attempt 2 (identity cost) both
lose ½(HOTA+IDF1) vs V4 on outer folds; gate-τ cue unstable (4 answers).
Decision: τ not adapted; develop on VisDrone2019-MOT-train. Diagnosis:
overfitting by scarcity (7 sequences). Revisit: no further attempts on val folds.

### D-012 — Val is not a confirmation set; fixed 40/16 train split
Date 2026-09-27 · Commit b691c86 · Amendment 5d · research/TRAIN_SPLIT_V5.json
Decision: development-40 / confirmation-16 (seed 20260927, length-stratified,
metadata only). Val = secondary check only. "Final V5 = fitted on all 40"
→ Superseded by D-015 (no fitting). Adoption/"V4 remains final" clause superseded by D-018.

### D-013 — C3 Optuna controller specification
Date 2026-09-27 · Commit bbe7271 · Amendment 5e
Decision: C3 = Optuna-TPE linear scene score (seed 0, 150 trials, SQLite).
Status after D-015: research/discovery tool only; must not be deployed.

### D-014 — Cross-hardware fidelity gate; constants; pre-freeze transfer use
Date 2026-09-27 · Commits feff14f (5f), 8293c7c (5g)
Decision: Mac-MPS vs T4 gate on 4 predeclared development sequences with fixed
PASS thresholds (protocol constants, never relaxed). Before freeze, Faster
R-CNN / BoT-SORT only for timing and detection-level fidelity; no quality
metrics, no tuning. Revisit: never after seeing the gate.

### D-015 — FINAL AC LAYER MUST BE TRAINING-FREE (V5-TF)
Date 2026-09-27 · Commit 3684684 · Amendment 6 · Supersedes: D-012 (final fit),
D-013 (deployment role), D-008 (for V5-TF)
Question: may the final AC controller be fitted offline (stumps/trees/Optuna)?
Evidence: owner requirement; D-011 (learned controllers overfit/generalise
poorly); plug-and-play goal. Decision: deployed AC is training-free, online
self-calibrating, causal; V4 VisDrone-selected values are category E and not
used; S1–S3/Optuna discovery-only; learned S3 controller = research upper
bound D; comparison A–E at matched compute; freeze tag
universal-acmot-v5tf-freeze only after rule design fixed + T4 gate PASS.
Why online self-calibration: it makes the same rule meaningful across
datasets and detector score scales (order statistics, Otsu on logits are
invariant to Platt/temperature recalibration). Alternatives rejected: fitted
controller (C1–C3) as final; restoring legacy SCI blindly. Revisit: only via
a new owner-approved amendment; evidence that training-free fails is reported,
not used to switch to training (C11).

### D-016 — F4 dropped from the V5-TF families
Date 2026-09-27 · Commit 71faf44
Reason: a z-gate with τ = Otsu T2 in z units is mathematically identical to F1.
Revisit: no.

### D-017 — Repository is the primary project memory
Date 2026-09-27 · Commit: the commit adding research/context/ (see GIT_STATE.md)
Decision: research/context/ + Graphify graph are the canonical session
memory; chat history is last in the source-of-truth hierarchy. Revisit: no.

### D-018 — FINAL TARGET = V5-TF; V4 is baseline/ablation only
Date 2026-09-27 · Commit: the commit adding Amendment 7 (see git log -- research/OPTIMIZATION_PROTOCOL.md)
Supersedes: the "otherwise V4 remains the final system" clauses of D-012 /
Amendments 5a, 5d; refines D-009 and C11.
Question: may V4 serve as the final system if V5-TF underperforms? Decision:
no. V5-TF is the final research target and contribution; V4 is historical
evidence, ablation and comparison baseline only. Confirmation-16 reports
V5-TF vs V4; it does not select the final system. Weaknesses are fixed in
V5-TF before freeze (declared families on development-40); post-confirmation
fixes need a new revision and new clean data. Reason: owner direction;
consistent with C1–C11 (honest reporting retained). Alternatives rejected:
V4 fallback; choosing between V4 and V5-TF as rival finals. Revisit: only by
owner.

### D-019 — Scene-state control required in the frozen V5-TF; R-res declared
Date 2026-09-27 · Amendment 7 §3–5
Question: how does V5-TF satisfy C6 (AC before the detector, scene-adaptive)?
Decision: add R-res — resolution chosen before detection from the tertile of
the causal ECDF rank of the smoothed object-size state (median log area
fraction of primary candidates), with a compute-budget guard; family F5 =
F3 + R-res; control F5R (random allocation, same guard). Selectable finals are
F3/F5 only; F1/F2 are ablations. Scene state computed in the V5-TF path, not by
scene_state.py (Z_REF, DECAY out of V5-TF). Retention not adapted (E32).
Evidence basis: E24–E26 rejected ABSOLUTE legacy cues, not online-normalised
relative state; S1 resolution headroom per window ≈2.0/2.6 MOTA pts (E32).
Revisit: if F5 ≤ F5R, report that the size state does not beat random
allocation (C11); a new declared family may be tried on development-40.

### D-020 — Constant audit rule for V5-TF
Date 2026-09-27 · Amendment 7 §6
Decision: one-at-a-time sensitivity of OTSU_BINS, Otsu window, RobustHistory
window, warm-up on the chosen family; E28 criterion (|ΔHOTA| ≤ 0.4, no new
catastrophic cell) → structural A; otherwise keep the declared default and
report as category E. Never adopt the best-scoring value.
