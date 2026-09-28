# VALIDATION PROTOCOL

Authoritative text: `research/OPTIMIZATION_PROTOCOL.md` (append-only
amendments). This file summarises; on conflict the protocol file wins.

## CURRENT protocol (Amendment 9, V6-TF = FINAL TARGET)
1. Development sandbox val-7 (iterative); development-40 = robustness check
   of the candidate (not iterated); V4's val-7 numbers are in-sample.
2. Acceptance priorities (owner): catastrophic cells → FP inflation →
   precision → MOTA → HOTA → IDF1 → IDS → recall → cross-detector
   consistency; each experiment logged in research/final/EXPERIMENT_LEDGER.md.
3. Causality: control decisions for frame t use frames < t (+ frame-t image
   cues); asserted by tests/test_v6_adaptive_layer.py.
4. Protected until universal-acmot-v6-freeze: confirmation-16, test-dev,
   UAVDT, Faster R-CNN and BoT-SORT quality. After the freeze each runs once.
   Confirmation-16 is reported with a paired sequence bootstrap (10,000,
   seed 42), vs V4 and shared-static, as a report and not a gate.
5. Evaluation: internal protocol (canonical, continuity) AND
   official-compatible VisDrone protocol, reported separately.
6. T4 fidelity gate (5f) and official timing: deferred to the pre-paper step
   (owner); freeze not conditioned on them.
7. After freeze: external transfer to published MOT systems with the same
   frozen layer (reproduced baseline first).

## PREVIOUS protocol (Amendments 6 + 7, V5-TF target) — superseded by Amendment 9 for the final target
V4 is a comparison baseline/ablation only; no step below selects between V4 and V5-TF.
1. Units are sequences; frames are never split; no random frame splitting.
2. Development = VisDrone2019-MOT-train development-40, YOLOv8n + RT-DETR-L
   jointly, ByteTrack, resolution 736 (exactly V4's compute). Data VALIDATE
   declared rule families; they never fit them.
3. Declared families (Amendments 6–7): F1, F2 (ablations: candidate handling
   only), F3 (+ motion-aware association), F5 (F3 + scene-adaptive resolution
   R-res), F5R (random-resolution control, not selectable). F4 dropped
   (Amendment 7 §2). Selectable: F3, F5. Choice: lexicographic
   (1) fewest catastrophic cells (sequence × detector with MOTA < 0),
   (2) worst-detector relative ½(HOTA+IDF1) gain vs V4 on the same sequences,
   (3) simplicity F3 < F5; F5 needs mean pixel cost ≤ 1.01·736².
   Then the constant audit (Amendment 7 §6) on the chosen family.
   Outcome (E36/E39): F3 selected; F5 did not beat F3 or random F5R.
   RobustHistory window/warm-up are insensitive A; the E39 OTSU_BINS/Otsu
   window findings apply only to recorded F3. E41 removes both dependencies
   with exact current-frame Otsu and is the locked pre-freeze candidate.
4. E41 (Amendment 8, predeclared before execution): exact 3-class Otsu on the
   current frame's logits, with thresholds restricted to gaps between sorted
   observations. It uses no histogram-bin count and no historical Otsu window.
   E41 is compared with the recorded F3 under the same catastrophic-cell /
   worst-detector relative gain / simplicity rule; it does not search values.
5. Comparison at matched compute (736): A static (default; shared-static raw
   0.5), B legacy SCI (V3; multi-resolution caches → val only), C V4
   compute-only, D S3 learned controller (research upper bound, never
   deployable), E V5-TF.
5. Cross-detector evidence: every conclusion must hold for both development
   detectors; transfer to Faster R-CNN is post-freeze evidence only.
6. Hardware fidelity (5f): Mac-MPS vs T4 on the 4 predeclared development
   sequences with fixed thresholds (protocol constants, 5g). FAIL → rebuild
   development caches on T4, re-run validation; never tune to hardware.
7. Freeze: tag universal-acmot-v5tf-freeze only after rule design fixed +
   gate PASS + confirmation run ready.
8. Confirmation-16 once: V5-TF vs V4 at matched compute — REPORTED
   (catastrophic cells, ½(HOTA+IDF1) per detector, paired sequence bootstrap
   10,000, seed 42). Not a selection gate (Amendment 7 §1).
9. Post-freeze, no retuning: val (secondary), Faster R-CNN, BoT-SORT,
   UAVDT, official T4 timing (component latencies, P95, FPS, GPU memory,
   AC overhead %), test-dev post-hoc.
10. Negative outcome is reported as a limitation of V5-TF (HARD_CONSTRAINTS C11); V5-TF stays final.

## HISTORICAL protocols (superseded, kept for provenance)
- Original global selection (V1): full 28-config grid, LOSO on val, J2 → Amendments 1–2 (J4, catastrophic-cell lexicographic rule).
- Amendment 3 (V3): invariant family, τ grid, same LOSO.
- Amendment 4 (V4): nested LOSO 360 configs on val; SCI removed.
- NMS audit rule (c1491ba).
- Amendments 5–5c (V5 on val: S1–S3 learned controller; not adopted).
- Amendment 5d (V5 on train, final learned fit) — its "final fit" clause superseded by Amendment 6.
- Amendment 5e (C3 Optuna) — discovery-only after Amendment 6.
- Amendments 5a/5d/6 adoption clauses ("otherwise V4 remains final") — superseded by Amendment 7 §1.
