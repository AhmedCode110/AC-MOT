# VALIDATION PROTOCOL

Authoritative text: `research/OPTIMIZATION_PROTOCOL.md` (append-only
amendments). This file summarises; on conflict the protocol file wins.

## CURRENT protocol (Amendment 6, V5-TF) — with 5d, 5f, 5g still in force
1. Units are sequences; frames are never split; no random frame splitting.
2. Development = VisDrone2019-MOT-train development-40, YOLOv8n + RT-DETR-L
   jointly, ByteTrack, resolution 736 (exactly V4's compute). Data VALIDATE
   declared rule families; they never fit them.
3. Declared families F1, F2, F3 (F4 dropped: D-016, commit 71faf44 — NOT yet
   written into research/OPTIMIZATION_PROTOCOL.md, whose Amendment 6 still lists F4). Choice: lexicographic
   (1) fewest catastrophic cells (sequence × detector with MOTA < 0),
   (2) worst-detector relative ½(HOTA+IDF1) gain vs V4 on the same sequences,
   (3) simplicity F1 < F2 < F3.
4. Comparison at matched compute (736): A static (default; shared-static raw
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
8. Confirmation-16 once: V5-TF vs V4 at 736; adopted iff (1) catastrophic
   cells V5-TF ≤ V4 and (2) ½(HOTA+IDF1) V5-TF ≥ V4 for both detectors
   (pooled); paired sequence bootstrap (10,000, seed 42) reported.
9. Post-freeze, no retuning: val (secondary), Faster R-CNN, BoT-SORT,
   UAVDT, official T4 timing (component latencies, P95, FPS, GPU memory,
   AC overhead %), test-dev post-hoc.
10. Negative outcome is reported as such (HARD_CONSTRAINTS C11).

## HISTORICAL protocols (superseded, kept for provenance)
- Original global selection (V1): full 28-config grid, LOSO on val, J2 → Amendments 1–2 (J4, catastrophic-cell lexicographic rule).
- Amendment 3 (V3): invariant family, τ grid, same LOSO.
- Amendment 4 (V4): nested LOSO 360 configs on val; SCI removed.
- NMS audit rule (c1491ba).
- Amendments 5–5c (V5 on val: S1–S3 learned controller; not adopted).
- Amendment 5d (V5 on train, final learned fit) — its "final fit" clause superseded by Amendment 6.
- Amendment 5e (C3 Optuna) — discovery-only after Amendment 6.
