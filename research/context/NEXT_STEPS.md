# NEXT STEPS (operational; keep short; move finished items to DECISIONS/EXPERIMENT_REGISTRY)

## CURRENT GOAL
Complete V5-TF — the FINAL system (V4 = baseline/ablation only). E36 selected
F3 over F5 by the declared rule; E41 removes the E39 OTSU_BINS and Otsu-window
dependencies with a predeclared training-free family. Complete the T4 gate and
freeze. Full checklist: PROJECT_COMPLETION.md.

## BLOCKERS
- T4 fidelity gate (E38) needs a Colab T4 session (owner: do not ask them to
  run Colab manually; run when a T4 is reachable).
- Official T4 timing still to run (harness ready: scene / calib / decision /
  detector / tracker / total / P95 timers, commit 1eaa3c6; Mac numbers are
  development-only).
- Single-writer rule: any interactive session must stop the supervisor or hold
  its lock before editing tracked files (AGENTS.md, D-021).
- Uncommitted working-tree code of unknown provenance: `universal_pipeline.py`
  (+effective-controls reporting, legacy-SCI path),
  `tests/test_full_pipeline_equivalence.py`; untracked `tools/calibrate_detector_*.py`,
  `tools/build_calibration_split.py`, `tools/audit/nms_audit_frcnn.py`,
  `tools/sync_t4_caches.sh`. Not on the V5-TF path. Never commit them with
  context/V5-TF changes; record `git status --short` next to E36.

## NEXT EXACT ACTION
E41 is complete and locked: 80/80 development-40 replays, 15 catastrophic
cells, and +0.69% worst-detector relative gain vs V4. The policy lock is
`research/V5TF_POLICY_LOCK.json`. Prepare the fixed T4 fidelity gate. Do not
run protected quality evaluations before the freeze.

## AFTER THAT
0. Live V5-TF parity: DONE (`outputs/v5tf_dev/live_replay_parity_v2.json`,
   80/80 frames; v1 preserved as a harness failure caused by process-global
   track IDs). Per-component timers: DONE (1eaa3c6).
1. Parameter audit: DONE for the locked E41 rule; E39 bins/window are excluded.
2. V5-TF policy file + lock are DONE; T4 fidelity gate (E38) → tag
   universal-acmot-v5tf-freeze.
3. Confirmation-16 once (reported vs V4); then val (secondary), Faster R-CNN,
   BoT-SORT, UAVDT, official T4 timing; test-dev post-hoc.

## DO NOT DO YET
- No quality metrics on confirmation-16, Faster R-CNN, BoT-SORT (V5-TF), UAVDT.
- Do not treat V4 as a fallback final system or compare "V4 vs V5-TF" as rival finals.
- Do not run `tools/v5_train.py final` (forbidden by Amendment 6).
- Do not run `tools/audit/nms_audit_frcnn.py` (Faster R-CNN tracking metrics).
- Do not run any undeclared family; new families must be added to the protocol first.
- Do not tag V5-TF before the fidelity gate passes.
- Do not change V1/V3/V4 policy files or locks.
