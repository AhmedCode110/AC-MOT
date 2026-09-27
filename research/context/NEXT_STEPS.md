# NEXT STEPS (operational; keep short; move finished items to DECISIONS/EXPERIMENT_REGISTRY)

## CURRENT GOAL
Validate the declared training-free V5-TF rule families (F1/F2/F3) on
VisDrone2019-MOT-train development-40 against V4 and static references at
matched compute (736), choose a family by the declared lexicographic rule,
then freeze V5-TF.

## BLOCKERS
- Train caches incomplete (as of 2026-09-27 06:53 UTC: YOLOv8n 0.45 53/56; RT-DETR-L,
  YOLOv8n native 0.7, visual cues 0/56). Note `outputs/det_cache_train_native/{rtdetr,visual_cues}`
  are symlinks to `outputs/det_cache_train/` (RT-DETR is NMS-free; cues are detector-independent) — Mac queue `tools/mac_cache_queue_v5tf.sh` running.
- T4 fidelity gate (E38) needs a Colab T4 session (owner said: do not ask
  them to run Colab manually; run when a T4 is reachable).
- Category-E / unverified constants in PARAMETER_STATUS.md (RobustHistory
  window 100 & warm-up 5, Otsu window 10, OTSU_BINS 64; Z_REF 0.75 and DECAY 0.9
  if any rule consumes scene_state.py cues) must be justified or
  sensitivity-checked on development-40 before freeze.
- HARD_CONSTRAINTS C6 risk: in the current families only F3 is scene-state driven
  (motion); the rest of the Scene State vector is logged, not used. Must be
  resolved (declared rule via amendment, or honest reporting) before freeze.
- Protocol text gap: Amendment 6 still lists F4 and its simplicity order; the
  drop is recorded only in D-016 / commit 71faf44. Needs a one-line protocol
  amendment before the family choice is reported.
- Uncommitted working-tree code of unknown provenance: `universal_pipeline.py`
  (+effective-controls reporting, legacy-SCI path) and
  `tests/test_full_pipeline_equivalence.py`; untracked `tools/calibrate_detector_*.py`,
  `tools/build_calibration_split.py`, `tools/audit/nms_audit_frcnn.py`,
  `tools/sync_t4_caches.sh`. Not on the V5-TF path (`tools/v5tf_dev.py` uses
  `universal_policy_pipeline.py`). Do not commit them with context changes; E36
  logs `git rev-parse HEAD` only, so record `git status --short` next to E36.

## NEXT EXACT ACTION
Nothing to launch: a waiter process auto-starts
`PYTHONPATH=. .venv/bin/python tools/v5tf_dev.py run` when
`outputs/det_cache_train_native/yolov8`, `outputs/det_cache_train/rtdetr` and
`outputs/det_cache_train/visual_cues` each hold 56 NPZ files. Verify it is
alive (`ps -Aww -o pid,etime,command | grep -E "v5tf|cache_queue|sleep 120"`; `-ww` is needed — the waiter's command line is long); if it died,
re-run the queue and then the command above. Then run
`tools/v5tf_dev.py report` and record E36 in EXPERIMENT_REGISTRY.md.

## AFTER THAT
1. Sensitivity checks for the flagged constants (development-40 only).
2. Decide whether any scene-state component (density/size/η/association)
   enters a declared rule — only by a protocol amendment BEFORE results that
   would use it; otherwise report V5-TF as candidate-handling + motion only.
3. Optional discovery E37 (`tools/v5_train.py s1|s2|s3`, never `final`) for the
   research upper bound D.
4. Amendment-5f T4 fidelity gate → freeze tag universal-acmot-v5tf-freeze.
5. Confirmation-16 once; then val (secondary), Faster R-CNN, BoT-SORT, UAVDT,
   official T4 timing; test-dev post-hoc.

## DO NOT DO YET
- No quality metrics on confirmation-16, Faster R-CNN, BoT-SORT (V5-TF), UAVDT.
- Do not run `tools/v5_train.py final` (forbidden by Amendment 6).
- Do not run `tools/audit/nms_audit_frcnn.py` (FRCNN tracking metrics).
- Do not tag anything V5-TF before the fidelity gate passes.
- Do not change V1/V3/V4 policy files or locks.
