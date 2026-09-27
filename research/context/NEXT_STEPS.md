# NEXT STEPS (operational; keep short; move finished items to DECISIONS/EXPERIMENT_REGISTRY)

## CURRENT GOAL
Complete V5-TF — the FINAL system (V4 = baseline/ablation only). Implement
Amendment 7 (scene-adaptive resolution R-res, families F3/F5, control F5R,
scene-state logging, constant audit), validate all declared families on
VisDrone2019-MOT-train development-40, choose between F3 and F5 by the declared
rule, then freeze. Full checklist: PROJECT_COMPLETION.md.

## BLOCKERS
- Caches: 736 train caches (queue `tools/mac_cache_queue_v5tf.sh`) and the new
  640/832 caches for R-res (`tools/mac_cache_queue_v5tf_res.sh`) must reach
  56/56 per detector, then be merged into `outputs/det_cache_train_res/`.
  Note `outputs/det_cache_train_native/{rtdetr,visual_cues}` are symlinks to
  `outputs/det_cache_train/`.
- T4 fidelity gate (E38) needs a Colab T4 session (owner: do not ask them to
  run Colab manually; run when a T4 is reachable).
- Unjustified constants (RobustHistory window 100, warm-up 5, Otsu window 10,
  OTSU_BINS 64): resolved only by the Amendment-7 §6 audit (insensitive → A;
  sensitive → kept and reported as E). Z_REF/DECAY are outside V5-TF.
- `tools/t4_benchmark.py` reports only detector / tracker / "adaptive"
  time; official V5-TF timing needs separate scene-analyzer, normaliser and
  AC-decision timers (PROJECT_COMPLETION C6).
- Uncommitted working-tree code of unknown provenance: `universal_pipeline.py`
  (+effective-controls reporting, legacy-SCI path),
  `tests/test_full_pipeline_equivalence.py`; untracked `tools/calibrate_detector_*.py`,
  `tools/build_calibration_split.py`, `tools/audit/nms_audit_frcnn.py`,
  `tools/sync_t4_caches.sh`. Not on the V5-TF path. Never commit them with
  context/V5-TF changes; record `git status --short` next to E36.

## NEXT EXACT ACTION
`tools/v5tf_waiter.sh` (log `outputs/v5tf_waiter.log`) waits for all
development caches (736 + 640/832), merges levels, then runs
`tools/v5tf_dev.py run` (E36 + report + family choice) and `sens` (E39 audit). Verify it is alive:
`ps -Aww -o pid,etime,command | grep -E "v5tf|cache_queue|cache_detections|sleep 120" | grep -v grep`.
If it died, re-run the queues and then those commands. Record E36 (families)
and E39 (constant audit) in EXPERIMENT_REGISTRY.md.

The separate `tools/autonomous_v5tf_supervisor.py` waits for this existing
waiter without restarting it. It requires the complete E36/E39 output sets and
valid `family_choice.json` / `constant_audit.json`, refreshes context health,
then launches one serialized non-interactive Codex continuation at a time.
State: `research/context/AGENT_STATE.md`; logs: `outputs/autonomous_v5tf/`.

## AFTER THAT
0. Live V5-TF: `universal_acmot.py` now computes frame motion via scene_state.image_stats when assoc_motion is on (was always empty → F3 inert live). Still TODO: a live==replay parity check for a V5-TF family on a few development frames, and per-component AC timers (scene analyzer / normaliser+Otsu / AC decision) in the pipeline audit for tools/t4_benchmark.py.
1. Parameter audit update (PARAMETER_STATUS.md) from the E39 audit.
2. Add per-component AC timers to the T4 benchmark (no behaviour change).
3. Write the V5-TF policy file + lock; T4 fidelity gate (E38) → tag
   universal-acmot-v5tf-freeze.
4. Confirmation-16 once (reported vs V4); then val (secondary), Faster R-CNN,
   BoT-SORT, UAVDT, official T4 timing; test-dev post-hoc.

## DO NOT DO YET
- No quality metrics on confirmation-16, Faster R-CNN, BoT-SORT (V5-TF), UAVDT.
- Do not treat V4 as a fallback final system or compare "V4 vs V5-TF" as rival finals.
- Do not run `tools/v5_train.py final` (forbidden by Amendment 6).
- Do not run `tools/audit/nms_audit_frcnn.py` (Faster R-CNN tracking metrics).
- Do not run any undeclared family; new families must be added to the protocol first.
- Do not tag V5-TF before the fidelity gate passes.
- Do not change V1/V3/V4 policy files or locks.
