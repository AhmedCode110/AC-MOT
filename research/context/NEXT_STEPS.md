# NEXT STEPS (operational; keep short; move finished items to DECISIONS/EXPERIMENT_REGISTRY)

## CURRENT GOAL
Complete V6-TF, the FINAL system (Amendment 9; V4 = baseline/ablation only;
E41/V5-TF = rejected development history). Freeze → one-way post-freeze
evaluations → external published-system transfer → paper package in
`research/final/`. Full checklist: PROJECT_COMPLETION.md.

## BLOCKERS
- T4 fidelity gate (Amendment 5f) and official T4 timing: deferred by the
  owner to the final pre-paper step (Amendment 9 §5). Harness:
  notebooks/Colab_T4_gate_and_benchmark.ipynb must be retargeted to the V6-TF
  freeze commit and configs/universal_acmot_policy_v6tf.json first.
- Single-writer rule (AGENTS.md): hold outputs/autonomous_v5tf/repo_writer.lock
  before editing tracked files; the old V5-TF supervisor must NOT be restarted
  (its stage machine targets the rejected E41 path).
- Unknown-provenance WIP preserved in git stash (D-028); do not pop it into a
  freeze or evaluation commit.

## NEXT EXACT ACTION
1. When `outputs/v6/after_frcnn.log` shows AFTER_DONE: regenerate
   `tools/v6/make_tables.py`, append the Faster R-CNN test-dev/UAVDT results to
   research/final/FINAL_RESULTS.md (re-run the compose step in CODEX_HANDOFF.md)
   and update E52 / PROJECT_COMPLETION C3.
2. External published-system transfer (C8): after the owner approves the
   downloads listed in research/final/EXTERNAL_PAPER_TRANSFER.md §3, clone
   BoostTrack (primary) and OC-SORT/SparseTrack, reproduce the baseline on
   MOT17 val-half, attach the SAME frozen V6-TF via an integration-only
   adapter, evaluate once.
3. T4 fidelity gate + official timing (pre-paper; retarget
   notebooks/Colab_T4_gate_and_benchmark.ipynb to 2cff95f / v6tf config).

## DO NOT DO
- No retuning of V6-TF after the tag; no second confirmation run.
- Do not treat V4 as a fallback final system.
- Do not restart tools/autonomous_v5tf_supervisor.py (targets E41).
- Do not run `tools/v5_train.py final` (Amendment 6).
- Do not change V1/V3/V4/V5-TF policy files, tags or locks.
