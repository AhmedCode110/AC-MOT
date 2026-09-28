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
After tag universal-acmot-v6-freeze exists: run the one-way post-freeze
evaluations with the frozen config, exactly once each, no retuning. Record
each result as it lands in research/final/FINAL_RESULTS.md and here.

## AFTER THAT
1. Confirmation-16 (`V6_SPLIT=conf16 tools/v6/dev.py run V6TF V4 shared_static`),
   internal + official-compatible, paired sequence bootstrap (10,000, seed 42).
2. Val-7 official-compatible table of the frozen policy (development, labelled).
3. Faster R-CNN transfer (val-7 + test-dev post-hoc + UAVDT caches exist),
   BoT-SORT transfer, UAVDT transfer; each under a transfer lock written
   before running.
4. Test-dev post-hoc (labelled post-hoc).
5. External published MOT system(s): select, reproduce the baseline, attach
   the SAME frozen layer (integration-only adapter), evaluate
   (`research/final/EXTERNAL_PAPER_TRANSFER.md`).
6. T4 gate + official timing (pre-paper step).

## DO NOT DO
- No retuning of V6-TF after the tag; no second confirmation run.
- Do not treat V4 as a fallback final system.
- Do not restart tools/autonomous_v5tf_supervisor.py (targets E41).
- Do not run `tools/v5_train.py final` (Amendment 6).
- Do not change V1/V3/V4/V5-TF policy files, tags or locks.
