# Prompt for Codex (paste as-is)

You are continuing the Universal AC-MOT Master's research project of Ahmed
Gouda (Military Technical College) in /Users/ahmedgouda/Desktop/Universal-ACMOT
(branch universal-adapters-v1). Do NOT restart the project and do NOT repeat
completed work.

1. Read `CODEX_HANDOFF.md` first, then `AGENTS.md`,
   `research/context/PROJECT_STATE.md`, `HARD_CONSTRAINTS.md`, `NEXT_STEPS.md`,
   `PROTECTED_EVALUATIONS.md`, and `research/final/FINAL_SUMMARY.md`.
2. The final method V6-TF is FROZEN (tag universal-acmot-v6-freeze, 2cff95f).
   Never change `universal_policy_pipeline.py`, `online_calibration.py`,
   `configs/universal_acmot_policy_v6tf.json`, or any file hashed in
   `research/V6TF_POLICY_LOCK.json`. Never tune on confirmation-16, test-dev,
   UAVDT, Faster R-CNN, BoT-SORT or any external system. No second
   confirmation run.
3. Hold the single-writer lock (`outputs/autonomous_v5tf/repo_writer.lock`)
   before editing tracked files. Never restart the old V5-TF supervisor.
4. Complete "In progress" item 1 in CODEX_HANDOFF.md (Faster R-CNN
   test-dev/UAVDT). Regenerate the tables, FINAL_RESULTS and figures; update
   the registry and PROJECT_COMPLETION; refresh the context and graph; run the
   health check; commit.
5. External published-system transfer: only after the owner has approved the
   downloads in research/final/EXTERNAL_PAPER_TRANSFER.md §3.
   1. Reproduce the published baseline with the authors' code, weights,
      split and evaluator.
   2. Record paper vs reproduction for every metric.
   3. Attach the SAME frozen V6-TF through an integration-only adapter.
   4. Evaluate once, internal vs published-protocol results kept separate.
   5. Fill sections 1–20 of EXTERNAL_PAPER_TRANSFER.md.

   Keep negative transfer results.
6. Report honestly. Separate functional success, experimental success and
   publication-quality evidence. Never claim state of the art.
