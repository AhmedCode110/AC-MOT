# Continue prompt — Universal AC-MOT V7 (paste into a cloud Claude Code / Codex session)

You are continuing Ahmed Gouda's AC-MOT / Universal AC-MOT Master's research,
V7 development cycle. Do NOT restart it and do NOT redo completed
experiments.

1. **Read first:**
   - `CODEX_HANDOFF_V7.md`: state, architecture, candidates, next
     experiments, data provisioning;
   - `research/final/V7_EXPERIMENT_LEDGER.md`: every experiment, including
     rejected ones;
   - `research/final/V7_DEV_RESULTS.json`;
   - `acmot_v7.py`, `tools/v7/`.
2. **Hardware.** Run `nvidia-smi`, check `torch.cuda.is_available()`, and
   record CPU/RAM/GPU/CUDA/PyTorch in `research/final/V7_CLOUD_RUNS.md`.
   Use CUDA if present, otherwise cloud CPU. Never use the owner's Mac for
   development runs. Label every timing with its exact hardware. There is no
   FPS work until V7 is frozen.
3. **Provision the data** listed in handoff §7:
   - caches, MOT17 val-half, VisDrone, external repos at the pinned commits
     plus the vendor diffs, TrackEval;
   - recreate the absolute paths with symlinks; never edit V6-locked files;
   - verify with the identity checks before anything else:
     - `V6EMU` must reproduce V6 on val-7: YOLO 18.6/34.3/38.6, RT-DETR
       25.0/41.6/48.1;
     - `NATIVE` / `BASELINE` must reproduce the official SparseTrack
       (68.876/77.849/81.974) and BoostTrack (68.492/75.502/81.413 online)
       results.
4. **Continue from handoff §5, in order:**
   - E11 motion-rule conditioning;
   - E12 VisDrone ID switches;
   - Stage D (UAVDT, test-dev, Faster R-CNN transfer, BoT-SORT);
   - stress tests (floors, score transforms);
   - tests;
   - bootstrap;
   - freeze (config, lock, tag `universal-acmot-v7-freeze`);
   - predeclared external set (TOPICTrack + 1–3 verified 2025/26 Q1/Q2
     systems), then faithful baselines vs exact frozen V7, reporting all
     outcomes;
   - final benchmark on one fixed device.
5. **Record keeping.** Record every experiment in the ledger with:
   - ID, hypothesis, mechanism, why it generalises, expected effect;
   - systems, metrics, per-sequence notes;
   - accepted/rejected, failure cases, next implication.

   Update the V7_* files in `research/final/`. Commit milestones; push
   without force.
6. **Constraints:**
   - training-free, online, causal (frames < t);
   - no labels, and no detector/tracker/dataset/sequence names or branches;
   - V6-TF (2cff95f) is immutable;
   - SparseTrack and BoostTrack are development hosts only;
   - TOPICTrack and confirmation-16 stay untouched until after the freeze;
   - no post-freeze policy changes;
   - keep negative results;
   - no state-of-the-art or universal-improvement claims unless the evidence
     supports them.
7. **Stopping.** Stop only when the full cycle is complete or at a genuine
   hard block. Before stopping, update both handoff files.
