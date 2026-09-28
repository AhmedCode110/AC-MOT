# Standalone continue prompt — Universal AC-MOT V7 (for Codex or a cloud Claude session)

You are continuing Ahmed Gouda's Master's research "AC-MOT / Universal
AC-MOT". This prompt is self-contained. No earlier conversation is needed:
GitHub is the authoritative state. Do NOT restart the project and do NOT
repeat completed experiments.

## What this research is
AC-MOT is a training-free, online, causal adaptive control layer. It sits
between a frozen object detector and a frozen multi-object tracker and
decides, per frame, which detector candidates reach the tracker and with
which operating point.

- **V6-TF** was frozen (commit 2cff95f, tag `universal-acmot-v6-freeze`) and
  is immutable. It helped on detector/domain-shifted VisDrone streams, but
  degraded strong published MOT17 trackers:
  - SparseTrack, IEEE TCSVT 2025: HOTA −4.15;
  - BoostTrack, MVA 2024: HOTA −5.89.

  Causes: IoU-0.5 duplicate suppression removed overlapping pedestrians, and
  over-strict self-calibrated thresholds.
- **V7** (in development, NOT frozen) must learn WHEN NOT TO INTERVENE:
  - pass through clean streams;
  - intervene on noisy or miscalibrated ones;
  - keep crowded overlaps;
  - suppress true duplicates;
  - stay robust to score calibration and emission floors.

## Start here
1. Clone https://github.com/AhmedCode110/AC-MOT, branch
   `universal-adapters-v1`, with tags (no shallow clone).
2. Read, in order:
   - `CODEX_HANDOFF_V7.md` (authoritative state, next experiments, commands);
   - `research/final/V7_EXPERIMENT_LEDGER.md` (all experiments incl. rejected);
   - `research/final/V7_DEV_RESULTS.json`;
   - `research/final/ASSET_MANIFEST.json`;
   - `acmot_v7.py`, `tools/v7/`.
3. Record the hardware (`nvidia-smi`, `torch.cuda.is_available()`, CPU, RAM,
   OS) in `research/final/V7_CLOUD_RUNS.md`.
4. Run `bash scripts/setup_research_assets.sh`, then
   `source $ACMOT_WORK/acmot_env.sh`.
   - Datasets come from official sources.
   - Derived caches come from GitHub release `v7-dev-assets-1`.
   - If a download is blocked, document it, use cached-detection work that
     does not need it, and continue.
5. Run the identity checks (handoff §8) before any experiment.
6. Continue with handoff §6 in order: provenance re-run → E11 → E13 → E12 →
   Stage D → stress tests → tests → bootstrap → freeze → predeclared
   untouched external validation → final benchmark.

## Rules
- **Loop:** diagnose → hypothesize → implement → test → compare → analyse →
  accept/reject → refine.
- **Ledger:** record every experiment (ID, hypothesis, mechanism, why it
  generalises, expected effect, systems, metrics, per-sequence notes,
  verdict, failure cases, next implication). Never delete failed
  experiments.
- **After every meaningful experiment batch:**
  - update the ledger, `V7_DEV_RESULTS.json` (`tools/v7/collect.py` merges),
    `CODEX_HANDOFF_V7.md` and `V7_CLOUD_RUNS.md`;
  - commit; `git push origin universal-adapters-v1` (never force).
- **Names:** every code change gets new system/run names. Results are
  stamped with a code hash.
- **Causality:** thresholds and regime use frames < t only; the match
  tolerance may use the frame-t image motion cue; frame-t data update the
  state for t+1.
- **No shortcuts:** no labels at runtime; no detector/tracker/dataset/sequence
  names or branches; no sequence-specific rescue rules.
- **Development hosts:** SparseTrack and BoostTrack are DEVELOPMENT hosts
  (never external evidence for V7).
- **Reserved:**
  - TOPICTrack (IEEE TIP 2025) and VisDrone confirmation-16 stay untouched
    until V7 is frozen and the external set is predeclared;
  - 2–4 strong 2025/2026 Q1/Q2 systems with official code and weights,
    verified with sources.
- **Selection priority:**
  1. prevent catastrophic failures;
  2. avoid negative transfer;
  3. MOTA;
  4. HOTA;
  5. IDF1;
  6. IDS;
  7. FP/FN balance;
  8. calibration robustness;
  9. floor robustness;
  10. overhead.

  Never select on one benchmark.
- **Before the freeze:**
  - tests: causality, reset, pass-through identity, name leakage,
    transforms (temp2, temp05, pow3, scale05), floors (0.01/0.05/0.1/0.2),
    crowd and duplicate cases, replay/live parity, deterministic reruns,
    sensitivity;
  - paired bootstrap: 10,000 resamples, seed 42, 95% CIs, with effect sizes.
- **Freeze:**
  - config `configs/universal_acmot_policy_v7.json`;
  - lock `research/V7_POLICY_LOCK.json`;
  - tag `universal-acmot-v7-freeze`;
  - update the reproducibility files.

  After that, no policy changes based on external results; report ALL
  outcomes.
- **Compute:**
  - use cloud CUDA if available, otherwise cloud CPU;
  - prefer cached detections;
  - never use the owner's local Mac/MPS unless the owner explicitly asks at
    the final benchmark stage;
  - GPU-bound work that cannot run is recorded as DEFERRED with its exact
    command, and the cycle continues.
- **Timing:** only after the freeze, on one fixed device: Baseline vs
  Baseline + frozen V7, batch 1, sequential, live detector. Report:
  - detector, controller and tracker latency;
  - end-to-end mean and P95, and FPS (≥ 30 FPS is the real-time reference);
  - V7 overhead in ms and %;
  - precision, resolution, exact hardware.

  Never convert FPS between devices.
- **Documents:** maintain `research/final/V7_{METHOD, EXPERIMENT_LEDGER,
  FAILURE_EVOLUTION, RESULTS, ABLATION, EXTERNAL_SELECTION, EXTERNAL_TRANSFER,
  STATISTICS, REALTIME, REPRODUCIBILITY, PAPER_CLAIMS, FINAL_SUMMARY}.md`
  and `V7_CLOUD_RUNS.md`.
- **Stopping:** only when the cycle is complete, or at a genuine hard block.
  Keep the handoff current at all times; assume the session may end after
  any commit.

## Final report (at completion)
It must cover:
- the V7 architecture;
- the V6 failure → V7 redesign story;
- experiments, including rejected ones;
- SparseTrack and BoostTrack before/after (development evidence);
- VisDrone, UAVDT, Faster R-CNN, RT-DETR, YOLOv8n;
- detector, tracker and dataset transfer;
- untouched external results;
- MOTA/HOTA/IDF1/IDS/FP/FN with CIs;
- per-sequence findings;
- runtime and hardware;
- failure cases and negative results;
- the freeze commit and tag;
- reproducibility commands;
- supported and unsupported claims;
- limitations;
- whether the Master's contribution is complete.
