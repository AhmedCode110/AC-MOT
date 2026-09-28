# CODEX HANDOFF — Universal AC-MOT (written 2026-09-28 by the interactive Claude session)

## Current authoritative state
- **Final method:** V6-TF, FROZEN. Tag `universal-acmot-v6-freeze` →
  commit `2cff95f8a565fdff17f5b1fb06c23cf78e0ffed0`, branch
  `universal-adapters-v1`, remote github.com/AhmedCode110/AC-MOT.
- **Policy file:** `configs/universal_acmot_policy_v6tf.json`. Lock (file
  hashes): `research/V6TF_POLICY_LOCK.json`. Transfer lock:
  `research/TRANSFER_LOCK_V6TF.json`.
- **Method:** research/final/FINAL_METHOD.md. It combines IoU-0.5 duplicate
  suppression, nested exact-Otsu bands on the pooled logits of frames
  t−10..t−1, ECDF order within band, and motion-conditioned association.
- **Development ledger:** research/final/EXPERIMENT_LEDGER.md. X1–X5j were
  developed on val-7; X5 = V6-TF.
- **Superseded:** E41 / V5-TF was rejected by audit (FX-18) and never
  frozen. V4 is a baseline only. Do NOT restart the old V5-TF supervisor
  (`tools/start_autonomous_v5tf.sh`); it targets E41.
- **Single-writer lock:** `outputs/autonomous_v5tf/repo_writer.lock/pid`
  (holder is a `sleep` process). Acquire it before editing tracked files.
- **Unknown-provenance WIP** that predates this work is in git stash
  `51a46971` (D-028). Never pop it into an evaluation commit.

## Completed (all post-freeze runs happened once, under the lock)
- confirmation-16 (E48)
- test-dev post-hoc (E49)
- UAVDT (E50)
- BoT-SORT (E51)
- Faster R-CNN val-7 (E52, partial)
- official-compatible evaluator (E47)
- live == replay parity (E46)
- 12 tests pass

Results: research/final/FINAL_RESULTS.md, TABLES/, FIGURES/. Claims:
PAPER_CLAIMS.md.

## Rejected experiments
- E41 (FX-18)
- X4 jitter band (FX-19)
- X5j
- F5 scene-adaptive resolution (FX-17)
- learned S3 controllers (FX-12/13)

## Current metrics (headline)
See FINAL_SUMMARY.md.
- Confirmation-16: V6-TF has 1 catastrophic cell vs V4 3. YOLOv8n ≈ V4;
  RT-DETR official HOTA +2.78 and MOTA +4.62 vs V4 (significant).
- Faster R-CNN val-7: official HOTA +2.52 and IDF1 +4.08 vs V4; MOTA +16.9
  vs shared static.

## Status (updated 2026-09-28, end of the external stage)
All post-freeze work is DONE:
- confirmation-16 (E48)
- test-dev post-hoc (E49)
- UAVDT (E50)
- BoT-SORT (E51)
- Faster R-CNN on val-7, test-dev and UAVDT (E52)
- External published systems:
  - BoostTrack (E53, MVA 2024): faithful execution; frozen V6-TF worse;
  - SparseTrack (E54, IEEE TCSVT 2025, headline): faithful execution;
    frozen V6-TF significantly worse (HOTA −4.15 [−5.54, −1.66]).

  Both are diagnosed (FAILURE_ANALYSIS F-K, F-L) and kept as negative
  results.

External assets:
- Location: `/Users/ahmedgouda/Desktop/acmot_external/`
  - `SparseTrack/`, `BoostTrack/`, `TOPICTrack/`;
  - venv (Python 3.12, torch 2.14 MPS, detectron2 0.6);
  - `data_mirror/MOT17` (byte-identical val-half mirror of the owner's
    Drive copy);
  - `runs/`, `reports/`.
- Patches, shim and reports are copied into `tools/v6/external/vendor/`.

Remaining (owner's pre-paper step):
- Amendment-5f T4 fidelity gate and official GPU timing.
- Optional TOPICTrack external run (IEEE TIP 2025; weights ready; same
  detector checkpoint).

## Exact commands
See research/final/EXTERNAL_PAPER_TRANSFER.md §20 and REPRODUCIBILITY.md.

## Important files
- `tools/v6/dev.py`: runner with the guard, lock verification and manifest.
- `tools/v6/confirm_report.py`: pooled metrics and bootstrap.
- `tools/v6/eval_official.py`: official-compatible evaluator.
- `tools/v6/make_tables.py`, `tools/v6/live_replay_parity.py`.
- `tests/test_v6_adaptive_layer.py`.
- `universal_policy_pipeline.py` and `online_calibration.py`: FROZEN; any
  change breaks the lock.

## Protected / must not be touched for tuning
- Nothing may change the frozen policy. `dev.py` refuses protected runs if
  any locked file hash differs.
- Confirmation-16, test-dev, UAVDT, Faster R-CNN and BoT-SORT results must
  never feed back into the method.
- No second confirmation run.

## Scientific constraints
research/context/HARD_CONSTRAINTS.md (C0 final target V6-TF; training-free;
causal frames < t; no category-E constants; honest reporting, negative
results kept).

## What Codex must do next
Finish in-progress item 1 above, then prepare the external transfer (item 2)
only after the owner approves the downloads. Reproduce the published
baseline first, then attach the SAME frozen layer through an
integration-only adapter and evaluate once. Keep research/final/ and
research/context/ consistent; run the health check; commit.
