# PROJECT STATE — read this first

<!-- AUTO:BEGIN (tools/update_project_context.py — do not edit by hand) -->
LAST VERIFIED COMMIT: b344434f4e0d2e2b0f81bae469f364945738b8bb
CURRENT BRANCH: universal-adapters-v1 (upstream origin/universal-adapters-v1, ahead 22, behind 0)
LAST CONTEXT UPDATE: 2026-09-28 09:37 UTC
LATEST FROZEN TAG (git): universal-acmot-v6-freeze → 2cff95f
WORKING TREE: 37 uncommitted path(s) — see GIT_STATE.md
<!-- AUTO:END -->

FINAL TARGET: V6-TF — the final research target and contribution (Amendment 9; successor of V5-TF after the E41 audit). V4 = historical baseline / ablation ONLY, never a fallback.
CURRENT RESEARCH VERSION: V6-TF (training-free online self-calibrating Universal AC-MOT, nested-Otsu candidate bands) — FROZEN (tag universal-acmot-v6-freeze → 2cff95f)
CURRENT PHASE: V6-TF freeze → one-way post-freeze evaluations (confirmation-16, Faster R-CNN, BoT-SORT, UAVDT, val official-compatible, test-dev post-hoc) → external published-system transfer → paper package (research/final/)

## What is the project?
Universal AC-MOT: an adaptive-control (AC) layer placed around a FROZEN
detector and a FROZEN tracker for multi-object tracking in drone video. It
controls which detector candidates reach the tracker and with which role
(birth/association vs extension-only), and the tracker's association
tolerance, from causal, online self-calibrated statistics of the stream. It
evolved from the original AC-MOT (YOLOv8n + ByteTrack + handcrafted Scene
Complexity Index, tag v1.0.0-acmot-frozen) into a detector/tracker-agnostic
plug-and-play layer.

## Final system (the contribution) = V6-TF
training-free · online self-calibrating · causal · detector-agnostic ·
tracker-agnostic · plug-and-play · real-time (Mac CPU overhead ≈ 4 ms/frame;
official T4 timing deferred, Amendment 9 §5).
Mechanisms (research/final/FINAL_METHOD.md): IoU-0.5 duplicate suppression;
nested exact-Otsu bands (background | extension | primary) on the pooled
detector logits of frames t−10..t−1; order-only ECDF score within band;
motion-conditioned association tolerance (F3 rule).
E41 (V5-TF lock) was rejected by audit (FX-18) and is never frozen.

## Status by label
- FINAL TARGET: Version: V6-TF (config configs/universal_acmot_policy_v6tf.json,
  lock research/V6TF_POLICY_LOCK.json).
- FROZEN, BASELINE/ABLATION ONLY: Version: V4 (tag universal-acmot-v4-freeze).
- SUPERSEDED DEVELOPMENT HISTORY: Version: V5-TF (F1–F5, E41; never frozen).
- FROZEN, SUPERSEDED (history): Version: V1, Version: V3, legacy AC-MOT
  (v1.0.0-acmot-frozen).
- SUPERSEDED, RESEARCH-ONLY: Version: V5 learned controller (S3 C1/C2/C3).

## Datasets seen / protected (details: DATASETS_AND_SPLITS.md, PROTECTED_EVALUATIONS.md)
- Development (seen): VisDrone2019-MOT-val (V1–V4 and the V6-TF iteration
  sandbox); VisDrone2019-MOT-train development-40 (V5-TF E36/E39/E41; V6-TF
  robustness check only, not iterated on).
- Seen once: VisDrone2019-MOT-test-dev (V4 held-out, E31) → post-hoc only.
- PROTECTED until tag universal-acmot-v6-freeze: train confirmation-16;
  Faster R-CNN quality; BoT-SORT quality; UAVDT quality.

## Exact next step
See NEXT_STEPS.md (NEXT EXACT ACTION) and research/final/.

## Must NOT be changed
HARD_CONSTRAINTS.md (C0 final target = V6-TF; training-free; causality; no
category-E constants; protected evaluations; immutable history); the frozen
V6-TF policy after the tag.
