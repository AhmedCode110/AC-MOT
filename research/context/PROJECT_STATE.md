# PROJECT STATE — read this first

<!-- AUTO:BEGIN (tools/update_project_context.py — do not edit by hand) -->
LAST VERIFIED COMMIT: b4a8052740f851810d961ec6695ba660f08c01d6
CURRENT BRANCH: universal-adapters-v1 (upstream origin/universal-adapters-v1, ahead 14, behind 0)
LAST CONTEXT UPDATE: 2026-09-27 20:23 UTC
LATEST FROZEN TAG (git): universal-acmot-v4-freeze → fc003bf
WORKING TREE: 9 uncommitted path(s) — see GIT_STATE.md
<!-- AUTO:END -->

FINAL TARGET: V5-TF — the final research target and contribution (Amendment 7). V4 = historical baseline / ablation ONLY, never a fallback.
CURRENT RESEARCH VERSION: V5-TF (training-free online self-calibrating Universal AC-MOT) — EXPERIMENTAL, not frozen
CURRENT PHASE: V5-TF pre-freeze rule revision — E36/E39/E41 complete; E41 exact-Otsu produced 80/80 development artifacts, 15 catastrophic cells, and +0.69% worst-detector relative gain vs V4; policy lock and T4 fidelity gate remain pending

## What is the project?
Universal AC-MOT: an adaptive-control (AC) layer placed around a FROZEN
detector and a FROZEN tracker for multi-object tracking in drone video. It
adapts detector-side operation (resolution under a compute budget), candidate
handling and tracker association from causal, online-normalised scene/state
observations. It evolved from the original AC-MOT (YOLOv8n + ByteTrack +
handcrafted Scene Complexity Index, tag v1.0.0-acmot-frozen) into a
detector/tracker-agnostic plug-and-play layer.

## Final system (the contribution) = V5-TF
training-free · online self-calibrating · scene/state adaptive ·
detector-agnostic · tracker-agnostic · plug-and-play · real-time · causal.
Target claim (only as far as evidence supports; state exactly what was tested):
"A training-free, causal, plug-and-play adaptive control framework for
heterogeneous detector–tracker MOT pipelines that self-calibrates online from
the incoming stream and requires no retraining or labeled calibration data."
Research question reported alongside: how much of the adaptive benefit does
training-free V5-TF capture relative to V4 (tuned baseline) and to the learned
S3 upper bound? The answer is reported; it never changes which system is final.

## Architecture (see ARCHITECTURE.md)
Frame → Scene/State Analyzer → Online Self-Calibration → Scene State vector →
Universal AC Controller → Compute/Latency Constraint (R-res: size-state
resolution under budget) → Detector Adapter → Detector → Universal
Score/Candidate Handling (ECDF + 3-class Otsu bands on logits) → Tracker
Adapter (F3: motion-aware association) → Tracker → Tracks → causal feedback.

## Status by label
- FINAL TARGET (EXPERIMENTAL, not frozen): Version: V5-TF — selectable
  families F3 (Otsu bands + motion-aware association) and F5 (F3 + R-res);
  F1/F2 = ablations; F5R = random-resolution control; F4 dropped.
- FROZEN, BASELINE/ABLATION ONLY: Version: V4 (tag universal-acmot-v4-freeze).
- FROZEN, SUPERSEDED (history): Version: V1, Version: V3, legacy AC-MOT
  (v1.0.0-acmot-frozen).
- SUPERSEDED, RESEARCH-ONLY: Version: V5 learned controller (S3 C1/C2/C3) =
  research upper bound, never deployable.

## Datasets seen / protected (details: DATASETS_AND_SPLITS.md, PROTECTED_EVALUATIONS.md)
- Development (seen): VisDrone2019-MOT-val (all V1–V5 development);
  VisDrone2019-MOT-train development-40 (V5-TF E36/E39 complete).
- Seen once: VisDrone2019-MOT-test-dev (V4 held-out, E31) → post-hoc only for V5-TF.
- PROTECTED until V5-TF freeze: train confirmation-16; Faster R-CNN quality;
  BoT-SORT quality for V5-TF; UAVDT quality.

## Exact next step
Record E41 as the validated candidate-band revision, write the V5-TF policy
lock, then run the fixed-threshold T4 fidelity gate; protected quality
evaluations remain forbidden until the freeze tag.

## Must NOT be changed
HARD_CONSTRAINTS.md (C0 final target = V5-TF; training-free; causality; no
category-E constants; protected evaluations; fidelity-gate constants;
immutable history).
