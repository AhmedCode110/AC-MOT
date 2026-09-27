# PROJECT STATE — read this first

<!-- AUTO:BEGIN (tools/update_project_context.py — do not edit by hand) -->
LAST VERIFIED COMMIT: 71faf441beb8fe910cf757d833f683c311c7c7d1
CURRENT BRANCH: universal-adapters-v1 (upstream origin/universal-adapters-v1, ahead 0, behind 0)
LAST CONTEXT UPDATE: 2026-09-27 06:54 UTC
LATEST FROZEN TAG (git): universal-acmot-v4-freeze → fc003bf
WORKING TREE: 16 uncommitted path(s) — see GIT_STATE.md
<!-- AUTO:END -->

CURRENT RESEARCH VERSION: V5-TF (training-free online self-calibrating Universal AC-MOT) — EXPERIMENTAL, not frozen
CURRENT PHASE: V5-TF development validation on VisDrone2019-MOT-train development-40 (waiting for train caches)

## What is the project?
Universal AC-MOT: an adaptive-control (AC) layer placed around a FROZEN
detector and a FROZEN tracker for multi-object tracking in drone video. It
adapts detector-side operation, candidate handling and tracker association
from causal scene/state observations. It evolved from the original AC-MOT
(YOLOv8n + ByteTrack + handcrafted Scene Complexity Index, frozen 2026-09-11,
tag v1.0.0-acmot-frozen) into a detector/tracker-agnostic plug-and-play layer.

## Current research question
Can a training-free, causal, online self-calibrating AC controller capture most
of the adaptive benefit (vs. V4 compute-only and vs. a learned upper bound)
without dataset-specific training? Target claim (only if supported): "A
training-free, causal, plug-and-play adaptive control framework for
heterogeneous detector–tracker MOT pipelines that self-calibrates online from
the incoming stream and requires no retraining or labeled calibration data."

## Current architecture (target, see ARCHITECTURE.md)
Frame → Scene/State Analyzer → Online Self-Calibration → Scene State vector →
Universal AC Controller → Compute/Latency Constraint → Detector Adapter →
Detector → Universal Score/Candidate Handling (ECDF + 3-class Otsu bands on
logits) → Tracker Adapter → Tracker → Tracks → causal feedback.

## Status by label
- FROZEN: Version: V1 (tag universal-acmot-v1-freeze, SUPERSEDED), Version: V3
  (universal-acmot-v3-freeze, SUPERSEDED), Version: V4
  (universal-acmot-v4-freeze — latest frozen Universal version; now an
  ABLATION/reference), legacy AC-MOT (v1.0.0-acmot-frozen).
- EXPERIMENTAL (current): Version: V5-TF — families F1/F2/F3 implemented in
  commit 71faf44 (F4 dropped: identical to F1). Nothing fitted, not yet validated.
- SUPERSEDED as final direction: Version: V5 learned scene-adaptive controller
  (S3 C1/C2/C3) → research upper bound only (Amendment 6).
- ABLATION: Version: V4 compute-budget-only.

## Datasets seen / protected (details: DATASETS_AND_SPLITS.md, PROTECTED_EVALUATIONS.md)
- Development (seen): VisDrone2019-MOT-val (all V1–V5 development);
  VisDrone2019-MOT-train development-40 (V5-TF validation, no results yet).
- Seen once: VisDrone2019-MOT-test-dev (V4 held-out, E31) → post-hoc only for V5-TF.
- PROTECTED until V5-TF freeze: train confirmation-16; Faster R-CNN quality;
  BoT-SORT quality for V5-TF; UAVDT quality.

## Exact next step
See NEXT_STEPS.md. In short: let the Mac cache queue finish; the waiter then
runs `tools/v5tf_dev.py run` (F1/F2/F3 + references on development-40, 736);
then `tools/v5tf_dev.py report` applies the declared lexicographic family choice.

## Must NOT be changed
HARD_CONSTRAINTS.md (training-free final AC; causality; no category-E
constants; protected evaluations; fidelity-gate constants; immutable history).
