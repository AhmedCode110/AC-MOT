# SESSION HANDOFF — paste this file into a new session to continue

Repo: `/Users/ahmedgouda/Desktop/Universal-ACMOT`, branch
`universal-adapters-v1`, remote github.com/AhmedCode110/AC-MOT.
Current HEAD / dirty state: see GIT_STATE.md (auto-generated) — verify with git.

## Direction
Universal AC-MOT = adaptive-control layer around a FROZEN detector and FROZEN
tracker (drone MOT). Current version under development: V5-TF — the FINAL AC
layer must be TRAINING-FREE, online self-calibrating, causal, plug-and-play,
detector-/tracker-agnostic, real-time (Amendment 6, commit 3684684). No fitted
controller, no VisDrone-tuned constants (category E), no GT at runtime.

## Architecture
Frame → Scene/State Analyzer → Online Self-Calibration → Scene State →
Universal AC Controller (declared rule family) → Compute budget → Detector
Adapter → Detector → ECDF + 3-class Otsu bands on logits (primary /
extend-only / discard) → Tracker Adapter (native defaults; F3 motion-aware
association) → Tracker → Tracks → causal feedback. Code: commit 71faf44
(`online_calibration.py`, `universal_policy_pipeline.py`, `tools/v5tf_dev.py`).

## Frozen state
Latest frozen Universal version: V4 (tag universal-acmot-v4-freeze, fc003bf),
held-out test-dev evaluated once (E31) — now an ABLATION/reference.
V1, V3 frozen & superseded; legacy AC-MOT v1.0.0-acmot-frozen.

## Protected (no quality metrics before V5-TF freeze)
confirmation-16 (train), Faster R-CNN, BoT-SORT (for V5-TF), UAVDT.
test-dev = post-hoc only. Val = secondary only. Fidelity-gate thresholds are
protocol constants.

## Latest findings
Legacy SCI ≤ random (E24–E26); learned V5 controllers on val lose ≈2
½(HOTA+IDF1) points vs V4 (E34–E35, overfitting by scarcity); S2 candidate
cues: det_gap, det_count, trk_survival, trk_match, img_motion,
img_motion_resp; rejected: edges, brightness, blur. V5-TF: no results yet.

## Running jobs (as of 2026-09-27 06:53 UTC; verify with `ps -Aww -o pid,etime,command`)
- `tools/mac_cache_queue_v5tf.sh` — train caches then transfer caches (caching only).
- waiter → `tools/v5tf_dev.py run` when train caches reach 56/56.

## Next exact step
NEXT_STEPS.md → wait for E36 (V5-TF dev validation), then report + sensitivity
of flagged constants, then T4 gate, freeze, confirmation-16 once.

## Prohibitions
No training of the AC layer; no `v5_train.py final`; no protected metrics;
no relaxing gate thresholds; no editing frozen versions/locks; no invented
facts (mark UNKNOWN / NEEDS VERIFICATION).
