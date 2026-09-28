# SESSION HANDOFF — paste this file into a new session to continue

Repo: `/Users/ahmedgouda/Desktop/Universal-ACMOT`, branch
`universal-adapters-v1`, remote github.com/AhmedCode110/AC-MOT.
Current HEAD / dirty state: see GIT_STATE.md (auto-generated) — verify with git.

## Direction
FINAL TARGET = V6-TF (Amendment 9, HARD_CONSTRAINTS C0). V4 = historical
baseline / ablation ONLY. V5-TF (F1–F5, E41) = rejected development
history (E41 audit: FX-18). V6-TF is training-free, online self-calibrating,
causal, detector-/tracker-agnostic: IoU-0.5 duplicate suppression +
nested exact-Otsu bands on pooled logits of frames t−10..t−1 + ECDF order
within band + motion-conditioned association. Details:
research/final/FINAL_METHOD.md; development ledger:
research/final/EXPERIMENT_LEDGER.md; runner: tools/v6/dev.py.

## Where things stand
Read NEXT_STEPS.md (NEXT EXACT ACTION) and research/final/. Freeze tag:
universal-acmot-v6-freeze (FROZEN_VERSIONS.md). Protected until the tag:
confirmation-16, Faster R-CNN, BoT-SORT, UAVDT (tools/v6/dev.py refuses).

## Historical notes (V5-TF era, superseded)
### (old) Direction
FINAL TARGET = V5-TF (Amendment 7, HARD_CONSTRAINTS C0). V4 = historical
baseline / ablation ONLY — never a fallback, never a rival final.
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
held-out test-dev evaluated once (E31) — baseline/ablation only.
V5-TF families: F3 / F5 (= F3 + scene-adaptive resolution R-res) selectable;
F1/F2 ablations; F5R random-resolution control (Amendment 7).
V1, V3 frozen & superseded; legacy AC-MOT v1.0.0-acmot-frozen.

## Protected (no quality metrics before V5-TF freeze)
confirmation-16 (train), Faster R-CNN, BoT-SORT (for V5-TF), UAVDT.
test-dev = post-hoc only. Val = secondary only. Fidelity-gate thresholds are
protocol constants.

## Latest findings
Legacy SCI ≤ random (E24–E26); learned V5 controllers on val lose ≈2
½(HOTA+IDF1) points vs V4 (E34–E35, overfitting by scarcity); S2 candidate
cues: det_gap, det_count, trk_survival, trk_match, img_motion,
img_motion_resp; rejected: edges, brightness, blur.

E36/E39 completed 2026-09-27. F3 selected over F5: catastrophic cells 17 vs
18; worst-detector relative ½(HOTA+IDF1) vs V4 −0.339% vs −2.646%. F5 cost
was eligible (0.954/0.952), but random F5R was better (17 cells, −1.683%):
tested scene-adaptive resolution did not add real benefit. F3 is the current
candidate but has 17 catastrophic cells vs V4's 3. E39: history window and
warm-up insensitive A; OTSU_BINS and Otsu window sensitive E, blocking freeze
under C5. Live/replay F3 parity passes exactly on 80/80 frames (E40); v1
harness failure is preserved and explained by process-global track IDs.

E41 exact-Otsu was declared and validated after commit 240eba9: 80/80
development artifacts, 15 catastrophic cells, and +0.69% worst-detector
relative gain vs V4. It does not change the prior F3-vs-F5 choice.

## Running jobs
The E36/E39 waiter is finished. `tools/mac_cache_queue_v5tf.sh` may still be
building allowed transfer caches; verify with `ps -Aww -o pid,etime,command`.
Never restart it while alive. The autonomous supervisor is restarted only
after the interactive writer releases its lock.

## Next exact step
Record the E41 parameter/audit decision, write the V5-TF policy lock, and run
the fixed-threshold T4 fidelity gate. No protected quality evaluation before
the freeze tag.

## Coordination (autonomous supervisor)
`tools/autonomous_v5tf_supervisor.py` holds `outputs/autonomous_v5tf/repo_writer.lock`
while waiting for the E36/E39 waiter, then launches one Codex child. Interactive
sessions must stop it (`tools/stop_autonomous_v5tf.sh`), hold the lock, commit,
release, and restart it (`tools/start_autonomous_v5tf.sh`) — never touching the
waiter or cache queues (D-021). Status: `.venv/bin/python tools/autonomous_v5tf_supervisor.py --status`.
The former `codex exec -a never` incompatibility was fixed in fd44a11 by using
the supported non-interactive bypass flag.

## Prohibitions
No training of the AC layer; no `v5_train.py final`; no protected metrics;
no relaxing gate thresholds; no editing frozen versions/locks; no invented
facts (mark UNKNOWN / NEEDS VERIFICATION).
