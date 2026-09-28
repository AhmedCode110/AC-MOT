# ARCHITECTURE

## CURRENT: V6-TF (final; research/final/FINAL_METHOD.md is authoritative)
Frame → image cue (global motion, frame t) → Detector (compute budget B) →
duplicate suppression (IoU 0.5) → logits → nested exact-Otsu bands from the
pooled logits of frames t−10..t−1 (background | extension | primary) →
ECDF order within band → Tracker adapter (native thresholds 0.5 / 0.1;
motion-conditioned IoU-match tolerance) → frozen tracker → tracks; frame-t
statistics update state for t+1. Pre-detector resolution adaptation (F5)
was rejected (FX-17); resolution is the compute-budget input.
Sections below describe the V5-TF target architecture (history).


Labels: CURRENT · FROZEN · SUPERSEDED · ABLATION · EXPERIMENTAL.
Code pointers are to the working tree at the commit in PROJECT_STATE.md.

## A. Original AC-MOT (FROZEN, SUPERSEDED) — tag v1.0.0-acmot-frozen (2026-09-11)
YOLOv8n + ByteTrack (tuned profile), root modules `core.py`, `experiment.py`,
`evaluate.py`; docs `README_ACMOT_*_2026-09-11.md`.
Frame → scene analysis every 10 frames (crowd n/30, tiny <32² px, edge
density/0.14, darkness brightness<80, blur Laplacian<180) → SCI = weighted sum
(.30/.30/.20/.10/.05), rolling mean 7 → controller picks resolution
{640,736,832} + confidence sensitivity (with hysteresis/recovery probe) →
detector → ByteTrack. All SCI numbers were heuristic (category E; see
research/PARAMETER_AUDIT.md §1). Detector- and tracker-specific.

## B. Intermediate Universal versions (research/EXPERIMENT_REGISTRY.md)
| Version | Label | Key idea | Fate |
|---|---|---|---|
| Version: V1 (e56c2f3) | FROZEN, SUPERSEDED | adapters + online histogram percentile normaliser + leader-relative ratio gate ρ + legacy SCI | not invariant to score temperature (E20) |
| Version: V2b / V2c / V2d–V2f | SUPERSEDED (never frozen) | density budget + Top-K; additive/proportional density thresholds; reliability feedback | failed (E02, E03, E07, E11, E13) |
| Version: V3 (c1e799d) | FROZEN, SUPERSEDED | order-only causal ECDF + z-logit leader gate (demote) τ 0.75; SCI still in path | exact Platt/temperature invariance (E23); SCI unjustified (E24–E26) |
| Version: V4 (fc003bf) | FROZEN, BASELINE/ABLATION ONLY (never final) | V3 core, SCI removed, resolution = compute budget, global τ/s/offsets by nested LOSO | held-out test-dev once (E31) |
| Version: V5 (62111e8…e6d48b7) | SUPERSEDED as final method; research upper bound | scene-state vector + learned per-target controller (C1 stump, C2 tree, C3 Optuna linear score) | val attempts not adopted (Amend. 5b/5c); forbidden as deployed method (Amend. 6) |
| Version: V5-TF (3684684, 71faf44, Amendment 7) | FINAL TARGET, CURRENT, EXPERIMENTAL | training-free online self-calibration + scene-state control | under validation |

## C. V4 compute-budget architecture (FROZEN, BASELINE/ABLATION ONLY — never the final system)
Frame → ResolutionBudget (largest level meeting target FPS; fixed 640/736/832
in experiments) → Detector Adapter (score floor 0.01, NMS request 0.45) →
ECDF normaliser (frames < t) → z-logit gate: z = (logit s − EMA leader
logit)/IQR, demote if z < −τ, τ 0.75 → sensitivity s 0.4 → tracker thresholds
(association offset 0.10, birth = association) → ByteTrack/BoT-SORT
(retention 45, match 0.86). Policy: `configs/universal_acmot_policy_v4.json`,
code `universal_policy_pipeline.py`, entry `universal_acmot.py`.
No scene cue in the decision path.

## D. FINAL TARGET: V5-TF training-free generalized AC (CURRENT, EXPERIMENTAL)
```
 Frame t ─┬─► Scene/State Analyzer (image motion; det/trk stats of frames < t)
          │        │
          │        ▼
          │   Online Self-Calibration (rolling median/MAD, ECDF, Otsu on logits;
          │        │                    all over causal windows, frames < t)
          │        ▼
          │   Scene State vector {density, size, score_reliability η, motion ratio,
          │        │               association/survival}   (robust z vs own history)
          │        ▼
          │   Universal AC Controller (declared rule family F1/F2/F3; no fitted params)
          │        │
          │        ▼
          │   Compute/Latency Constraint (budget B; R-res: size-state tertile →
          │        │                    640/B/832, budget guard)
          ▼        ▼
     Detector Adapter (detector-native suppression) ─► Detector (frozen)
                   │
                   ▼
     Universal Score/Candidate Handling: ECDF rank u; 3-class Otsu on candidate
     logits → primary (score 0.5+0.5u) / extend-only (0.1+0.4u) / discard
                   │
                   ▼
     Tracker Adapter (native defaults; F3: motion-aware association tolerance)
                   │
                   ▼
     Tracker (frozen: ByteTrack | BoT-SORT) ─► Tracks ─► causal feedback (frames > t)
```
Rule families (Amendment 6; code `universal_policy_pipeline.py`
`candidate_mode`/`assoc_motion`, `online_calibration.py`, dev script
`tools/v5tf_dev.py`): F1 Otsu-3 bands over the causal window; F2 within-frame
Otsu; F3 = F1 + match_t = min(0.95, 1 − (1 − m0)/max(1, r_t)),
r_t = motion / rolling-median motion; F4 dropped (identical to F1, 71faf44);
F5 = F3 + R-res (Amendment 7): s̃ = 10-frame median of the size state
(median log area fraction of primary candidates, frames < t), p = causal ECDF
rank of s̃ in its own history; p < 1/3 → 832, p > 2/3 → 640, else budget
level B; if running mean pixel cost ≥ B², 832 is served at B. F5R = same with
random levels (10-frame blocks) — control. Selectable finals: F3, F5 (the
frozen V5-TF must contain scene-state control); F1/F2 = ablations.
Choice on development-40: fewest catastrophic cells → worst-detector relative
½(HOTA+IDF1) vs V4 → simplicity (F3<F5).

Implementation status (Amendment 7 commit): scene-state control = motion
ratio → association tolerance (F3) and size state → resolution before the
detector (R-res, F5; `_tf_level` / `_tf_observe` in
`universal_policy_pipeline.py`). R-res decides at the start of each 10-frame
block, compares the size state only with history observed at the same level
(removes the action's self-induced bias), budget guard at block start. The
V5-TF scene-state vector (size, density z, η z, survival) is logged as
`tf_state_*`; density/η/survival drive no rule. V5-TF families run with
`scene_state=False`, so `scene_state.py` (Z_REF, DECAY) is not in the V5-TF
path. Metric-free crash test (one val cache, 200 frames): F5 pixel cost
0.83–0.97 of 736², 7–13 level switches; F5R ≈0.98.

## Component table (V5-TF)
| Component | Input | Output | Causal dependency | Online-derived | Structural | Adaptive | Learned/fitted params |
|---|---|---|---|---|---|---|---|
| Scene/State Analyzer (`scene_state.py`, `tools/visual_cues.py`) | frame t image (motion via phase correlation vs t−1); detector/tracker outputs of frames < t | raw cues | frames ≤ t | yes | cue definitions | n/a | none |
| Online Self-Calibration (`online_calibration.py`) | cue / logit history | robust z, ratio to median, Otsu T1/T2, η | frames < t (window) | yes | window lengths | yes | none |
| ECDF normaliser (V3/V4) | detector scores history | rank u ∈ (0,1) | frames < t | yes | memory 10×20 | yes | none |
| Universal AC Controller (F1–F3) | scene state, Otsu thresholds | band assignment; association tolerance (F3) | frames ≤ t | yes | rule form | yes | none (declared) |
| Compute/Latency Constraint (`ResolutionBudget`) | target FPS, probed latency | resolution level | online latency | yes | level set 640/736/832 (D) | yes (live) | none |
| Detector Adapter (`adapters/detectors/*`) | frame, level | candidates (floor 0.01, max 1000, native NMS) | frame t | no | API (D) | no | none |
| Tracker Adapter (`adapters/trackers/*`) | banded candidates | tracker update | t | no | API (D), native defaults | F3 only | none |
| Tracker (ByteTrack / BoT-SORT) | detections | tracks | frames ≤ t | — | frozen | — | pretrained/none |
| Detector (YOLOv8n / RT-DETR-L / Faster R-CNN ResNet50-FPN v2) | image | boxes, scores | t | — | frozen COCO weights | — | pretrained, never retrained |

Research-only components (never deployed): `v5_controller.py`,
`tools/v5_s1_headroom.py`, `tools/v5_s2_cues.py`, `tools/v5_s3_nested.py`,
`tools/v5_train.py` (its `final` stage is FORBIDDEN under Amendment 6).
