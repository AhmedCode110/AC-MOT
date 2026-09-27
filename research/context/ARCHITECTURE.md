# ARCHITECTURE

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
| Version: V4 (fc003bf) | FROZEN, ABLATION | V3 core, SCI removed, resolution = compute budget, global τ/s/offsets by nested LOSO | held-out test-dev once (E31) |
| Version: V5 (62111e8…e6d48b7) | SUPERSEDED as final method; research upper bound | scene-state vector + learned per-target controller (C1 stump, C2 tree, C3 Optuna linear score) | val attempts not adopted (Amend. 5b/5c); forbidden as deployed method (Amend. 6) |
| Version: V5-TF (3684684, 71faf44) | CURRENT, EXPERIMENTAL | training-free online self-calibration | under validation |

## C. V4 compute-budget architecture (FROZEN, ABLATION)
Frame → ResolutionBudget (largest level meeting target FPS; fixed 640/736/832
in experiments) → Detector Adapter (score floor 0.01, NMS request 0.45) →
ECDF normaliser (frames < t) → z-logit gate: z = (logit s − EMA leader
logit)/IQR, demote if z < −τ, τ 0.75 → sensitivity s 0.4 → tracker thresholds
(association offset 0.10, birth = association) → ByteTrack/BoT-SORT
(retention 45, match 0.86). Policy: `configs/universal_acmot_policy_v4.json`,
code `universal_policy_pipeline.py`, entry `universal_acmot.py`.
No scene cue in the decision path.

## D. Target: V5-TF training-free generalized AC (CURRENT, EXPERIMENTAL)
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
          │   Compute/Latency Constraint (resolution budget; 736 in experiments)
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
r_t = motion / rolling-median motion; F4 dropped (identical to F1, 71faf44).
Choice on development-40: fewest catastrophic cells → worst-detector relative
½(HOTA+IDF1) vs V4 → simplicity (F1<F2<F3).

Implementation status (verified by reading code at 71faf44): the rules use
only Otsu bands (F1/F2) and the motion ratio (F3). The full robust-z Scene
State vector of Amendment 6 is only partly logged (`tf_otsu_eta`,
`tf_motion_ratio`, `tf_n_primary`, plus `scene_state.py` EMA cues, which still
reference Z_REF = 0.75 = V4 τ for logging). No rule currently consumes the
density/size/association components → the "scene-state adaptive" claim for
V5-TF rests on F3 only: NEEDS VERIFICATION/decision after dev validation.

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
