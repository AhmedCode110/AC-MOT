# Universal AC-MOT — whole-model parameter / design audit

Question asked of every item: *why is this value/rule valid for a different
detector, tracker, scene distribution and dataset?*

Categories: **A** structural/mathematical · **B** data-derived online ·
**C** globally optimized (nested sequence-level CV, both detectors) ·
**D** API-required adapter parameter · **E** arbitrary/manual legacy.
No E item may survive into the final model.

Evidence IDs refer to research/EXPERIMENT_REGISTRY.md. "Legacy" = inherited
from AC-MOT `core.py` / `generic_controls.py`; the 2026-09-11 decision log
itself calls the SCI numbers "heuristic design priorities / engineering
choices, not learned values".

## 1. Scene analyzer, SCI and scene controller (legacy, used by V1…V3)

| Component | Parameter / rule | Value | Why originally chosen | Evidence | Cat. | Action (V4) |
|---|---|---:|---|---|---|---|
| Scene analysis | analysis interval | every 10 frames | ≈0.33 s at 30 FPS, engineering | none (log: "not optimal") | E | **Removed** with SCI |
| SCI smoothing | rolling mean window | 7 samples | ≈2 s history, engineering | none | E | **Removed** |
| Crowd cue | min(n/30, 1) | 30 | heuristic | E25 ρ sign flips; E26 < random | E | **Removed** (cue rejected) |
| Tiny cue | area < 32×32 px | 1024 px² | heuristic | E25/E26 no consistent benefit | E | **Removed** |
| Edge cue | min(edges/0.14, 1) | 0.14 | heuristic | E13 (visual cues same on 0137/0268), E26 < random | E | **Removed** |
| Darkness cue | brightness < 80 | 80 | heuristic | E26 < random; no night data in val | E | **Removed** |
| Blur cue | Laplacian var < 180 | 180 | heuristic | E26 < random | E | **Removed** |
| SCI weights | .30/.30/.20/.10/.05 (sum .95) | — | "design priorities" | E24: legacy SCI no better than uniform mix; E26 ≤ random | E | **Removed** |
| Feedback filter | normalized conf ≥ 0.18 | 0.18 | legacy raw conf floor | only feeds SCI | E | **Removed** (no SCI) |
| Scene labels | tiny>.5, crowd>.65, edges>.13 | — | heuristic | only feeds SCI/sensitivity bonus | E | **Removed** |
| Recovery probe | peak≥5, n<.4·peak, decay .8, 3 steps → 832 | — | heuristic | part of SCI resolution rule (E24) | E | **Removed** |
| Resolution rule | SCI>.60 or tiny>.5 → 832; SCI>.35 → 736 | — | heuristic | E24, E26 | E | **Replaced** by compute budget |
| Resolution hysteresis | keep 832 if SCI>.5/tiny>.4; 736 if SCI>.25; dwell 30 f | — | heuristic | E24 | E | **Removed** |
| Resolution levels | {640, 736, 832} | — | multiples of stride 32 spanning 0.59–1.0 pixel cost | E24: accuracy monotone in resolution for both detectors | D | Kept as **budget levels**; chosen by deployment budget / latency controller |

## 2. Detector control and adapters

| Component | Parameter / rule | Value | Why | Evidence | Cat. | Action |
|---|---|---:|---|---|---|---|
| Detector adapter | inference score floor | 0.01 | emit (almost) all candidates; policy decides | same in all adapters; policy is order-based | D | Keep (API emission floor) |
| Detector adapter | max detections | 1000 | API cap, never binding at 0.01 floor on val | — | D | Keep |
| Detector adapter | classes | COCO person/car/bus/truck (UAVDT: vehicles) | task definition | — | D | Keep (task, not tuning) |
| Detector adapter | resolution semantics | longest side = level | uniform meaning across APIs | — | D | Keep |
| Detector control | NMS IoU request | 0.45 | legacy AC | pending sensitivity test (E27) | E→? | see E27 |
| Policy | raw score floor inside policy | 0.01 (V1–V2g) → 0 (V3+) | raw-scale constant breaks invariance | E20/E23 | E→A | Removed (0) |

## 3. Score normalization and gate

| Component | Parameter / rule | Value | Why | Evidence | Cat. | Action |
|---|---|---:|---|---|---|---|
| Normalizer | V1: 64-bin histogram, decay .95 | — | legacy | E20: breaks under temperature (saturated bins) | E | Replaced by ECDF |
| Normalizer | ECDF (order-only) | — | exact invariance to monotone recalibration | E23 byte-identical tracks under T=2, 0.5 | A/B | Keep |
| Normalizer | ECDF sampling stride / window | 10 frames / 20 samples (200-frame memory) | same data span as V1 histogram | pending sensitivity (E28) | B (memory length E→?) | E28 |
| Gate | statistic z = (logit s − EMA leader logit)/IQR | — | exact Platt/temperature invariance | E21 AUC 0.77–0.80, E23 | A | Keep |
| Gate | τ | 0.75 (V3) | LOSO 7/7 folds (V3) | E22 | C | Re-selected in V4 nested CV |
| Gate | IQR window | 10 frames | legacy-like | pending (E28) | E→? | E28 |
| Gate | leader EMA decay | 0.9 | legacy-like | pending (E28) | E→? | E28 |
| Gate | demote vs drop | demote | ByteTrack-style low-score extension | E15 (drop fails YOLO-0268) | A (design, evidence) | Keep |
| Density / Top-K | budget 12+36·SCI, Top-K | — | V2b | E18: never selected (gate-only wins) | E | **Removed** |

## 4. Generic sensitivity → tracker thresholds

| Component | Parameter / rule | Value | Why | Evidence | Cat. | Action |
|---|---|---:|---|---|---|---|
| Sensitivity | .22 + .18·SCI (+.05 scene, +.08 recovery), clip [.2,.5] | — | legacy | E24: equals a constant 0.4 | E | Replaced by global s (C) |
| Association threshold | low + 0.18 | 0.18 | legacy | none | E | Global offset (C) |
| Birth threshold | high + 0.05 | 0.05 | legacy | none | E | Global offset (C) |
| Caps | high ≤ .95, new ≤ .98 | — | keep thresholds < 1 in percentile space | bounds | A | Keep |

## 5. Tracker adapter

| Component | Parameter / rule | Value | Why | Evidence | Cat. | Action |
|---|---|---:|---|---|---|---|
| ByteTrack/BoT-SORT | retention (buffer) | 45 (AC) vs 30 (native) | legacy "A1 tuned" | none cross-detector | E | C: {AC, native} in V4 CV |
| ByteTrack/BoT-SORT | association IoU-cost match | 0.86 (AC) vs 0.8 (native) | legacy | none | E | C (same switch) |
| ByteTrack | track_low_thresh | 0.04 (percentile space) | legacy | never binding (eligibility low ≥ 0.4) | A (non-binding) | Keep, documented |
| ByteTrack | fuse_score, frame_rate 30 | native | tracker defaults | — | D | Keep |
| BoT-SORT | GMC sparse optical flow, ReID off | native defaults | tracker defaults | — | D | Keep |

## 6. Evaluation (not part of the method)
IoU 0.5, GT filter {1,4,5,6,9}, score==1, trunc<2, occ<2, class-agnostic:
fixed internal protocol (A), preserved for comparability; not official
VisDrone.

## V4 final parameter table (frozen)

| Group | Parameter | Final value | How obtained | Evidence / stability |
|---|---|---:|---|---|
| 1 online | ECDF normalizer | order-only mid-rank CDF of the detector's own recent scores | estimated online (frames < t) | exact monotone invariance (E23, V4 check) |
| 1 online | gate reference | EMA of the frame-leader logit | estimated online | — |
| 1 online | gate scale | IQR of candidate logits (last 10 frames) | estimated online | makes the gate Platt-invariant |
| 1 online | resolution level (live) | largest level meeting the FPS budget | detector latency probed online | budget = user input |
| 2 optimized | τ (gate) | 0.75 | nested LOSO, both detectors | 5/7 folds; largest τ with 0 catastrophic cells |
| 2 optimized | s (eligible fraction) | 0.4 | nested LOSO | 5/7 folds; same constraint edge |
| 2 optimized | association offset | 0.10 | nested LOSO | 5/7 folds; objective monotone decreasing in offset |
| 2 optimized | tracker retention/match | 45 frames (1.5 s @30 FPS) / 0.86 | nested LOSO switch {AC, native} | 7/7 folds |
| 2 optimized | NMS request | 0.45 | pre-declared rule vs detector-native 0.7 | 6/7 sequences at 736 and 832 (YOLO only; RT-DETR has no NMS) |
| 3 structural | birth threshold | = association threshold | birth offset objective flat (0 = 0.05) | removed as a parameter |
| 3 structural | ECDF memory 10×20, gate window 10, leader decay 0.9 | — | memory lengths; sensitivity analysed | ±0.4 HOTA over 5–40 windows, 0.8–0.95 decay (E28) |
| 3 structural | caps .95/.98, clip 1e-6 | — | keep percentile thresholds inside (0,1) | non-binding bounds |
| 3 structural | resolution levels 640/736/832 | — | compute-budget options (stride-32 multiples) | accuracy monotone in level (E24, E28) |
| 3 API (D) | score floor .01, max_det 1000, class map, longest-side resolution | — | adapter invocation only | identical across adapters |

Removed (were category E): every SCI cue, weight, normalizer, threshold,
smoothing window, analysis interval, scene label, recovery probe,
resolution rule and hysteresis; the SCI→sensitivity mapping; the
density budget / Top-K; the policy raw floor; the birth offset.
No category-E constant remains in V4.
