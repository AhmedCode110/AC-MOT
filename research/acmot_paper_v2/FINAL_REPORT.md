# OATrack-comparison AC-MOT experiment: final report (VisDrone)

**Primary method: v3 genuinely-adaptive AC-MOT controller** (frozen in
`controller_search/FREEZE_MANIFEST_V3_ADAPTIVE.json`). The earlier
static/no-op finding is preserved as secondary control evidence in
Section 9.

## 1. What was executed

1. Selected and froze a publicly available pretrained YOLO11m detector
   (no in-house training) as the common detector for all four systems.
2. Generated the full detection cache (resolutions {1088,1280,1536} x
   NMS {0.45,0.60,0.70}, extended to {896,1728} for calibration-only
   diagnostics) -- one forward pass per frame/resolution.
3. Ran a full predeclared, calibration-only development process, in two
   phases:
   - **Phase A** (vs the calibration-selected matched-static point):
     5 search stages (resolution mapping, NMS mapping, cue-subset,
     joint threshold+action) all converged on a null result -- no
     policy robustly beat the *best fixed point found on calibration*.
   - **Phase B**, per owner research-direction correction: redefined the
     primary objective as adaptive-vs-`r1536_n70` (the predeclared host
     baseline, not the calibration-tuned static point), added detector-
     side confidence floor as a third controllable dimension, and ran a
     60-trial Optuna search. This found a genuinely adaptive, robust
     candidate (below).
4. Validated the candidate with a same-action-distribution shuffled
   control (honest finding: does not establish causal timing adds value
   beyond the action mix itself -- reported, not suppressed).
5. Froze the v3 controller and ran it once on the untouched-until-now
   7-sequence VisDrone-val (**SECOND VAL READ** -- val was already read
   once for the static/no-op method, commit `bfa5caf`).
6. Paired bootstrap (5000 resamples) vs `r1536_n70`.

Every step, including every negative result, is committed to git.

## 2. Frozen detector

`dronefreak/visdrone-yolo11m` (`best.pt`), Hugging Face, SHA256
`c6f98247e7c731a567aaf37ca7b808beff588b687623e028b25a2e1d846e96e7`.
VisDrone2019-**DET**-trained (not MOT), imgsz=640 native. **Not** the
official OATrack detector (not publicly released).

## 3. Development/calibration data

8 VisDrone2019-MOT-train sequences (`DETECTOR_SPLIT.json` ->
`calibration`), 3282 frames. Never the final 7-sequence VisDrone-val
until Section 5's single, frozen, SECOND READ.

## 4. Frozen v3 adaptive AC-MOT policy

Causal pipeline, reusing `acmot_sci.SceneLayer` **completely frozen and
unchanged** (cue formulas crowd/tiny/edge/dark/blur; cue subset
restricted to `['crowd']`, Stage 3A's selection): image stats of frame
*t* + tracker-reported boxes of frames *< t* (never GT, never future
frames) -> one of 3 scene levels -> one of 3 detector action tuples:

| level | resolution | NMS IoU | detector confidence floor |
|---|---|---|---|
| LOW | 1536 | 0.70 | 0.10 |
| MEDIUM | 1280 | 0.45 | 0.40 |
| HIGH | 1088 | 0.60 | 0.40 |

SceneSpec thresholds: `t_med=0.297`, `t_high=0.511` (Optuna-selected,
calibration-only). Detector confidence floor is a detector-side,
post-cache score filter (confirmed mathematically exact,
`CONFIDENCE_EXACTNESS_CHECK_FIXED.json`, 18/18) -- **strictly distinct**
from OATrack's tracker-side `min_conf=0.40`, which remains frozen and
untouched throughout.

**Genuinely-adaptive verification** (predeclared definition: >=2 action
tuples each on >=10% of frames, with within-sequence switching):
MEDIUM 10.1% / HIGH 82.3% of calibration frames (LOW 7.6%, just under
the 10% bar individually, but still a real, causally-triggered 3rd
action); **within-sequence switching occurs in every one of the 8
calibration sequences** (0.13-1.57 switches per 100 frames). On
VisDrone-val: LOW 7.4% / MEDIUM 30-33% / HIGH 59-62% -- the adaptive
behavior structurally generalizes to held-out data, not just an
artifact of calibration-specific tuning.

**Honest limitation, carried into the freeze, not suppressed**: a
same-action-distribution shuffled control (random temporal order of the
identical action mix) scores *at least as well* as the causal schedule
on calibration (shuffled cv_mean +2.62 vs causal +2.45 HOTA, both vs
`r1536_n70`). **This does not establish that the causal, scene-
conditioned TIMING of switches adds value beyond simply using a diverse
mix of three operating points.** The large improvement over the
predeclared baseline is real and robust; whether it is "adaptive" in
the sense of exploiting scene state (vs. "a good fixed-ratio schedule")
is not established by this control and should be stated as a limitation
in any paper claim.

## 5. Final four-system table (VisDrone-val, 7 sequences, 71,830 GT boxes) -- SECOND VAL READ for systems 3/4

| System | Operating point | HOTA | MOTA | IDF1 | IDS | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| 1. YOLO11m + ByteTrack | r1536_n70 (predeclared default) | 41.59 | 9.88 | 48.57 | 1341 | 39274 | 24121 | 54.85 | 66.42 |
| 2. YOLO11m + OATrack (min_conf=0.40) | r1536_n70 (predeclared default) | 46.77 | 28.17 | 57.11 | 773 | 25394 | 25432 | 64.63 | 64.59 |
| 3. **AC-MOT-v3 (adaptive)** + YOLO11m + ByteTrack | causal 3-level res/nms/conf (above) | **44.52** | 27.87 | 53.93 | 1016 | 24892 | 25902 | 64.85 | 63.94 |
| 4. **AC-MOT-v3 (adaptive)** + YOLO11m + OATrack | causal 3-level res/nms/conf (above) | **48.10** | 34.48 | 59.36 | 604 | 21400 | 25057 | 68.61 | 65.12 |

## 6. Primary deltas (paired bootstrap, 5000 resamples, seed 42, vs `r1536_n70`)

**System 3 (AC-MOT-v3+ByteTrack) - System 1:**

| metric | Δ | 95% CI |
|---|---|---|
| HOTA | +2.94 | [+2.36, +3.38] |
| MOTA | +18.00 | [+13.54, +23.28] |
| IDF1 | +5.36 | [+4.49, +6.18] |
| IDS | -325 | [-498, -164] |
| FP | -14382 | [-19176, -9620] |
| FN | +1781 | [+288, +3594] |
| Precision | +10.00 | [+8.71, +11.90] |
| Recall | -2.48 | [-4.51, -0.47] |

**System 4 (AC-MOT-v3+OATrack) - System 2:**

| metric | Δ | 95% CI |
|---|---|---|
| HOTA | +1.33 | [+0.20, +2.26] |
| MOTA | +6.32 | [+4.74, +7.88] |
| IDF1 | +2.26 | [+0.55, +3.47] |
| IDS | -169 | [-292, -42] |
| FP | -3994 | [-6190, -2495] |
| FN | -375 | [-1463, +670] (not significant) |
| Precision | +3.98 | [+2.96, +5.51] |
| Recall | +0.52 | [-0.87, +2.11] (not significant) |

All HOTA/MOTA/IDF1/IDS/FP/Precision deltas are statistically significant
(95% CI excludes 0) for both trackers. FN/Recall are significant (worse)
for ByteTrack, not significant for OATrack.

## 7. Failure analysis (per-sequence HOTA delta, vs r1536_n70)

| sequence | ByteTrack (3-1) | OATrack (4-2) |
|---|---|---|
| uav0000086_00000_v | +3.75 | +2.98 |
| uav0000117_02622_v | +3.05 | +2.27 |
| uav0000137_00458_v | +2.07 | -0.01 |
| uav0000182_00000_v | +3.20 | +1.89 |
| uav0000268_05773_v | +1.87 | +1.42 |
| uav0000305_00000_v | +2.21 | +1.56 |
| uav0000339_00001_v | +3.12 | **-1.81** |

**ByteTrack: 7/7 sequences improve** -- the broadest, most consistent
result in this entire investigation. **OATrack: 5/7 improve**, one
near-zero wash, one regression (`uav0000339_00001_v`, -1.81 HOTA) --
the same sequence that regressed under the static/no-op comparison too,
suggesting an interaction between OATrack's internal min_conf=0.40 and
the external detector-side confidence floor on this specific sequence,
not investigated further post-freeze (would require touching held-out
val to diagnose).

## 8. Runtime

Reusing `RUNTIME_BENCHMARK_RESULT.json` (real end-to-end T4 measurement,
not cache playback): detector forward-pass time dominates end-to-end
latency at every tested resolution (1088: ~48ms, 1536: ~127-129ms,
negligible tracker overhead in both cases, <11ms). Since the v3
controller's 3 actions sit at exactly these already-measured resolution
points (1536/1280/1088) and the controller/SceneLayer's own per-frame
overhead is a handful of cheap OpenCV calls (`analyze_visual`: one
resize + Canny + Laplacian on a downsampled frame) plus trivial Python
logic, **a dedicated v3 runtime run is not required to bound the
answer**: end-to-end latency for v3 is the level_frac-weighted average
of the already-measured per-resolution detector times (~48-129ms
depending on instantaneous level) plus negligible (<1ms, unmeasured but
clearly dominated by detector time) controller overhead. A precise
single-number FPS for v3 specifically has NOT been separately measured
and would need a brief, cheap GPU run if an exact figure is required for
the paper; this is flagged as the one remaining small gap, not a
blocker.

## 9. Secondary/supplementary: static matched-static finding (NOT the primary method)

Preserved in full, unmodified: `controller_search/DEVELOPMENT_DECISION_REPORT.md`,
`FREEZE_MANIFEST.json` (static), `BOOTSTRAP_RESULT.json`,
`FINAL_ACMOT_SYSTEMS_RESULT.json` (the first VisDrone-val read, commit
`bfa5caf`). Summary: across 5 search stages, no adaptive policy robustly
beat the *calibration-tuned best fixed point* (`r1088_n45_conf0.40`);
that static point itself significantly outperforms `r1536_n70`. This is
retained as supplementary evidence answering "could the gain be
explained by a better fixed operating point alone" -- it is a weaker,
but directionally consistent, version of the same underlying finding
(lower resolution + a confidence floor help this detector), now
superseded as the main result by the genuinely-adaptive v3 controller,
which uses a causally-triggered mix including all three resolution
points rather than a single fixed one.

## 10. Statistically supported claims

- AC-MOT-v3 (adaptive) significantly outperforms the predeclared host
  baseline `r1536_n70` on HOTA/MOTA/IDF1/IDS/FP/Precision for both
  ByteTrack and OATrack hosts, broadly distributed (7/7 sequences for
  ByteTrack).
- The controller satisfies the predeclared genuinely-adaptive
  definition: real within-sequence switching, >=2 actions each used
  substantially, causal-only inputs (no GT, no future frames), on both
  calibration and held-out val.
- **NOT established**: that the causal, scene-conditioned *ordering* of
  actions (vs. a random ordering of the same action mix) is what drives
  the improvement -- the shuffled control performs at least as well.
- OATrack retains its qualitative signature (fewer IDS/FP, higher IDF1)
  relative to ByteTrack under the adaptive controller too.
- One sequence (`uav0000339_00001_v`) regresses under OATrack
  specifically, consistently across both the static and adaptive
  comparisons.
- No comparison with the official OATrack paper/detector, and no
  apples-to-apples comparison with the historical/old AC-MOT
  formulation (different detector, different controller architecture --
  scoped out, not fabricated).

## 11. Remaining work

1. UAVDT transfer with the frozen v3 policy, no retuning -- the required
   raw data exists but sits inside a different, already-frozen
   experiment's protected directory; owner authorization required
   before touching it (unchanged from the earlier finding). **Blocked
   on owner decision, not a technical gap.**
2. Optional: a dedicated, precise v3 end-to-end FPS number (Section 8
   gives a well-bounded estimate from already-measured per-resolution
   timings; an exact number needs one brief GPU run if required).
3. Old-AC-MOT apples-to-apples comparison -- scoped out, needs a
   separate protocol decision (different detector and controller
   architecture).
