# SCI + V7f — experiment ledger

Branch `sci-v7f-general-layer-dev` (created from `universal-adapters-v1-y0zkeh` at 741813a; the
files of this cycle are new). Frozen V7f (tag `universal-acmot-v7-freeze` → 488df9a) is not modified:
`tests/test_sci_v7.py` checks the ten files of `research/V7_POLICY_LOCK.json` and that `acmot_v7.py`
is byte-identical to the tag.

## 0. Pre-registration (written and committed before any SCI arm was scored)

### Question
Does the scene layer of the original AC-MOT (the hand-designed SCI) still add accuracy or save
compute once frozen V7f handles detector-score interpretation, and does any effect come from
scene-dependent allocation rather than from the compute budget?

### Architecture under test
frame → scene layer (SCI, `acmot_sci.py`) → compute level LOW / MEDIUM / HIGH → detector adapter
(`adapters/detectors/compute_profile.py`, profile `configs/sci_v7_profiles.json`) → frozen detector
(cached outputs) → frozen V7f (`acmot_v7.py`, system V7f) → ByteTrack adapter → tracks → history
for t+1. Orchestrator: `acmot_sci_v7.py`.

- The scene layer controls the compute level only. It reads no detector score and sets no
  confidence, NMS, V7f or tracker threshold (unit test).
- The SCI rule is the resolution rule of the historical controller `core.Controller`
  (policy adaptive, stable, detector feedback; `run_universal_acmot.build_config`): weights
  0.30 / 0.30 / 0.20 / 0.10 / 0.05, crowd/30, tiny < 1024 px², edges/0.14, gray < 80,
  Laplacian variance < 180, stride 10, window 7, HIGH if SCI > 0.60 or tiny > 0.50, MEDIUM if
  SCI > 0.35 or scene crowded/tiny, hysteresis 0.50 / 0.40 / 0.25, dwell 30 frames, recovery probe.
  A unit test asserts the same level sequence as `core.Controller` on 9,000 random frames.
  These constants were chosen for the historical controller on VisDrone; they are copied, not
  re-selected, and they are not category-A constants in the sense of HARD_CONSTRAINTS C5.
- One change, required by the separation rule: the crowd and tiny cues count the objects the
  tracker reported at frame t−1 instead of the detections above a raw score of 0.18 (a
  detector-specific score constant).
- Level → setting: 640 / 736 / 832 px for YOLOv8n and RT-DETR-L (the historical levels; the
  three resolutions cached for both detectors). 736 px is the operating point of the V7f record.

### Data, hosts, evaluation
VisDrone2019-MOT-val, 7 sequences (row P6 of PROTECTED_EVALUATIONS: development split, not clean),
cached detections of release `v7-dev-assets-1`, annotations from the official zip
(sha256 e5357199…5705). ByteTrack with its library defaults (0.25 / 0.25 / 0.1, match 0.8), the host
of the V7f record. Metrics: internal protocol (`tools/seqstats.py`, the V7 record's primary
protocol) and official-compatible protocol (`tools/v6/eval_official.py`). Paired sequence bootstrap
of `tools/v7/bootstrap.py` (10,000 resamples, seed 42), per detector and pooled over the 14
(detector, sequence) cells. `tools/sci_v7/dev.py` refuses every split except val-7.

### Arms (same detector, tracker, data, evaluator)
| Arm | System | Meaning |
|---|---|---|
| A | NATIVE+MEDIUM | host alone at 736 px (= NATIVE of the V7 record) |
| B | NATIVE+SCI | SCI-chosen level, host alone |
| C | V7f+MEDIUM | frozen V7f at 736 px (= V7f of the V7 record) |
| D | V7f+SCI | SCI-chosen level + frozen V7f |
| static curve | {NATIVE, V7f}+{LOW, HIGH} | fixed 640 and 832 px |
| budget-matched | {NATIVE, V7f}+PERM{1,2,3} | the arm's own SCI level schedule per sequence, cut into 30-frame segments, segment order shuffled (seeds 1–3): identical level counts and compute, scene dependence removed |

Compute = mean of r² / 736² over frames (pixel count of a square input relative to 736 px).

### Decision rule (pooled cells; per-detector results reported alongside)
1. Accuracy gain over V7f (CASE 1): ΔHOTA(D − C) 95% CI > 0 and ΔIDF1(D − C) CI lower bound > −0.5,
   and D beats every PERM seed of its own layer with ΔHOTA CI > 0.
2. Efficiency (CASE 2): D compute ≤ 0.90 of C, ΔHOTA(D − C) and ΔIDF1(D − C) CI lower bounds ≥ −0.5
   (non-inferiority margin 0.5 point, declared here), and no new catastrophic sequence.
3. Scene attribution (CASE 3): if D − PERMk ΔHOTA CIs contain 0, the effect of D is attributed to its
   compute budget, not to scene-dependent allocation; a CASE 2 result then reads "a lower static
   budget is enough", not "the scene layer saves compute".
4. Degradation (CASE 4): ΔHOTA(D − C) CI < 0 at no compute saving.
5. Catastrophic sequence: MOTA < 0 (project definition). A sequence catastrophic in D but not in C
   is a new catastrophic failure and blocks adoption.
6. Host protection under switching: V7f's gain at the SCI schedule (D − B) is compared with its
   gain at fixed 736 (C − A); V7f regime shares and intervention rate are reported for C and D.

SCI + V7f becomes the next candidate only under CASE 1 or CASE 2 without a new catastrophic
sequence, and only if D is not below every PERM seed. Otherwise V7f at a fixed level stays the
candidate and the SCI result is reported as attribution evidence.

Follow-up policy: no threshold or weight of the SCI rule is changed after this commit. At most one
structural follow-up may be run, declared in this ledger with its hypothesis before it is scored.

## 1. Runs

### E-SCI-1 — pre-registered historical SCI + V7f (candidate C1 = 5a8502f)
Development run in the container (4 vCPU, cached detections; no timing), results in
`research/final/sci_v7f/dev_local_5a8502f/`. Authoritative cloud run of the same commit:
`.github/workflows/sci_v7_val.yml` → `research/final/sci_v7f/C1_5a8502f/`.
Reproduction check: `V7f+MEDIUM` and `NATIVE+MEDIUM` reproduce the V7 record (`tools/v7/dev.py`, 736 px)
exactly (identical track files; GitHub run 36995662755 gives the same NATIVE / V7f table).

Internal protocol, pooled 14 cells, Δ = B − A with 95% CI (10,000 resamples):

| Comparison | ΔHOTA | ΔIDF1 | compute B / A |
|---|---|---|---|
| D − C (V7f+SCI − V7f+MEDIUM) | −0.17 [−0.56, +0.21] | −0.19 [−0.88, +0.60] | 0.979 / 1.000 (YOLOv8n), 1.012 / 1.000 (RT-DETR-L) |
| B − A (NATIVE+SCI − NATIVE+MEDIUM) | −0.22 [−0.45, −0.01] | −0.27 [−0.71, +0.11] | 0.949, 1.026 |
| D − PERM1 / PERM2 / PERM3 | −0.03 [−0.31, +0.24] / +0.04 [−0.25, +0.33] / +0.05 [−0.32, +0.48] | −0.33 / −0.06 / +0.08 (all CIs contain 0) | equal |
| D − B (V7f gain under SCI switching) | +3.03 [+1.73, +4.62] | +4.62 [+2.50, +7.24] | equal |
| C − A (V7f gain at fixed 736) | +2.98 [+1.76, +4.47] | +4.54 [+2.54, +6.85] | equal |

SCI level use: YOLOv8n LOW/MEDIUM/HIGH 0.17/0.75/0.08, RT-DETR-L 0.08/0.80/0.12 (with V7f); the first
30 frames of every sequence are LOW by construction (start level LOW, 30-frame dwell).
Static curve with V7f (HOTA, internal): YOLOv8n 32.0 / 33.9 / 35.3 and RT-DETR-L 40.2 / 41.1 / 41.3 at
640 / 736 / 832 px (compute 0.756 / 1.000 / 1.278). Catastrophic sequences (MOTA < 0): C 0, D 0.

Outcome under the pre-registered rule: neither CASE 1 (no accuracy gain) nor CASE 2 (no compute saving:
the rule sits at MEDIUM most of the time). D equals its budget-matched scene-blind controls (CASE 3),
and SCI without V7f is slightly below fixed 736 px (B − A HOTA CI just below 0). V7f's host protection
is unchanged under resolution switching (D − B ≈ C − A). Decision: historical SCI + V7f is NOT adopted;
V7f at a fixed level stays the candidate. Archived unchanged.

### D-SCI-1 — headroom diagnostic (GT oracle; run after E-SCI-1, before any new SCI design)
`tools/sci_v7/oracle.py`: per-frame MOTA numerator (TP − FP − IDS) of the static runs at 640 / 736 / 832,
summed over 30-frame segments; levels assigned by that GT benefit. Not a controller (uses labels,
in-sample), only an upper bound on what any segment-level allocation could gain here.

| Comparison (pooled) | ΔHOTA | ΔIDF1 |
|---|---|---|
| V7f+ORACLEM − V7f+MEDIUM (same compute as fixed 736: 0.999 / 0.998) | +0.47 [+0.005, +0.97] | +1.04 [+0.22, +1.97] |
| V7f+ORACLE − V7f+SCI (same level counts as SCI) | −0.01 [−0.36, +0.38] | +0.03 [−0.58, +0.68] |
| NATIVE+ORACLEM − NATIVE+MEDIUM | +0.14 [−0.73, +0.82] | +0.62 [−0.34, +1.57] |

Reading: with these three levels, even a GT-informed allocation at matched compute gains about half a
HOTA point over the static operating point; at the SCI's own level counts it gains nothing. The headroom
for scene-dependent allocation between 640 and 832 px is small relative to the resolution of a 7-sequence
test, so a failure of a causal cue rule here says little about cues and much about headroom.
