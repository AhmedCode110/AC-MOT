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

Authoritative cloud run (GitHub Actions run 36998503435, AMD EPYC 7763 × 4, no GPU; candidate worktree
5a8502f, code sha256 5db5deb5…8181; policy lock 10/10; 21 tests passed, 1 skipped (needs local outputs)):
every internal-protocol number above is reproduced exactly (`research/final/sci_v7f/C1_5a8502f/`).
Official-compatible protocol, pooled cells (`bootstrap_official.txt`): D − C HOTA −0.26 [−0.59, +0.04],
IDF1 −0.40 [−0.93, +0.16], MOTA −0.77 [−1.51, −0.20]; D − PERM1/2/3 HOTA −0.07 / −0.19 / −0.18 (all CIs
contain 0); C − A HOTA +2.29 [+1.19, +3.59]; V7f+HIGH − V7f+MEDIUM HOTA +0.96 [+0.60, +1.42] at 1.278×
compute; V7f+LOW − V7f+MEDIUM HOTA −1.27 [−1.80, −0.82] at 0.756× compute.

Outcome under the pre-registered rule: neither CASE 1 (no accuracy gain) nor CASE 2 (no compute saving:
the rule sits at MEDIUM most of the time). D equals its budget-matched scene-blind controls (CASE 3),
and SCI without V7f is slightly below fixed 736 px (B − A HOTA CI just below 0). V7f's host protection
is unchanged under resolution switching (D − B ≈ C − A). Under the official protocol the combination is
slightly worse than fixed 736 px on MOTA (CASE 4 on that metric). Decision: historical SCI + V7f is NOT adopted;
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

## 2. General-SCI cycle (started after E-SCI-1 was archived)

Rule for this cycle: no cue, weight, threshold, smoothing constant or compute profile is kept because it
existed historically; each retained choice needs a documented reason (prior evidence, a declared
engineering constraint, an ablation, a stability/transfer test or a measured accuracy–compute curve).
Development data: val-7 only (row P6). No protected split is opened.

### D-SCI-2 — cue audit at 640 / 736 / 832 px (`tools/sci_v7/cue_audit.py`)
Target per 30-frame segment: benefit of more compute b = q(832) − q(640) (MOTA numerator of the static
runs). 13 causal candidate cues: image (edges, darkness, blur, global motion, motion unreliability),
canonical detections in the score layer's own bands (primary density per megapixel, relative size,
ambiguous share), tracks (density, relative size, churn) and a label-free probe (new primary candidates
revealed by one extra pass at the next level). Within-sequence Spearman over 14 (detector, sequence)
cells, V7f layer (`dev_local_5a8502f/cue_audit_V7f_3levels.json`):

| cue | mean ρ | cells +/− | YOLOv8n | RT-DETR-L |
|---|---|---|---|---|
| img_edges | −0.02 | 7/7 | +0.22 | −0.26 |
| img_motion_unrel | +0.21 | 10/3 | +0.16 | +0.25 |
| det_density | +0.15 | 10/4 | +0.21 | +0.09 |
| det_small | −0.10 | 6/8 | −0.06 | −0.15 |
| trk_density | +0.07 | 7/7 | +0.24 | −0.10 |
| probe_up | −0.01 | 5/4 | −0.07 | +0.12 |
| others | |ρ| ≤ 0.07 | split | | |

No cue reaches |ρ| = 0.3, and the sign of most cues changes between detectors. The NATIVE layer gives the
same picture (`cue_audit_NATIVE_3levels.json`). This repeats the E24–E26 audit under V7f, with
score-free and probe cues added.

### D-SCI-3 — is the compute benefit a property of the scene?
Spearman of the segment benefit b between two runs that see the same frames:

| pair | within-sequence mean ρ (cells +/−) | pooled ρ |
|---|---|---|
| same detector, V7f vs NATIVE layer (YOLOv8n) | +0.56 (6/1) | +0.64 |
| same detector, V7f vs NATIVE layer (RT-DETR-L) | +0.54 (7/0) | +0.55 |
| same layer, YOLOv8n vs RT-DETR-L (V7f) | +0.10 (5/2) | +0.01 |
| same layer, YOLOv8n vs RT-DETR-L (NATIVE) | −0.15 (3/4) | −0.06 |
| 640→736 benefit vs 736→832 benefit (four cells) | −0.17 … −0.01 | −0.21 … +0.07 |

The benefit of more compute is reproducible for one detector (it survives a change of score layer) but
does not transfer between detectors: where extra resolution pays off is a property of the
(scene, detector) pair, not of the scene. A detector-independent scene index therefore has no stable
target to predict at these levels; this, together with the small oracle headroom (D-SCI-1), explains
E-SCI-1 structurally.

### Pre-registration P-GSCI-1 (written before the wide-range cue audit was run)
Cue selection rule for a General SCI, applied to the wide span (target b = q(960) − q(512), cues from the
736 px reference run, sweep cache): a cue is retained only if its within-sequence Spearman ρ with b has the
same sign in both detectors, |mean ρ| ≥ 0.10 in each detector, and that sign in at least 10 of the 14
(detector, sequence) cells. Retained cues enter with equal weight after a causal per-stream ECDF rank
(no constants); if no cue is retained, the General SCI is a constant compute request (a static operating
point) and the scene-awareness clause is recorded as unsupported.

### D-SCI-4 — resolution sweep (512–960 px), one device
Sweep cache built on GitHub-hosted CPU runners (run 36999160296; release `sci-v7f-sweep-1`, tar sha256
dfa22765…905b). At 640 / 736 / 832 px it reproduces the V7-record cache (built on Mac/MPS) box for box:
100% of boxes with score ≥ 0.25 have an IoU ≥ 0.9 same-class partner, mean |score difference| ≈ 1e-6.

Static curves, V7f layer, internal protocol HOTA (compute = r²/736²):

| px | 512 | 576 | 640 | 704 | 736 | 768 | 832 | 896 | 960 |
|---|---|---|---|---|---|---|---|---|---|
| compute | 0.48 | 0.61 | 0.76 | 0.92 | 1.00 | 1.09 | 1.28 | 1.48 | 1.70 |
| YOLOv8n | 27.9 | 30.0 | 32.0 | 33.5 | 33.9 | 34.0 | 35.3 | 36.2 | 38.0 |
| RT-DETR-L | 39.0 | 39.9 | 40.2 | 40.7 | 41.1 | 41.4 | 41.3 | 41.6 | 41.9 |
| YOLOv8n NATIVE | 26.1 | 27.9 | 30.7 | 31.5 | 31.7 | 32.9 | 33.4 | 34.3 | 35.7 |
| RT-DETR-L NATIVE | 34.9 | 35.0 | 35.9 | 36.3 | 36.8 | 37.2 | 37.1 | 37.8 | 38.0 |

V7f is above the host alone at every compute level for both detectors (YOLOv8n +1.8 to +2.3, RT-DETR-L
+3.9 to +4.4 HOTA) and has no catastrophic sequence at any level for RT-DETR-L. The value of compute is
detector-specific: +10.1 HOTA from 512 to 960 px for YOLOv8n, +2.9 for RT-DETR-L. Measured mean detector
latency on one CPU host for YOLOv8n (`sweep/sweep_summary.txt`): 35.8 ms at 512 px to 100.9 ms at 960 px;
the RT-DETR-L jobs ran on two CPU models, so its latency curve is not used for decisions.

GT-oracle frontier over all nine resolutions (`tools/sci_v7/oracle.py budget`, Lagrangian allocation of
one resolution per 30-frame segment; in-sample, optimistic):

| budget (mean compute) | oracle − static at about the same compute (pooled HOTA) | IDF1 | MOTA |
|---|---|---|---|
| 0.58 vs 576 px (0.61) | +0.23 [−0.29, +1.08] | +1.00 [+0.15, +2.20] | +2.18 [+0.77, +3.81] |
| 0.96 vs 736 px (1.00) | +0.60 [−0.12, +1.56] | +1.64 [+0.69, +2.84] | +4.59 [+3.35, +6.28] |
| 1.23 vs 896 px (1.48) | +0.35 [−0.32, +1.08] | +1.00 [+0.14, +1.95] | +3.53 [+1.21, +5.41] |

The oracle optimises the MOTA numerator on the same frames, hence its MOTA gain; on HOTA, even a
GT-informed allocation is within about half a point of the static curve at every budget.

### P-GSCI-1 outcome
Wide-span cue audit (`general_sci/cue_audit_V7f_512_960.json`, `..._NATIVE_...`): no cue satisfies the
rule (same sign in both detectors, |mean ρ| ≥ 0.10 each, ≥ 10/14 cells). Closest: `probe_down`
(YOLOv8n −0.08, RT-DETR-L −0.16, 3+/7−) and `img_motion` (−0.26 vs +0.04). Selected cue set: none.
By the pre-registered consequence the General SCI is a constant compute request, and the
scene-awareness clause is recorded as **not supported** on this data.

Stream-level observation (not a controller, not tested as one): the score layer's own primary-band
count per frame rises with resolution for YOLOv8n (6.3 → 12.5 objects per frame from 512 to 960 px) and is
flat for RT-DETR-L (≈ 13), matching the shapes of their accuracy curves; per sequence it does not predict
the gain (Spearman 0.04 and −0.18). A label-free compute calibration per detector is therefore a
hypothesis for future work, not a result.

## 3. Candidate G1 (frozen development candidate)
`configs/general_acmot_g1.json`: constant compute request MEDIUM → adapter profile (736 px for every
detector; unseen detectors get the shared profile without calibration) → frozen V7f → host contract.
Behaviourally identical to V7f at 736 px; the compute slot stays in the architecture as an interface
whose scene policy was not supported by evidence. Lock: `research/GENERAL_ACMOT_G1_LOCK.json`.

Freeze: tag `general-acmot-g1-freeze` → 751c602 (published by workflow run 37009594004 after verifying
`research/GENERAL_ACMOT_G1_LOCK.json` and `research/V7_POLICY_LOCK.json` at that commit).

## 4. G1 transfer and host tests (frozen G1; nothing changed after the freeze)

### T-G1-OCSORT — OC-SORT host (official args), val-7, container (`sci_v7f/G1_transfer_local/`)
| | internal ΔHOTA | internal ΔIDF1 | internal ΔMOTA | official ΔHOTA | official ΔMOTA |
|---|---|---|---|---|---|
| YOLOv8n | +14.86 [+12.32, +19.50] | +20.18 | +10.23 | +12.09 [+10.94, +14.23] | +4.16 [−1.47, +8.50] |
| RT-DETR-L | +6.76 [+2.88, +11.82] | +9.75 | +2.46 [−1.87, +7.11] | +3.75 [+0.38, +8.42] | −4.11 [−15.39, +3.65] |
| pooled | +9.73 [+6.37, +13.76] | +14.28 [+9.31, +19.62] | +6.34 [+2.63, +9.62] | +6.97 [+3.68, +10.45] | +0.03 [−6.04, +4.79] |

OC-SORT's single 0.6 threshold leaves it at 10% (YOLOv8n) and 28% (RT-DETR-L) recall on this data; G1
lowers its effective operating point from the stream's own score bands, which raises recall and HOTA on
14/14 cells. Failure: under the official protocol (class-aware, ignored regions dropped) the host alone has
no sequence with MOTA < 0, G1 has 4 (YOLOv8n uav0000182 −6.4; RT-DETR-L uav0000182 −9.6, uav0000268 −2.1,
uav0000305 −32.7). Under the internal protocol neither has one. With ByteTrack (C1, official protocol)
V7f reduces the catastrophic sequences from 7 to 4 and makes the remaining ones less negative
(e.g. RT-DETR-L uav0000305 −77.4 → −50.2).
