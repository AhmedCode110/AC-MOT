# AC-MOT detector-side controller: development decision report

Calibration data only (8 sequences, `research/acmot_paper_v2/DETECTOR_SPLIT.json` ->
`calibration`). Final 7-sequence VisDrone-val was never read, inspected, or
scored during any part of this development process.

## Every formulation tested

| stage | what varied | n configs | best cv_mean ΔHOTA | robustness gate | verdict |
|---|---|---|---|---|---|
| Oracle gate (narrow) | none (diagnostic upper bound) | 9 static points | headroom +0.178 (not a policy) | n/a | headroom concentrated in 1/8 sequences |
| Stage 1 | level->resolution, NMS fixed=0.70 | 27 (exhaustive) | -0.786 (best of 27, all negative) | FAIL | null |
| Stage 2 | level->NMS, resolution fixed | 27 (exhaustive) | +0.031 | FAIL (1/4 folds) | null |
| Stage 3A | cue subset, action fixed | 31 (exhaustive) | +0.032 | FAIL (1/4 folds) | null |
| Oracle gate (broadened) | none (diagnostic, +896/1728 res) | 15 static points | headroom +0.504 (85% from 1 sequence) | n/a | same single-sequence pattern, larger magnitude |
| Segment-level headroom | none (diagnostic, 30-frame windows, detection-F1 proxy) | 9 static points x windows | overall mean +0.004 (F1 units) | n/a | no hidden within-sequence opportunity |
| Stage 3B+4 joint (amended) | thresholds + action + (fixed) cue subset | 60 (Optuna TPE) | +0.0318 | FAIL (1/4 folds) | null |

**Every adaptive formulation tested converges to the same ~0.03 HOTA
ceiling, every time traced to the same single fold/sequence
(`uav0000020_00406_v`), and every time failing the predeclared robustness
gate (>=3/4 folds must show positive delta).** Five independent search
dimensions (resolution mapping, NMS mapping, cue subset, joint
threshold+action, and the broadened operating-point space) all hit the
identical wall.

## Root cause (Phase 2 diagnosis, `PHASE2_DIAGNOSIS.md`)

The frozen `acmot_sci.SceneLayer`'s historical thresholds, when run on
this detector/dataset, assign **zero frames** to HIGH across all 8
calibration sequences, and 7/8 sequences are MEDIUM-dominant -- the
3-level signal is functionally 1-2 levels here. Retuning thresholds
(Stage 3B+4) can restore level diversity (confirmed: trials reach real
LOW/MEDIUM/HIGH splits), but this does not translate into generalizing
benefit, because the *detector itself* (a VisDrone-DET-trained, 640px-
native public checkpoint, not VisDrone-MOT-trained, not the OATrack
detector) does not show resolution/NMS-conditional quality variation that
correlates broadly with the existing scene cues (crowd/tiny/edge/dark/
blur). One sequence is a genuine outlier (benefits from higher
resolution despite being classified as low-complexity by every cue
combination tried), but one outlier out of eight cannot support a
generalizing policy under leave-some-out CV.

## Best fixed control (matched static)

`r1088_n45` (resolution 1088, NMS IoU 0.45): pooled calibration HOTA
50.24 (narrow 9-point sweep); `r896_n45` wins the broadened 15-point
sweep at a similar pooled HOTA (50.89) -- both very close, both far
simpler than any adaptive candidate tested.

## Robustness controls (Phase 5)

- **Best fixed point**: `r1088_n45` (used as the comparison baseline
  throughout; equivalent to systems 1/2 already run).
- **Random/shuffled control**: not separately run as its own experiment --
  every one of the 60+27+27+31 = 145 trials across Stages 1/2/3A/3B4
  already samples the action-mapping space broadly (TPE and exhaustive
  grids), and NONE of them robustly beat matched-static; a literal
  shuffled-level-assignment control would be expected to perform at or
  below the same ceiling, consistent with every result above.
- **Cost-matched control**: not separately constructed; given every
  *quality*-optimized candidate already fails to beat matched-static,
  a cost-matched variant (same compute, different schedule) cannot
  succeed where the quality-optimized search already failed.

## Old AC-MOT comparison (Phase 6) -- scoped out, documented not fabricated

The historical frozen AC-MOT (`v1.0.0-acmot-frozen`, `FrozenSCIController`)
is a **different architecture** (tracker-threshold-adaptive: it adjusts
ByteTrack's association/birth thresholds, not the detector's
resolution/NMS) running on a **different detector** (YOLOv8n). An
apples-to-apples comparison requires a protocol decision -- which of its
outputs would even map onto a detector-side action -- that is not
resolvable from the repository alone (`GAP_AUDIT.md` 2.1 already flags
this: "has not been used for AC-MOT" through the modern adapter
interface). Not attempted here; flagged for a future, explicitly-scoped
decision rather than a fabricated comparison.

## Decision

**No candidate from any tested formulation passes the predeclared
robustness gate.** Per the predeclared null-result handling rule
(`PREDECLARATION_STAGE3PLUS.json`): the matched-static operating point
(`r1088_n45`) is frozen as the de facto "AC-MOT" entry for systems 3 and
4. This is a **negative finding for detector-side AC-MOT adaptation on
this particular public, DET-trained detector**, not evidence against
AC-MOT in general (the repository's own `GAP_AUDIT.md` records a similar
null against a tuned static point for two other detector/tracker
combinations: YOLOv8n -0.09 [-0.25,+0.06] HOTA, U2MOT -0.06 [-0.31,+0.22]
HOTA -- this is now a *third* independent null against a matched-static
reference, strengthening rather than contradicting that existing pattern).

## Addendum: confidence-floor correction and final static reference

After the above was written, a bug was found in the confidence-exactness
verification used to justify excluding confidence floor from the search
(`CONFIDENCE_CORRECTION_NOTE.md`): the original check compared cache
rows (eval5-filtered) against *unfiltered* fresh inference, producing a
spurious "not exact" result. Corrected (class-filtered both sides):
18/18 exact. Confidence floor is exact and should have been searchable.

A post-hoc confidence-floor sweep at the matched-static (resolution,NMS)
point found that **conf=0.40 robustly beats conf=0.01**: +0.88 pooled
HOTA, 7 of 8 sequences improve, 3 of 4 CV folds positive -- the first
broadly-distributed (non-single-sequence) finding anywhere in this
investigation. This is a **static, non-adaptive** improvement: applying
a fixed, higher confidence floor uniformly, no scene-conditioning
involved.

A diagnosis-motivated adaptive follow-up was tested: level-dependent
confidence (LOW level -> floor 0.10, MEDIUM/HIGH -> floor 0.40), since
the one sequence that does *not* like floor=0.40 is also the one
uniquely classified as scene-level LOW. Result: this candidate is worse
than flat floor=0.40 on **every one of the 8 sequences**, including the
targeted outlier itself. Hypothesis refuted -- adaptive confidence
control does not beat the flat choice either.

**Updated frozen policy**: resolution=1088, NMS=0.45, confidence
floor=0.40 (static, uniform, non-adaptive). This supersedes the
(resolution,NMS)-only matched-static used during Stages 1-3B4; those
stages' conclusions (no adaptive resolution/NMS/cue/threshold policy
robustly beats the best fixed point) are unaffected by the correction --
only the identity of the best fixed point improved, strengthening rather
than weakening the overall null-adaptation conclusion.

Systems 3 and 4 in the final held-out test will therefore use the
identical cached detections as systems 1 and 2 at the FROZEN policy's
settings (resolution=1088, NMS=0.45, confidence>=0.40), same tracker
configs -- AC-MOT is a documented no-op (a constant, non-adaptive
operating point) on this detector. This will be reported plainly, not
hidden, and the four-system table will show the comparison directly.
Note systems 1/2 (already run, before this correction) used a different,
independently-predeclared default operating point (resolution=1536,
NMS=0.70, no confidence pre-filter) -- that is their own, separate,
predeclared "vanilla baseline" definition (OATRACK_DESIGN.md), not the
calibration-selected matched-static control, and does not need to be
rerun.
