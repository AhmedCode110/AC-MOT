# Global parameter selection protocol (declared BEFORE running, 2026-09-27)

## Question
Which shared global setting of the Universal AC-MOT candidate controller
gives the most robust tracking across detector families, without any
detector-specific parameter?

## Fixed before running
* Development data: VisDrone2019-MOT-val, 7 sequences. Test-dev untouched.
* Detectors seen jointly by the optimizer: YOLOv8n, RT-DETR-L.
* Tracker: ByteTrack (legacy AC-MOT settings buffer 45, match 0.86).
* Evaluator: reference protocol (HOTA TrackEval; rest motmetrics).
* Search space (28 configs, full grid — low-dimensional and cheap, so a
  full grid gives the complete response surface; no adaptive search needed):
  * architecture ∈ {gate-only (V1 base), gate + density Top-K (V2cA base)}
  * leader-relative gate ρ ∈ {0 (off), 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}
  * density budget multiplier κ ∈ {0.5, 1, 2} (scales 12/36/12/48)
* Constraint (catastrophe guard): overall MOTA > 0 for EVERY detector.
* Primary objective J2: maximize the worst detector's ½(HOTA + IDF1).
  Rationale: the claim is universality, so no detector may be sacrificed;
  HOTA and IDF1 are the tracking-centric metrics; MOTA enters as a guard
  because it is dominated by raw FP/FN counts.
* Comparison objectives (reported, not used for the final choice):
  J1 = mean over detectors of ½(HOTA + IDF1);
  J3 = mean relative gain of ½(HOTA + IDF1) over V1 (ρ=0, gate-only).
* Validation: leave-one-sequence-out (7 folds). Each fold selects on 6
  sequences × 2 detectors and scores the held-out sequence. Pooled
  held-out metrics = CV estimate of the whole selection procedure.
  Final parameters = the same procedure applied to all 7 sequences.
  Stability = agreement of fold choices with the final choice.
* Frames are never split; sequences are the unit.

## Not tuned here (documented as inherited AC-MOT defaults; sensitivity
analysed separately): normalizer bins/decay/update period, generic
sensitivity formula, resolution controller, ByteTrack buffer/match,
leader EMA decay 0.9.

## Amendment 1 (2026-09-27, after the grid, BEFORE computing J4)
Observed: J2 (worst-detector absolute ½(HOTA+IDF1)) selected gate_r0.4,
whose RT-DETR MOTA is only 3.3. Structural reason: YOLOv8n is uniformly the
weaker detector in absolute terms, so min() is always attained by YOLO and
RT-DETR never enters the objective — max-min over unnormalized metrics
degenerates to single-detector optimization when detector capacities differ.
Declared now (before computing it): J4 = min over detectors of the relative
gain of ½(HOTA+IDF1) over each detector's own V1 on the same sequences
(scale-normalized max-min). Decision rule: if the two scale-normalized
objectives (J3 mean relative gain, J4 worst relative gain) select the same
config, that config is final; otherwise J4 (more conservative) is used.
J2 results are reported unchanged as the original primary objective.

## Amendment 2 (2026-09-27, after viewing the response surface)
Disclosure: the full 7-sequence response surface had been printed before
this amendment. Reason: the user's requirements (stated before this
protocol) say a solution that collapses on one detector/sequence is
unacceptable; the original constraint (overall MOTA > 0 per detector) did
not encode that. J4's choice (gate_r0.4) has RT-DETR MOTA −45/−25/−9 on
three sequences. Definition with no new tuned number: a (detector,
sequence) cell is catastrophic if MOTA < 0 (worse than an empty tracker,
whose MOTA is exactly 0). Selection becomes lexicographic: (1) minimise
the number of catastrophic cells on the selection sequences, (2) maximise
J4. Applied identically inside every LOSO fold (training sequences only).
Both the Amendment-1 result and this result are reported.
