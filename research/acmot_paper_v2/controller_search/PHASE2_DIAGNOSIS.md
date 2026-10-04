# Phase 2 failure diagnosis (calibration data only)

## 1. Action headroom

- **Narrow space** (1088/1280/1536 x 0.45/0.60/0.70): oracle headroom = +0.178 HOTA
  mean-per-sequence; concentrated in 1/8 sequences (`uav0000020_00406_v`).
- **Broadened space** (896/1088/1280/1536/1728, same NMS set): oracle headroom
  = +0.504 HOTA; `n_seqs_with_gap=4`, but the per-sequence gaps are
  `{289: 0.10, 076: 0.09, 307: 0.42, 020: 3.43}` (sums to the 4.03 total
  underlying the 0.504 mean). **85% of the broadened headroom still comes
  from the same single sequence.** The other three "affected" sequences
  show gaps within plausible scoring noise, not a broadly distributed
  opportunity.
- **Segment-level (within-sequence) headroom** (detection-only F1 proxy,
  30-frame windows, 9 narrow-space points, no ID-tracking confound):
  overall mean = **0.0040** (F1 units, 0-1 scale) -- i.e. essentially zero.
  Per-sequence: 6/8 sequences show `mean_headroom <= 0.008`; the same two
  outliers (`020_00406`, `316_01288`) show the only non-negligible values
  (0.017, 0.008). **Windowing does not reveal hidden exploitable headroom
  that sequence-level aggregation was masking** -- the signal is the same
  single sequence, not a temporal-granularity artifact.

**Conclusion: headroom is real but narrow -- one calibration sequence
(`uav0000020_00406_v`) is a genuine outlier that wants a different
detector operating point; the other seven are consistently well-served by
the single matched-static point at every granularity tested.**

## 2. Why: SceneLayer level distribution is degenerate on this dataset

Measured by running the frozen, unchanged `acmot_sci.SceneLayer` on every
calibration frame (image stats only, boxes-observed set to empty since
this is a pure scene-statistics probe):

| sequence | LOW | MEDIUM | HIGH |
|---|---|---|---|
| uav0000316_01288_v | 30 | 97 | 0 |
| uav0000289_06922_v | 30 | 180 | 0 |
| uav0000013_00000_v | 49 | 220 | 0 |
| uav0000076_00720_v | 180 | 181 | 0 |
| uav0000307_00000_v | 30 | 384 | 0 |
| **uav0000020_00406_v** | **490** | **11** | **0** |
| uav0000315_00000_v | 30 | 602 | 0 |
| uav0000243_00001_v | 30 | 738 | 0 |

**No sequence ever reaches HIGH.** Seven of eight sequences are
MEDIUM-dominant; the outlier sequence is LOW-dominant. The historical
`t_med=0.35`/`t_high=0.60` thresholds (copied verbatim from a different
controller/dataset context, per `acmot_sci.py`'s own docstring) do not
discriminate scenes on this VisDrone-MOT-train calibration subset -- the
3-level design is *de facto* a 2-level (mostly-constant) signal here, so
no 3-way action mapping built on top of it can express more than what a
2-point (really: 1-point, since HIGH is unreachable) policy already
covers. This -- not the action-mapping search, not the cue subset -- is
the actual bottleneck Stages 1/2/3A were operating under.

Also notable: the outlier sequence is classified **LOW** (low scene
complexity) by the frozen cues, yet it is the one sequence that benefits
from **higher** resolution in the oracle sweep. This directly contradicts
a naive "low complexity -> low resolution" prior (which the predeclared
protocol explicitly warned not to assume) -- the existing crowd/tiny/
edge/dark/blur cues do not correlate with this sequence's actual
resolution sensitivity. Worth a brief visual note: this sequence's
annotations/frame count pattern (501 frames, one of the longer
calibration sequences) suggests a mostly-uncluttered scene with a few
small/distant objects where detection at higher resolution recovers more
true positives without the FP cost seen on busier scenes -- consistent
with "tiny objects, not clutter" rather than "crowd/edge complexity,"
which the existing cue formula only partially captures (`tiny` is one of
five cues, diluted by the other four in the weighted sum).

## 3. Decision

Per the predeclared robustness gate and the evidence above:

- The broadened-oracle decision rule (`n_seqs_with_gap > 2`) mechanically
  says "re-run the staged search on the broadened space." Taken at face
  value this is followed (Stage 1-equivalent on the broadened grid is
  queued). But the diagnosis above gives a strong, specific, mechanistic
  reason to expect the SAME CV-fold-concentration failure: whichever fold
  contains `uav0000020_00406_v` will show a real gain, the other three
  folds will not, failing the `>=3/4 folds positive` gate again -- exactly
  the pattern already observed in the oracle gate, Stage 1, Stage 2, and
  Stage 3A, four times in a row with four different search dimensions.
- Rather than repeat the same blind grid search at larger scale (expected
  to cost ~4-5h of CPU for a predictable result), the targeted,
  diagnosis-driven next step is **Stage 3B (SCI threshold retuning)**,
  already predeclared and designed for exactly this situation: if
  `t_med`/`t_high` can be retuned (via Optuna TPE, calibration-only) to
  make the SceneLayer actually discriminate scenes on this dataset
  (reach all 3 levels, correlate with the outlier sequence), that is a
  legitimate, well-motivated redesign test, not a repeat of the same
  search. Proceeding to Stage 3B now, then Stage 4 (joint refinement)
  per the predeclared plan, before any freeze decision.
