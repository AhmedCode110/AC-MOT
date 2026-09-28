# Universal AC-MOT V7 — method (frozen policy V7f)

Source of truth: `configs/universal_acmot_policy_v7.json` (resolved `V7Spec`),
implementation `acmot_v7.py`, lock `research/V7_POLICY_LOCK.json`, freeze
commit 488df9a.

## Setting
A training-free, online, causal control layer between a frozen detector and a
frozen tracker. Per frame it sees only:
- the detector's candidate list (boxes, scores, class labels);
- optionally one image statistic (global motion of frame t vs t−1);
- the tracker's output tracks of frame t−1;
- the host tracker's declared operating point, `HostContract(assoc, birth,
  low, match)`: first-stage association threshold, birth threshold, lowest
  score the host can use, IoU-match tolerance (1 − min IoU). These are
  documented properties of the published tracker (a single-stage tracker has
  low = its threshold).
It contains no detector, tracker, dataset or sequence names (unit-tested).
It outputs, per frame: which candidates reach the tracker, the scores passed,
and the host's association/birth thresholds and match tolerance.

## Per frame t
1. **Stream statistics (frames < t).** Nested exact Otsu on the pooled logits
   of frames t−10..t−1: t1 (background | foreground), t2 (ambiguous |
   confident). ρ = confident share of the foreground.
   - *Interpretability check:* the bands are read as background | ambiguous |
     confident only when the class below t1 holds at least as many pooled
     candidates as the foreground; otherwise (no background mode, e.g. a high
     emission floor) the frame's ρ counts as clean evidence (ρ = 1).
2. **Regime.** clean iff the median of ρ over ALL past non-cold frames of the
   stream is ≥ ½ (the regime is a property of the detector × scene stream;
   transient confidence dips do not flip it). "Cold" = no valid bands yet
   (frame 1): the host's own operating point applies.
3. **Clean / cold regime (self-limiting).** The host's thresholds are kept and
   only lowered to t2 if the host would reject the confident class
   (min(host, t2)); raw scores; only cross-class duplicates (one box, two
   class hypotheses) are removed.
4. **Noisy regime.** V6 bands: primary ≥ t2, extension [t1, t2) (may continue
   tracks, may not start them), discard < t1; rank (ECDF) score remap;
   track-context duplicate rule: a weaker overlapping candidate (IoU > ½) is
   removed unless a different track of frame t−1 claims it (crowd-safe).
5. **Foreground track-consistent rescue.** A foreground candidate (score ≥ t1;
   every emitted candidate when the stream has no background mode) that
   continues an existing track of frame t−1 (IoU ≥ ½) which no usable
   candidate covers, but which the host cannot see at the operating point
   passed this frame, is handed to the host's lowest stage (score raised just
   above it). For a two-stage host (low < its thresholds) the band is empty by
   construction; for a single-stage host it is the missing continuation stage.
6. **Motion rule (noisy frames only).** IoU-match tolerance
   min(0.95, 1 − (1 − m0)/max(1, r)), r = frame-t motion / median of past
   motion. Inactive when no image cue is available.
7. **State update.** Frame-t data enter the state after the decision.

Causality: thresholds, bands and regime use frames < t only; the match
tolerance uses the permitted frame-t image cue; duplicate handling and rescue
use frame-t geometry plus tracks of t−1 (unit-tested).

## Relation to earlier versions
V6-TF (frozen, immutable) = always-noisy bands with IoU duplicate removal:
strong on miscalibrated/noisy streams, harmful on clean crowded streams
(SparseTrack −4.15, BoostTrack −5.89 HOTA). V7c/V7d added the ρ regime,
host-anchored clean regime and crowd-safe duplicates; V7f adds whole-stream
regime, the split interpretability check and the rescue (ledger E15–E18).
