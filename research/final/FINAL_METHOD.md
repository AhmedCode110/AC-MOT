# V6-TF — the final Universal AC-MOT adaptive layer

Status: FROZEN at tag `universal-acmot-v6-freeze` (see
`FREEZE_RECORD_V6TF.md`). Config: `configs/universal_acmot_policy_v6tf.json`.
Code: `universal_policy_pipeline.py` (`UniversalPolicyPipeline.process`,
`_otsu_bands`), `online_calibration.py` (`dedup`, `exact_otsu2`,
`nested_otsu`, `RobustHistory`), `adapters/detectors/online_normalizer.py`
(ECDF), `universal_acmot.py` (live wrapper).

## 1. What the layer is
A training-free, online, causal control layer between a frozen detector and
a frozen tracker. The layer is one shared policy for every detector and
tracker, with no detector, tracker, dataset or sequence names or constants
in its code (asserted by a test). The detector adapter only translates the
interface: it exposes every candidate box with score ≥ 0.01 (up to 1000) and
the detector's native suppression. The tracker adapter exposes three
generic controls:
- the association (first-stage) threshold;
- the birth threshold;
- the IoU-match tolerance.

The layer decides, per frame, which candidates reach the tracker and in
which role (primary: may associate first and start tracks; extension: may
only continue an existing track in the low-score stage; discarded), and how
tolerant association is, from statistics of the stream itself.

## 2. Per-frame causal ordering (frame t)
1. **Image cue (frame t).** Global motion `m_t` = phase-correlation shift
   between the downscaled frames t−1 and t, divided by the image diagonal
   (`scene_state.image_stats`, identical live and cached).
2. **Detector** at the compute-budget resolution B (736 in all experiments,
   the same compute as V4). It returns candidates `{(b_i, s_i)}`.
3. **Duplicate suppression.** A greedy, class-agnostic pass in descending
   score keeps a candidate only if its IoU with every kept candidate is
   ≤ 0.5. This is not a control decision; it is geometric processing of the
   frame's own list. The rationale is that under the IoU-0.5 correspondence
   rule two boxes with IoU > 0.5 cannot both be matched to distinct objects,
   so the weaker one is redundant.

   *Scope correction (post-freeze, 2026-09-28; the method is unchanged):*
   this holds only when distinct objects rarely overlap above IoU 0.5, as in
   top-down aerial views. In side-view pedestrian crowds (MOT17), occluding
   people overlap heavily, and the rule removes real objects (external
   SparseTrack analysis, FAILURE_ANALYSIS F-K).
4. **Logits.** `ℓ_i = log(s_i / (1 − s_i))`, with s clipped to [1e−9, 1−1e−9].
5. **Band thresholds from frames < t only.** Let
   `H_t = ∪_{k=t−W}^{t−1} {ℓ}_k` be the pooled post-suppression logits of the
   previous W = 10 frames. Nested exact Otsu gives the two thresholds:
   - `t1 = Otsu₂(H_t)`, the background | foreground split;
   - `t2 = Otsu₂({ℓ ∈ H_t : ℓ ≥ t1})`, the extension | primary split inside
     the foreground.

   `Otsu₂(v)` is the exact two-class Otsu threshold: the midpoint of the gap
   between sorted observations that maximises the between-class variance
   `n₁n₂(μ₁ − μ₂)²`. It uses no histogram bins. If `H_t` is empty (frame 1),
   no candidate is admitted.
6. **Bands and tracker scores.** `u_i` is the order-only ECDF rank of `s_i`
   among the scores of previous frames (stride 10, window 20 samples). The
   bands map onto the tracker's own native thresholds (birth/association
   0.5, low 0.1):
   - **primary** (`ℓ_i ≥ t2`): score `0.5 + 0.5·u_i`;
   - **extension** (`t1 ≤ ℓ_i < t2`): score `0.1 + 0.4·u_i`;
   - **discard** (`ℓ_i < t1`).

   The tracker thresholds are set to association = birth = 0.5 and low = 0.1.
7. **Motion-conditioned association.** `r_t = m_t / median(m_{t−100..t−1})`
   (RobustHistory, warm-up 5; r = 1 before warm-up). The IoU-match tolerance
   is `match_t = min(0.95, 1 − (1 − m0)/max(1, r_t))`, where m0 is the
   tracker's native value (0.8 for ByteTrack). Larger relative camera
   motion loosens the match gate; calm frames keep the native gate.
8. **Tracker update** (frozen ByteTrack/BoT-SORT) returns the tracks.
9. **State update, used from frame t+1 on:** the frame-t logits enter `H`,
   the scores enter the ECDF, and `m_t` enters the motion history.

Causality is asserted by `tests/test_v6_adaptive_layer.py`: perturbing
frames ≥ k leaves every output of frames < k and the frame-k thresholds
unchanged.

## 3. Adaptive state (all online, reset per video)
- Pooled logits of the last 10 frames: they give the band thresholds.
- ECDF sample of past scores: gives the order within a band.
- Motion history of 100 samples: gives the association tolerance.

Different detectors observing the same scene produce different
statistics, and therefore different thresholds, under the same policy.
Measured per-detector threshold distributions are in
`FIGURES/fig_thresholds.*`.

## 4. Why each mechanism exists (evidence: EXPERIMENT_LEDGER.md)
| Mechanism | Failure it addresses | Evidence |
|---|---|---|
| Duplicate suppression at IoU 0.5 | Native suppression (YOLO NMS 0.7, RT-DETR none) fed duplicates into the primary band: 19% duplicate output boxes in E41 | X1: MOTA +11/+25 on val-7; without it 4 (val-7) / 6 (dev-40) catastrophic cells |
| Thresholds from frames < t | E41 used the frame's own detections to threshold that frame (causality) | X3 ≡ X1 in quality; causal by test |
| Nested Otsu (background → foreground → primary) | 3-class Otsu is dominated by the volume of background emissions (≈200/frame for RT-DETR), which pulls the primary boundary into clutter; primaries reach 2–3× GT in failing scenes | X5b/X5: 0 catastrophic cells on val-7; precision 71/74 vs 62/59 (X3) |
| Extension band = foreground's lower class | Without an extension-only role, tracks fragment (HOTA/IDF1 drop ≈ 4 / 5–7 points) | X5 vs X5b (val-7 and dev-40) |
| Order-only ECDF within band | Keeps the within-band score order independent of calibration | Platt invariance exact (test) |
| Motion-conditioned association | Camera motion breaks IoU association | small, consistent: IDS −6%/−2% (val-7), −12%/−13% (dev-40); HOTA +0.3–0.5 |

## 5. Invariances and dependencies (measured)
- **Exact invariance** to Platt/temperature recalibration `ℓ → aℓ + b` with
  a > 0. Nested Otsu is affine-equivariant, and the ECDF and duplicate
  suppression are order-only.
- **Not invariant** to non-affine monotone recalibration (s³, 0.5·s). RT-DETR
  degrades to 2–4 catastrophic cells on val-7, and V4 degrades too.
- **Depends on the adapter's emission contract:** the adapter must expose
  the low-score candidate stream. Raising the floor from 0.01 to 0.05 or 0.1
  lowers YOLO recall (HOTA −4.9 / −8.9) without catastrophic cells.
- **Insensitive** to the Otsu memory (5/10/20 frames), the ECDF memory and
  the motion history window and warm-up.

## 6. What it does not do (tested and rejected)
- Pre-detector resolution adaptation from scene state (F5, R-res). It did
  not beat random allocation (FX-17), so resolution is a compute-budget
  input.
- Learned controllers (S3). Research upper bound only, never deployable
  (FX-12/13).
- Current-frame 3-class Otsu (E41, FX-18); jitter-width extension band
  (X4, FX-19).

## 7. Cost
Arithmetic per frame:
- two sorts of ≤ 10·N logits, where N ≈ 100–260 candidates per frame;
- one O(N²) worst-case greedy suppression;
- O(1) motion update.

Mac CPU development measurement: 3.6–3.9 ms mean and 7.6–8.4 ms P95 per
frame excluding the tracker. Official T4 timing is deferred (Amendment 9 §5).
