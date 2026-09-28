# EXTERNAL PUBLISHED-SYSTEM TRANSFER — frozen V6-TF on SparseTrack (headline) and BoostTrack (supporting)

**Bottom line:** the frozen V6-TF layer does **not** improve either published
system; it degrades both significantly. It does cut false positives on
every sequence, and it reduces ID switches on BoostTrack. These are negative
transfer results, kept as such. V6-TF was not retuned.

## 1. Selected published paper
**Headline: SparseTrack.** Zelin Liu, Xinggang Wang, Cheng Wang, Wenyu Liu,
Xiang Bai, "SparseTrack: Multi-Object Tracking by Performing Scene
Decomposition based on Pseudo-Depth".
**Supporting: BoostTrack.** V. D. Stanojević, B. T. Todorović, "BoostTrack:
boosting the similarity measure and detection confidence for improved
multiple object tracking".

## 2. Full citations
- Z. Liu, X. Wang, C. Wang, W. Liu, X. Bai. SparseTrack: Multi-Object
  Tracking by Performing Scene Decomposition based on Pseudo-Depth. *IEEE
  Transactions on Circuits and Systems for Video Technology*, 2025
  (published May 2025; arXiv:2306.05238).
- V. D. Stanojević, B. T. Todorović. BoostTrack: boosting the similarity
  measure and detection confidence for improved multiple object tracking.
  *Machine Vision and Applications* 35(3), 2024, doi:10.1007/s00138-024-01531-5.
  Correction: doi:10.1007/s00138-024-01605-4.

## 3. Venue / ranking
Secondary-source rankings (SCImago/Clarivate are bot-protected;
EXTERNAL_PAPER_SELECTION.md):
- IEEE TCSVT: Q1.
- Machine Vision and Applications: SJR Q2, 0.530 (2024).

## 4. Official repositories and weights
- SparseTrack: github.com/hustvl/SparseTrack @ 499844f32c5bb2332f9811f26cd70cf4e517d4e7
  (the owner's pinned commit).
- BoostTrack: github.com/vukasin-stanojevic/BoostTrack @ fb5bfc3.
- Detector: ByteTrack YOLOX-X MOT17 ablation checkpoint
  `bytetrack_ablation.pth.tar`, sha256 26cb8d28…48b1a. Used by both papers;
  TOPICTrack's `topictrack_ablation.pth.tar` has the same hash.

## 5. Why selected
- **SparseTrack:** Q1 2025 journal; official code and checkpoint; detector
  and tracker separable; a two-stage (high/low) ByteTrack-lineage
  association in which candidate control is meaningful; the owner already
  had a pinned deterministic runtime; the owner prioritised it.
- **BoostTrack:** a published tracker whose own contribution is
  detection-confidence control (DLO/DUO boosting), the function closest to
  AC-MOT. This makes it the most informative complementarity test.

Rejected candidates and reasons: EXTERNAL_PAPER_SELECTION.md.

## 6. Original architectures and contributions
- **SparseTrack**
  - Pipeline: YOLOX-X (conf 0.01, NMS 0.7, 800×1440) → SparseTracker.
  - Pseudo-depth scene decomposition: the Depth Cascading Matching (DCM)
    splits dense crowds into depth levels and associates level by level.
  - Two score stages: high above `track_thresh` 0.6; low 0.1–0.6 with IoU
    0.3.
  - GMC: OpenCV videostab similarity estimate.
  - Contribution: occlusion-aware association ordering.
- **BoostTrack**
  - Pipeline: YOLOX-X (conf 0.1, NMS 0.7) → one-stage association with an
    IoU + Mahalanobis + shape similarity weighted by detection-tracklet
    confidence.
  - DLO/DUO confidence boosting of detections near or far from tracklets,
    then filtering at `det_thresh` 0.6; ECC camera-motion compensation.
  - Contribution: similarity and detection-confidence boosting.
- **AC-MOT V6-TF (the added function):** training-free, causal,
  self-calibrated candidate control from the stream's own history.
  - duplicate suppression;
  - nested exact-Otsu primary / extension / discard bands on the logits of
    frames t−10..t−1;
  - ECDF order within band;
  - motion-conditioned association tolerance.

  It is complementary *in principle*: SparseTrack decides the association
  order and BoostTrack re-scores detections near tracks, while V6-TF decides
  which candidates are admitted and in which role.

## 7. Published results (MOT17 val-half)
- **SparseTrack** (paper Table VI): HOTA 69.2, AssA 72.3, IDF1 81.4,
  MOTA 76.8. IDS/FP/FN are not reported.
- **BoostTrack** (authors' re-reported Table 3 in the official repo, issue
  #8, corrected shape similarity), online, SB+DCB, CMC, no LI/GBI/ReID:
  HOTA 68.371, MOTA 75.561, IDF1 81.354, IDSW 118. With GBI: 71.326 /
  80.549 / 83.839 / 106.

## 8. Reproduction procedure (EXTERNAL_PAPER_REPRODUCTION.md)
- Owner's Drive MOT17; the val-half split was verified frame-for-frame
  against the official definition and the authors' shipped GT.
- Official code, checkpoint and configs.
- Compatibility-only patches are in `tools/v6/external/vendor/*.diff`:
  device, NumPy-2 aliases, dtype strings, dataset paths.
- SparseTrack's GMC C++ body is compiled verbatim against OpenCV 5.0 through
  a C ABI.
- The authors' ablation instruction was followed (per-sequence overrides
  disabled).
- Hardware: Apple A18 Pro (8 GB), PyTorch 2.14 MPS, fp16 detector.

## 9. Reproduced baselines

| Metric | SparseTrack paper | Our execution (TrackEval) | Our execution (SparseTrack's motmetrics) | Owner's earlier T4 run (motmetrics) |
|---|---:|---:|---:|---:|
| HOTA | 69.2 | 68.88 | – | – |
| MOTA | 76.8 | 77.85 | 76.65 | 76.87 |
| IDF1 | 81.4 | 81.97 | 81.50 | 81.67 |
| IDS | – | 124 | 119 | 118 |
| FP | – | 2231 | 2856 | 2794 |
| FN | – | 9582 | 9607 | 9555 |
| AssA | 72.3 | 71.79 | – | – |

| Metric | BoostTrack authors (online) | Our execution | Δ |
|---|---:|---:|---:|
| HOTA | 68.371 | 68.492 | +0.12 |
| MOTA | 75.561 | 75.502 | −0.06 |
| IDF1 | 81.354 | 81.413 | +0.06 |
| IDSW | 118 | 113 | −5 |

Both baselines are **credible**. In addition, the replay drivers used for
the +V6 arms reproduce each official output **byte-for-byte on all 7
sequences**, so the +V6 arm differs from the baseline only by the frozen
layer.

## 10. AC-MOT integration architecture
```
published YOLOX-X detections (logged from the official run; identical in both arms)
  → PublishedDetections adapter (format only)
  → frozen V6-TF: universal_acmot.UniversalACMOT with configs/universal_acmot_policy_v6tf.json
  → tracker adapter (generic controls only)
  → OFFICIAL SparseTracker / BoostTrack
  → official output filter / post-processing
```

Generic control mapping (integration only; no constants of their own):

| Host | association threshold | birth threshold | low stage | IoU-match tolerance |
|---|---|---|---|---|
| SparseTrack | `track_thresh` | `det_thresh` | native 0.1 | `match_thresh` (native 0.85) |
| BoostTrack | `det_thresh` | `det_thresh` (same parameter) | none | 1 − `iou_threshold` (native 0.3) |

Code:
- `tools/v6/external/sparsetrack_v6.py`
- `tools/v6/external/boosttrack_v6.py`
- `tools/v6/external/mot17_eval.py`

Pre-declared rule (declared before any +V6 result): the +V6 arm receives
exactly the published detection stream. For BoostTrack that stream starts
at 0.1, below the layer's documented 0.01 emission contract. This is a
reported interaction, not adjusted.

## 11. Proof that V6-TF remained frozen
- Every +V6 run first calls `tools/v6/dev.py::verify_lock()`. It checks the
  sha256 of all files in `research/V6TF_POLICY_LOCK.json` (policy file,
  pipeline, calibration primitives, adapters, live wrapper, evaluators,
  tests) against the freeze. Any mismatch aborts the run.
- The layer is the repository's `UniversalACMOT` live wrapper, loaded from
  the frozen config. It is the path whose live == replay parity passed 80/80
  frames.
- No parameter was changed for either host. Diagnostic variants
  (`--diag`) are labelled DIAGNOSTIC, written to separate outputs, and never
  reported as the method.

## 12. Before / after (TrackEval, MOTChallenge protocol, MOT17 val-half)
| System | MOTA ↑ | HOTA ↑ | IDF1 ↑ | IDS ↓ | FP ↓ | FN ↓ | Prec | Rec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SparseTrack — paper | 76.8 | 69.2 | 81.4 | – | – | – | – | – |
| SparseTrack — our execution | 77.85 | 68.88 | 81.97 | 124 | 2231 | 9582 | 95.21 | 82.22 |
| **SparseTrack + frozen V6-TF** | **71.71** | **64.72** | **77.49** | **108** | **601** | **14537** | 98.50 | 73.02 |
| **Δ AC-MOT** | **−6.14** | **−4.15** | **−4.49** | −16 | −1630 | +4955 | +3.29 | −9.20 |
| BoostTrack — authors (online) | 75.56 | 68.37 | 81.35 | 118 | – | – | – | – |
| BoostTrack — our execution (online) | 75.50 | 68.49 | 81.41 | 113 | 1637 | 11452 | 96.29 | 78.75 |
| **BoostTrack + frozen V6-TF (online)** | **66.64** | **62.61** | **75.19** | **75** | **286** | **17619** | 99.22 | 67.31 |
| **Δ AC-MOT** | −8.87 | −5.89 | −6.22 | −38 | −1351 | +6167 | +2.93 | −11.44 |
| BoostTrack — our execution + GBI | 81.03 | 71.72 | 84.16 | 97 | 2674 | 7451 | 94.56 | 86.17 |
| BoostTrack + frozen V6-TF + GBI | 72.30 | 66.35 | 78.51 | 70 | 676 | 14183 | 98.33 | 73.68 |

SparseTrack's own motmetrics protocol shows the same direction: MOTA
76.65 → 71.30, IDF1 81.50 → 77.30.

## 13. Per-sequence results (SparseTrack; TrackEval)
| Seq | HOTA | MOTA | IDF1 | IDS | FP | FN |
|---|---|---|---|---|---|---|
| 02 | 48.4 → 47.2 | 55.0 → 51.3 | 59.7 → 58.3 | 60 → 48 | 677 → 200 | 3712 → 4562 |
| 04 (crowded) | 79.1 → 73.4 | 89.4 → 81.6 | 91.2 → 85.5 | 21 → 26 | 780 → 102 | 1754 → 4310 |
| 05 | 61.7 → 56.8 | 76.7 → 73.3 | 76.4 → 68.5 | 15 → 7 | 111 → 84 | 655 → 806 |
| 09 | 63.8 → 56.2 | 81.4 → 66.8 | 76.8 → 66.9 | 10 → 9 | 6 → 3 | 519 → 944 |
| 10 | 57.8 → 55.8 | 71.8 → 65.1 | 77.6 → 75.3 | 13 → 9 | 148 → 94 | 1511 → 1965 |
| 11 | 69.7 → 69.8 | 70.8 → **72.9** | 82.3 → 82.4 | 3 → 4 | 445 → 75 | 870 → 1145 |
| 13 | 69.5 → 63.7 | 80.1 → 73.0 | 88.7 → 82.7 | 2 → 5 | 64 → 43 | 561 → 805 |

- FP falls on all 7 sequences and FN rises on all 7.
- MOTA improves on 1/7 (MOT17-11, the most FP-heavy relative to its
  misses). HOTA is flat there and falls on the other 6.
- Figure: `FIGURES/fig_external_per_sequence.*`.

## 14. Statistical comparison
Paired sequence bootstrap: 10,000 resamples, seed 42, 95% percentile CI.

| Host | ΔMOTA | ΔHOTA | ΔIDF1 | ΔIDS | ΔFP | ΔFN |
|---|---|---|---|---|---|---|
| SparseTrack | −6.14 [−8.10, −2.74] | −4.15 [−5.54, −1.66] | −4.49 [−6.19, −1.96] | −16 [−47, +12] | −1630 [−2977, −490] | +4955 [+1973, +9493] |
| BoostTrack (online) | −8.87 [−12.37, −7.14] | −5.89 [−7.76, −2.87] | −6.22 [−9.33, −2.85] | −38 [−81, −1] | −1351 [−2749, −357] | +6167 [+3090, +10367] |
| BoostTrack + GBI | −8.73 [−12.59, −6.50] | −5.37 [−7.30, −2.75] | −5.65 [−8.96, −2.77] | −27 [−73, +11] | −1998 [−3546, −668] | +6732 [+3238, +10997] |

- HOTA, MOTA and IDF1 degradations are significant on both hosts.
- The FP reduction is significant on both.
- The IDS reduction is significant only for BoostTrack online.

## 15. Failure analysis (why the transfer fails)
Decomposition over SparseTrack's 49,709 true detections (matched to GT at
IoU 0.5, frames with history):

| Fate under V6-TF | True detections |
|---|---:|
| Stay primary | 36,032 |
| Demoted from SparseTrack's first stage (raw > 0.6) to the extension band | 5,787 |
| Removed by IoU-0.5 duplicate suppression | 3,527 |
| Fall below the background split | 1,403 |

Of the 20,077 boxes that duplicate suppression removes, 82% are non-GT;
this is the source of the FP gain.

Self-calibrated thresholds on SparseTrack's stream (median raw score):
- t1 (background | foreground) ≈ 0.33;
- t2 (extension | primary) ≈ 0.78, versus the published 0.6.

On BoostTrack's 0.1-floored stream, t2 ≈ 0.87. About 41% of the detections
BoostTrack normally admits are demoted to extension, and BoostTrack has no
low-score stage to use them.

Diagnostic ablations on SparseTrack (not the method):

| Variant | HOTA | MOTA | IDF1 |
|---|---:|---:|---:|
| without duplicate suppression | 67.31 | 75.27 | 80.23 |
| without motion rule | 64.86 | 71.73 | 77.78 |

About 62% of the HOTA loss comes from duplicate suppression; the motion rule
is neutral.

Mechanism:
1. **Duplicate suppression.** V6-TF's IoU-0.5 rule was justified for
   top-down aerial scenes, where distinct objects rarely overlap above 0.5.
   In MOT17's side-view pedestrian crowds, occluding people overlap heavily,
   and the published detector deliberately keeps them (NMS 0.7). The rule
   then deletes real people. The justification "two boxes with IoU > 0.5
   cannot both match distinct objects" is **false when GT objects themselves
   overlap**. This is a scope limitation of the frozen design.
2. **Over-conservative bands on an in-domain, high-precision detector.**
   The nested Otsu targets a clutter-dominated candidate stream; that is the
   aerial COCO-detector regime V6-TF was developed in (100–260 candidates
   per frame, precision 60–80%). A MOT17-trained YOLOX-X emits about 35
   candidates per frame at about 95% precision. The foreground class is
   then mostly true objects, and the second split cuts through the
   confident true-object mode.
3. **Host architecture.** BoostTrack's single stage discards the extension
   band, so the damage is larger than on SparseTrack, whose low stage
   partly uses it.

## 16. Computational cost
See REALTIME_ANALYSIS.md.
- SparseTrack on Apple A18 Pro (MPS):
  - detector 783 ms per frame mean (P95 884 ms);
  - official end-to-end run: 1.17 FPS.
- Matched replay:
  - baseline tracker 7.20 ms per frame;
  - +V6-TF: tracker 6.86 ms (18.2 vs 34.5 candidates) plus layer 4.67 ms
    mean (P95 5.69 ms);
  - total overhead about +0.6% of end-to-end latency.
- V6-TF makes no extra detector calls and uses the same resolution and NMS.

## 17. Supported claims
- The frozen layer integrates into two independently published trackers
  through integration-only adapters, with the lock verified. Overhead is
  about 4.7 ms per frame and there is no extra detector compute.
- On these hosts it consistently reduces false positives (−73% SparseTrack,
  −83% BoostTrack; significant).
- It reduces BoostTrack's ID switches (online −38, significant).

## 18. Unsupported claims (must NOT be claimed)
- That frozen AC-MOT improves SparseTrack or BoostTrack. It degrades HOTA,
  MOTA and IDF1 significantly on both.
- That AC-MOT's gains transfer to in-domain, high-precision pedestrian
  detectors or crowded side-view scenes.
- Any SOTA or universal-superiority statement.

## 19. Exact commits / configs
- AC-MOT frozen: tag `universal-acmot-v6-freeze` → 2cff95f;
  `configs/universal_acmot_policy_v6tf.json`; lock `research/V6TF_POLICY_LOCK.json`.
- Integration and evaluation code: this repository, `tools/v6/external/`
  (commit in FINAL_SUMMARY / git log).
- SparseTrack @499844f + `vendor/sparsetrack_499844f_compat.diff`.
- BoostTrack @fb5bfc3 + `vendor/boosttrack_fb5bfc3_compat.diff`.
- GMC shim: `vendor/gmc_shim.cpp`, `vendor/pbcvt.py`.
- Environment: `acmot_external/venv`, Python 3.12.14 with:
  - torch 2.14.0, torchvision, detectron2 0.6 @a2f4a87;
  - numpy 2.2.6, opencv-python 4.11;
  - Homebrew OpenCV 5.0.0 (GMC only);
  - ultralytics 8.3.200, motmetrics 1.4.0, TrackEval 12c8791.

## 20. Exact reproduction commands
```bash
# SparseTrack official (logs detections + timing)
cd acmot_external/SparseTrack && KMP_DUPLICATE_LIB_OK=TRUE PYTHONHASHSEED=0 ../venv/bin/python run_official_logged.py \
  --num-gpus 1 --config-file mot17_ab_track_cfg.py train.init_checkpoint=<bytetrack_ablation.pth.tar> \
  train.device=mps train.output_dir=../runs/sparsetrack_A_official
# identical-input arms
../venv/bin/python <repo>/tools/v6/external/sparsetrack_v6.py --mode replay_baseline --cache ../runs/sparsetrack_A_official/published_detections.pkl --out ../runs/sparsetrack/MOT17-val/ST_replay_baseline/data
../venv/bin/python <repo>/tools/v6/external/sparsetrack_v6.py --mode v6 --cache ../runs/sparsetrack_A_official/published_detections.pkl --out ../runs/sparsetrack/MOT17-val/ST_plus_V6TF/data
# BoostTrack official, then the arms
cd ../BoostTrack && ../venv/bin/python run_mac.py --dataset mot17 --exp_name BoostTrack --no_reid --btpp_arg_iou_boost --btpp_arg_no_sb --btpp_arg_no_vt --s_sim_corr
../venv/bin/python <repo>/tools/v6/external/boosttrack_v6.py --mode replay_baseline --name BT_replay_baseline
../venv/bin/python <repo>/tools/v6/external/boosttrack_v6.py --mode v6 --name BT_plus_V6TF
# evaluation + statistics + tables
../venv/bin/python <repo>/tools/v6/external/mot17_eval.py ../runs/sparsetrack ST_A_official ST_plus_V6TF
../venv/bin/python <repo>/tools/v6/external/mot17_eval.py --boot ../runs/sparsetrack ST_A_official ST_plus_V6TF
<repo>/.venv/bin/python <repo>/tools/v6/external/make_external_tables.py
```

Not run: TOPICTrack (IEEE TIP 2025). The environment and weights are
prepared, and its detector checkpoint is identical (same sha256). It
consumes the same 0.1-floored stream as BoostTrack, so the same over-pruning
mechanism is expected. That is an untested expectation, not a result. The
owner prioritised SparseTrack (headline) and BoostTrack (supporting).
