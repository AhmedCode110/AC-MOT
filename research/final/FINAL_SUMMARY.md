# FINAL SUMMARY — Universal AC-MOT V6-TF

## What the final Adaptive Layer does
V6-TF is a training-free, online, causal control layer between a frozen
detector and a frozen tracker. One shared policy serves all detectors and
trackers. Per frame it:
1. removes duplicate candidates (IoU > 0.5, the evaluation's
   correspondence rule);
2. splits candidates into primary / extension / discard with a nested exact
   Otsu (background | foreground, then extension | primary) on the pooled
   detector logits of the previous 10 frames;
3. orders scores within each band by an online ECDF;
4. loosens the tracker's IoU-match tolerance when global camera motion
   exceeds its own recent median.

It has no learned parameters, no labels and no dataset-selected constants,
and it is exactly invariant to Platt/temperature recalibration. Details:
FINAL_METHOD.md.

## Why this design
- E41, the V5-TF lock, was rejected by audit. It inflated candidates (the
  3-class Otsu boundary sat in clutter), let duplicates start tracks (19% of
  output boxes) and let ghosts persist through a broad extension band.
- Six hypotheses were tested on the val-7 sandbox (ledger X1–X5j):
  - duplicate suppression (accepted);
  - causal thresholds (accepted);
  - nested split (accepted);
  - foreground extension band (accepted);
  - jitter-width band (rejected);
  - primary-only band (rejected: loses association quality).
- The candidate survived an untouched robustness check on development-40:
  2 catastrophic cells vs V4 3 and E41 15.

## Evidence at a glance
| Setting | Status | V6-TF vs V4 | V6-TF vs shared static | Catastrophic cells (V6 / V4 / static) |
|---|---|---|---|---|
| confirmation-16, ByteTrack | ONE-WAY | YOLO ≈ (n.s.); RT official HOTA +2.78*, MOTA +4.62* | YOLO HOTA +3.7*, IDF1 +7.7*; RT n.s. | 1 / 3 / 3 |
| confirmation-16, BoT-SORT | ONE-WAY | YOLO ≈; RT official HOTA +3.09*, MOTA +5.56* | — (vs BoT-SORT default: RT MOTA +22.6*) | 1 / 2 / 3 |
| Faster R-CNN (unseen), val-7 | ONE-WAY | official HOTA +2.52*, IDF1 +4.08* | MOTA +16.9* | 0 / 0 / 3 |
| UAVDT | ONE-WAY | YOLO HOTA −0.48*, IDF1 −0.99*; RT n.s. | YOLO HOTA +4.4*, IDF1 +8.1* | 12 / 12 / 6 |
| test-dev | POST-HOC | YOLO n.s.; RT MOTA +3.98* | YOLO HOTA +3.9*; RT HOTA −2.4* | 0 / 0 / 0 |

(* = 95% paired bootstrap CI excludes 0.) Faster R-CNN on test-dev and UAVDT:
see FINAL_RESULTS.md, where they are appended when the reference caches
finish.

## Distinguishing the kinds of success
- **Functional success: YES.** The frozen layer runs live and in replay with
  identical results. It is deterministic, causal by test, and 12 tests pass.
  The same file serves 3 detectors × 2 trackers × 2 datasets.
- **Experimental success: MODERATE, not uniform.**
  - Robustness improves consistently: fewest catastrophic failures on
    VisDrone, and no collapse where V4 or fixed thresholds collapse.
  - V4's accuracy (a system tuned on VisDrone) is matched without any
    tuning.
  - Clear gains appear where score calibration differs from the tuning
    target (RT-DETR under the official protocol, Faster R-CNN).
  - A calibrated detector with a good fixed threshold (RT-DETR at 0.5)
    remains a strong competitor, and UAVDT domain shift is not solved.
- **Publication-quality evidence: PARTIAL.** Missing before submission:
  - the T4 fidelity gate and official timing (deferred by the owner);
  - the external published-system transfer (needs download permission;
    no aerial candidate with public weights runs without CUDA);
  - Faster R-CNN on test-dev/UAVDT (running).

  The present evidence supports a robustness / calibration-invariance
  claim, not a state-of-the-art claim (PAPER_CLAIMS.md).

## Pointers
FINAL_METHOD · EXPERIMENT_LEDGER · FINAL_RESULTS · FAILURE_ANALYSIS · ABLATION ·
REPRODUCIBILITY · PAPER_CLAIMS · EXTERNAL_PAPER_TRANSFER · FREEZE_RECORD_V6TF ·
TABLES/ · FIGURES/.
