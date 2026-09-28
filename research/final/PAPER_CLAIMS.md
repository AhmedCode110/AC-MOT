# PAPER CLAIMS — what the evidence supports

Evidence levels:
- ONE-WAY: first and only post-freeze evaluation (confirmation-16,
  BoT-SORT, Faster R-CNN, UAVDT).
- POST-HOC: test-dev, where V4 was evaluated once before.
- DEVELOPMENT: val-7, dev-40.

"Significant" means the 95% paired sequence bootstrap CI excludes 0
(10,000 resamples, seed 42).

## A. Fully supported
1. **Training-free and label-free.** The layer has no trained parameters
   and uses no labels or GT at runtime. Every constant is structural (A),
   online-derived (B), a safety bound (C) or an API default (D); see
   PARAMETER_STATUS.md. No constant was selected by a quality metric; the
   memory lengths are shown insensitive.
2. **Causal and online.** Every control decision for frame t uses frames
   < t plus frame-t image cues. This is asserted by a future-perturbation
   test, and live == replay parity passes (80/80 frames) with real detector
   inference.
3. **One shared policy with no detector-specific logic.** A test asserts no
   detector, dataset or tracker names in the layer. The same frozen file ran
   YOLOv8n, RT-DETR-L and Faster R-CNN with ByteTrack and BoT-SORT.
4. **Exact invariance to Platt/temperature recalibration** of detector
   scores (test, plus identical val-7 outputs under ×2 / ×0.5 logit scaling).
5. **Observable adaptivity.** The self-calibrated primary threshold differs
   per detector and per sequence under the same policy (FIGURES/fig_thresholds).
6. **Fewer catastrophic failures than the VisDrone-tuned V4 on the VisDrone
   evaluations:**
   - confirmation-16: 1 vs 3 (ByteTrack) and 1 vs 2 (BoT-SORT);
   - development-40: 2 vs 3;
   - Faster R-CNN val-7: 0 vs 0.

   The largest avoided collapse is RT-DETR uav0000266_04830, where V4 gets
   MOTA −252 and V6-TF +3.6.
7. **Transfer to an unseen detector without retuning (Faster R-CNN, val-7).**
   - vs V4: official-compatible HOTA +2.52 [0.25, 5.02] and IDF1 +4.08
     [1.20, 7.09].
   - vs one shared raw threshold: MOTA +16.9 [12.3, 21.6].
   - A fixed raw threshold chosen for other detectors fails on a
     differently calibrated detector; the self-calibrating layer does not.
8. **A large, consistent gain over a single shared raw threshold for
   YOLOv8n:** HOTA +3.5 to +4.4 and IDF1 +6.3 to +8.1 on confirmation-16,
   test-dev and UAVDT, all significant.
9. **Negative results reported:**
   - E41 rejected (15 catastrophic cells);
   - scene-adaptive resolution F5 did not beat random allocation;
   - learned controllers did not generalise;
   - jitter band rejected.

## B. Partially supported (state with the qualifier)
1. **"Matches the VisDrone-tuned V4 without tuning."** Supported for
   YOLOv8n: no significant difference on confirmation-16, test-dev or
   BoT-SORT. On UAVDT YOLOv8n is slightly worse (HOTA −0.48, IDF1 −0.99,
   significant).
2. **"Improves over V4 with RT-DETR-L."** Significant under the
   official-compatible protocol:
   - confirmation-16: HOTA +2.78, MOTA +4.62;
   - BoT-SORT: HOTA +3.09, MOTA +5.56.

   It is NOT significant under the internal protocol (positive point
   estimates). On test-dev (post-hoc) MOTA is +3.98 with CI [0.01, 8.69].
3. **Tracker-agnostic.** Shown for two trackers of the same family
   (ByteTrack and BoT-SORT, both two-stage IoU association from Ultralytics).
   Trackers with a different association design have not been tested yet;
   the external published-system transfer is pending.
4. **Dataset transfer (UAVDT).** V6-TF ≈ V4 (RT-DETR n.s.; YOLOv8n small
   deficit). UAVDT is dominated by detector domain shift (12 catastrophic
   cells for both); shared static has fewer (6) because it outputs almost
   nothing.
5. **Real-time.** The measured overhead is 3.6–3.9 ms/frame mean (P95
   ≈ 8 ms) on a Mac CPU. Official T4 timing is deferred, so no FPS claim on
   the target GPU yet.
6. **Scene-state adaptation.** Only motion-conditioned association remains,
   and its effect is small (IDS −12% on dev-40, HOTA +0.3). The adaptive
   core is self-calibrated candidate control, not scene-conditioned
   resolution.

## C. NOT supported (must not be claimed)
1. State of the art on VisDrone or UAVDT. The detectors are COCO-pretrained
   and the protocols are internal or official-compatible ports, not the
   leaderboard.
2. "Works with every detector/tracker." Tested: YOLOv8n, RT-DETR-L, Faster
   R-CNN R50-FPN v2; ByteTrack, BoT-SORT.
3. Uniform superiority over a fixed threshold. With RT-DETR-L, shared static
   0.5 is as good or better on HOTA/IDF1 (test-dev −2.4 HOTA, significant).
4. Invariance to arbitrary monotone score recalibration. Only affine-in-logit
   (Platt) invariance holds; s³ and 0.5·s degrade it.
5. Independence from the detector adapter's emission contract. The layer
   needs the low-score candidate stream (floor 0.01).
6. That pre-detector scene-adaptive resolution helps (F5 failed).
7. Any published-system transfer result. It is not run yet
   (EXTERNAL_PAPER_TRANSFER.md).
8. Hardware-independent numbers. The T4 fidelity gate is deferred; all
   quality numbers come from Mac-MPS caches.
