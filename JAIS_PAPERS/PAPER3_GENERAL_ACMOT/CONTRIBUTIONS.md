# Contributions (as stated in the manuscript)

1. **Attribution of a scene-adaptive controller's success.** A matched static operating point reproduces the calibrated AC-MOT controller on held-out VisDrone test-dev. HOTA differs by −0.09 [−0.25, +0.06] (post hoc). On validation, almost the whole gain over the static default comes from the tracker profile (+1.03) and the calibrated operating point (+5.21); scene switching adds +0.03. Operating-point calibration is therefore the transferable mechanism. The scene complexity index was the procedure that found that operating point.

2. **Why direct transfer fails, and a separation of translation from decision.**
   - Stage-1 scene-index thresholds do not carry over to another pipeline.
   - Raw detector scores do not share a scale: one raw threshold gives 0 to 6 catastrophic sequences across four detectors.
   - Halving the scores drives two trackers to zero HOTA.
   - Control is therefore split into detector and tracker adapters (translation, implementation-specific) and one decision policy that contains no detector, tracker, dataset or sequence names.

3. **V7f, a causal, training-free online self-calibration layer.**
   - Nested exact Otsu thresholds on the logits of the previous ten frames give primary / extension / reject bands.
   - A whole-stream regime estimate with an interpretability check decides whether to intervene.
   - Rank remapping is applied in noisy frames only.
   - Duplicate handling is crowd-safe.
   - A track-consistent continuation stage is included.
   - No per-detector thresholds exist in the policy, and nothing is fitted or labelled at deployment.

4. **Host-aware selective intervention through a declared host contract** (association, birth, lowest-stage threshold, matching tolerance). This lets one policy be attached to single-stage, two-stage, coordinate-only and two-view trackers without tracker-specific branches. In the clean regime, host thresholds are never raised.

5. **Frozen-transfer evaluation of one policy, with evidence kept apart by status:**
   - development detectors and hosts;
   - two predeclared external trackers (PD-SORT +0.613 [+0.268, +1.646] HOTA; Hybrid-SORT identical);
   - three post-freeze external trackers (mixed; one significant loss);
   - one unseen detector under a transfer lock (RetinaNet +4.01 [+2.25, +5.46] with ByteTrack, on development sequences);
   - score-calibration shift (7 of 14 conditions significant recoveries, none significant degradations).

   Nothing was retuned on any transfer target.

6. **Boundary analysis.**
   - Sparse streams near the regime boundary cause the one significant external loss (C-TWiX, KITTIMOTS cars).
   - An under-confident detector with two-stage trackers loses MOTA.
   - New catastrophic sequences appear with OC-SORT under the official-compatible protocol.
   - On the tested aerial data, dynamic allocation of detector compute has too little headroom to justify a scene controller: the GT oracle gains at most about 0.6 HOTA, and the supplementary pre-registered oracle gates were not met.

7. **Reproducibility record.**
   - Frozen version tags and file-hash locks (10/10 and 14/14 files verified at the evidence snapshot).
   - A transfer lock written before evaluation.
   - Fixed bootstrap seeds.
   - Scripts that regenerate every table, figure and number from the committed result files and checksum-verified release assets.
