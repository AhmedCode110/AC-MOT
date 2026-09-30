Dear Editor-in-Chief,

Please consider the enclosed manuscript, "Training-Free Host-Preserving Control of Detector-Score Operating Points for Multi-Object Tracking," for publication as a full-length paper in the Journal of Aerospace Information Systems.

Perception pipelines on aircraft and other autonomous platforms combine detectors and trackers that are developed, retrained, compressed, and qualified separately. Trackers consume detector confidence through fixed score thresholds, so a change of detector or of its score calibration can silently break a qualified tracker. The manuscript shows how fragile these operating points are (a rank-preserving halving of all scores reduced two published trackers to zero accuracy) and that naive online adaptation is not a safe remedy (an earlier self-calibrating layer of this work degraded two strong trackers by 4 to 6 points).

The manuscript presents a training-free, causal control layer that sits between a frozen detector and a frozen tracker. It estimates score bands and a regime statistic from past frames only, keeps each tracker's declared operating point when the stream appears well calibrated, and intervenes otherwise. One frozen configuration was evaluated unchanged with nine published tracker implementations, four detector sources, and three datasets, under a freeze-and-transfer protocol with predeclared and preregistered external systems, paired bootstrap intervals, and a reproduction audit. The layer preserved strong trackers, improved a predeclared single-stage tracker with intervals above zero, and recovered performance in 7 of 14 calibration-shift conditions, at about 3 ms per frame on a four-core processor. The manuscript also reports, rather than corrects, a statistically significant regression on one system evaluated after the freeze, and it identifies the cause.

We believe the contribution is relevant to aerospace information systems because it addresses a practical integration problem of onboard perception, namely keeping a qualified tracker usable when the detector or its calibration changes, with an auditable, training-free, and causal component. We state plainly that the evaluation uses ground-level public benchmarks (MOT17, KITTI, DanceTrack); no aerial-platform result of the final layer is reported.

Related manuscript. The author has also submitted "Scene-Adaptive Detector Operating-Point Control for Unmanned Aerial Vehicle Multi-Object Tracking" to the Journal of Aerospace Information Systems. The two manuscripts belong to one research program but have different methods, questions, and evidence:
- The other manuscript concerns a scene-driven controller of the detector operating point (a scene complexity index mapped to confidence, suppression, and resolution) for one detector and one tracker, evaluated on VisDrone2019 test-dev and UAVDT.
- This manuscript concerns a later, different method: a tracker- and detector-independent layer that self-calibrates from the score distribution and is evaluated on multiple published trackers and detectors with a freeze-and-transfer protocol.
No result, table, or figure appears in both manuscripts; this manuscript cites the other one only where the earlier design is mentioned. A copy can be provided on request.

The manuscript has not been published and is not under consideration elsewhere. The frozen layer, configuration and lock, per-sequence results, bootstrap records, and the scripts that regenerate every table and figure are available in the project repository.

Sincerely,

Ahmed Gouda
