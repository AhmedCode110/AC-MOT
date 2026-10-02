Dear Editor-in-Chief,

Please consider the enclosed manuscript, "Calibration Versus Scene Switching of Detector Operating Points for Aerial Multi-Object Tracking," for publication as a full-length paper in the Journal of Aerospace Information Systems.

Onboard perception for unmanned aerial vehicles must track many small objects through scenes that change within a single flight, and it must do so within a fixed compute budget. The manuscript asks how much of the benefit of adapting the detector operating point comes from switching it with the scene and how much from calibrating it. Its instrument is a causal, training-free controller that reads five inexpensive scene cues and sets the detector's confidence threshold, suppression threshold, and input resolution every ten frames. All parameters were selected on validation data and frozen before a locked held-out test on VisDrone2019 test-dev and a zero-tuning test on UAVDT, with paired bootstrap intervals throughout.

Besides the accuracy gains over a static default and over a hand-designed controller, the manuscript reports an attribution analysis: a matched static operating point reaches the same held-out accuracy as the optimized adaptive controller, so the gain comes from calibrating the detector operating point and the tracker profile on aerial validation data rather than from frame-to-frame switching; on validation, most of it comes from the detector operating point. The ablations identify when scene-driven switching does add accuracy. A supplementary cross-pipeline check rescored archived outputs of two further detector-tracker pipelines: a separately recalibrated controller on U2MOT showed no measurable held-out HOTA change against that tracker's published static operating point, and an edge-based switching rule on SparseTrack gained slightly in-sample over its declared comparator but not over the tracker's default setting. The pipelines share this outcome; they do not share one transferred controller. We believe this result is useful to designers of airborne tracking systems, because it separates the value of calibration from the value of adaptation and gives a procedure for testing both.

Related manuscript. The authors are also submitting a separate manuscript, "Training-Free Self-Calibrating Control Layer for Heterogeneous Multi-Object Trackers," to Signal, Image and Video Processing. The two manuscripts share the research program and some public benchmarks, but they have different methods, research questions, and evidence:
- This manuscript concerns a scene-driven controller of the detector operating point (scene complexity index mapped to confidence, suppression, and resolution) with one primary detector and tracker, selected on VisDrone validation data and evaluated on VisDrone test-dev and UAVDT, plus a supplementary cross-pipeline check (U2MOT on VisDrone test-dev, SparseTrack on MOT17 validation).
- The other manuscript concerns a different, later method: a detector- and tracker-independent intervention layer that self-calibrates online from the score distribution of each host and does not use scene cues, evaluated on multiple published trackers and detectors, mainly on ground-level benchmarks, with a separate development and post-freeze protocol.
No result, table, or figure appears in both manuscripts. SparseTrack on the MOT17 validation half is used in both, as a host for different methods and in separate runs; the static baseline HOTA here (69.17, archived Colab run rescored with TrackEval) and there (68.88, separate reproduction) come from different executions and are not reused across the manuscripts. A copy of the other manuscript can be provided to the editor on request.

The manuscript has not been published and is not under consideration elsewhere. The evaluation code, frozen configurations, and result records are available from the corresponding author.

Sincerely,

Ahmed Gouda Ismail (corresponding author), Mohamed S. Mohamed
Computer Engineering and Artificial Intelligence Department, Military Technical College, Cairo, Egypt
Tarek Ahmed Mahmoud
Computer Engineering Department, Egypt University of Informatics, Cairo, Egypt
a7medgouda1@gmail.com
