Dear Editor-in-Chief,

Please consider the enclosed manuscript, "Scene-Adaptive Detector Operating-Point Control for Unmanned Aerial Vehicle Multi-Object Tracking," for publication as a full-length paper in the Journal of Aerospace Information Systems.

Onboard perception for unmanned aerial vehicles must track many small objects through scenes that change within a single flight, and it must do so within a fixed compute budget. The manuscript studies a causal, training-free controller that reads five inexpensive scene cues and sets the detector's confidence threshold, suppression threshold, and input resolution every ten frames. All parameters were selected on validation data and frozen before a locked held-out test on VisDrone2019 test-dev and a zero-tuning test on UAVDT, with paired bootstrap intervals throughout.

Besides the accuracy gains over a static default and over a hand-designed controller, the manuscript reports an attribution analysis: a matched static operating point reaches the same held-out accuracy as the optimized adaptive controller, so the gain comes from calibrating the operating point on aerial validation data rather than from frame-to-frame switching. The ablations identify when scene-driven switching does add accuracy. We believe this result is useful to designers of airborne tracking systems, because it separates the value of calibration from the value of adaptation and gives a procedure for testing both.

Related manuscript. The authors are also submitting a separate manuscript, "Training-Free Self-Calibrating Control Layer for Heterogeneous Multi-Object Trackers," to the Journal of Aerospace Information Systems. The two manuscripts share the research program and some public benchmarks, but they have different methods, research questions, and evidence:
- This manuscript concerns a scene-driven controller of the detector operating point (scene complexity index mapped to confidence, suppression, and resolution) for one detector and one tracker, selected on VisDrone validation data and evaluated on VisDrone test-dev and UAVDT.
- The other manuscript concerns a different, later method: a detector- and tracker-independent intervention layer that self-calibrates online from the score distribution of each host and does not use scene cues, evaluated on multiple published trackers and detectors, mainly on ground-level benchmarks, with a separate development and post-freeze protocol.
No result, table, or figure appears in both manuscripts. The other manuscript mentions this one as a companion study, without results, where it discusses scene-driven adaptation. A copy can be provided to the editor on request.

The manuscript has not been published and is not under consideration elsewhere. The evaluation code, frozen configurations, and result records are available from the corresponding author.

Sincerely,

Ahmed Gouda Ismail (corresponding author), Mohamed S. Mohamed
Computer Engineering and Artificial Intelligence Department, Military Technical College, Cairo, Egypt
Tarek Ahmed Mahmoud
Computer Engineering Department, Egypt University of Informatics, Cairo, Egypt
a7medgouda1@gmail.com
