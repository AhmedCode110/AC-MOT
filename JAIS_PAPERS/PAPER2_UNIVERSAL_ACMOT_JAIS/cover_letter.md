Dear Editor-in-Chief,

Please consider the enclosed manuscript, "Training-Free Self-Calibrating Control Layer for Heterogeneous Multi-Object Trackers," for publication as a full-length paper in the Journal of Aerospace Information Systems.

Tracking pipelines in airborne and space-borne perception systems are assembled from detectors and trackers developed separately. When a detector is replaced, retrained, or quantized for flight hardware, its confidence scale changes, and the thresholds of the tracker no longer fit. The manuscript presents a control layer, placed between a frozen detector and a frozen tracker, that self-calibrates online from the detector's score stream, decides when to intervene, and adapts the tracker's input and thresholds through a declared host contract, without training or labelled calibration. One frozen configuration is evaluated on nine published trackers, with development evidence, predeclared external systems, and post-freeze external systems reported separately under a registered protocol with paired bootstrap statistics. The manuscript reports where the layer helps (single-stage trackers, shifted or noisy detector scores), where it is transparent (well-matched two-stage trackers), and where it fails.

Related manuscript. The authors are also submitting a separate manuscript, "Calibration Versus Scene Switching of Detector Operating Points for Aerial Multi-Object Tracking," to the Journal of Aerospace Information Systems. The two manuscripts belong to the same research program but have different methods, research questions, and evidence:
- The other manuscript studies a scene-driven controller that sets one detector's operating point (confidence, suppression, and resolution) from image cues, with one tracker, on aerial benchmarks.
- This manuscript studies a later, different method that reads the detector's score stream rather than the image, acts on the tracker's input and thresholds, and is evaluated across many published trackers and detectors, mainly on ground-level benchmarks.
No result, table, or figure appears in both manuscripts. This manuscript mentions the other as a companion study in its discussion of scene-driven adaptation. A copy can be provided to the editor on request.

The manuscript has not been published and is not under consideration elsewhere. The code, frozen configuration, lock file, and result records are available from the corresponding author.

Sincerely,

Ahmed Gouda Ismail (corresponding author), Mohamed S. Mohamed
Computer Engineering and Artificial Intelligence Department, Military Technical College, Cairo, Egypt
Tarek Ahmed Mahmoud
Computer Engineering Department, Egypt University of Informatics, Cairo, Egypt
a7medgouda1@gmail.com
