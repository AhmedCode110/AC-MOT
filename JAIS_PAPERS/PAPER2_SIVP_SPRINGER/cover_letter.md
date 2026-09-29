Dear Editor-in-Chief,

Please consider the enclosed manuscript, "Training-Free Self-Calibrating Control Layer for Heterogeneous Multi-Object Trackers," for publication as an original research article in Signal, Image and Video Processing.

Multi-object tracking pipelines are assembled from detectors and trackers developed separately. When a detector is replaced, retrained, or quantized, its confidence scale changes and the thresholds of the tracker no longer fit; in our experiments, halving the detector scores drives two published trackers from about 67 to 0 HOTA. The manuscript presents a control layer, placed between a frozen detector and a frozen tracker, that self-calibrates online from the detector's score stream with nested Otsu thresholds, decides when to intervene, and adapts the tracker's input and thresholds through a declared host contract, without training or labelled calibration. One frozen configuration is evaluated on nine published trackers, with development evidence, predeclared external systems, and post-freeze external systems reported separately, and with paired bootstrap intervals throughout. The manuscript reports where the layer helps (single-stage trackers without a low-score stage, shifted or noisy detector scores), where it is transparent (well-matched two-stage trackers), and where it fails.

Related manuscript. The authors are also submitting a separate manuscript, "Scene-Adaptive Detector Operating-Point Control for Unmanned Aerial Vehicle Multi-Object Tracking," to the Journal of Aerospace Information Systems. The two manuscripts belong to the same research program but have different methods, data, and results: the other manuscript studies a controller that sets one detector's operating point from image cues on aerial benchmarks, whereas this manuscript studies a layer that reads the detector's score stream, acts on the tracker, and is evaluated across many published trackers and detectors on MOT17, KITTI, and DanceTrack. No result, table, or figure appears in both manuscripts. A copy can be provided on request.

The manuscript has not been published and is not under consideration elsewhere. All authors have approved the submission.

Sincerely,

Ahmed Gouda Ismail (corresponding author), Mohamed S. Mohamed
Computer Engineering and Artificial Intelligence Department, Military Technical College, Cairo, Egypt
Tarek Ahmed Mahmoud
Computer Engineering Department, Egypt University of Informatics, Cairo, Egypt
a7medgouda1@gmail.com
