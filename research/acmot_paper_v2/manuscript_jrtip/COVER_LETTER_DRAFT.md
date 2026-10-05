# Cover Letter Draft — Journal of Real-Time Image Processing

Dear Editor,

We submit the manuscript “AC--MOT: Adaptive Detector Operating-Point Control for Real-Time Aerial Multi-Object Tracking” for consideration in the *Journal of Real-Time Image Processing*.

The manuscript addresses a real-time systems problem that is easy to obscure with a single FPS number: aerial multi-object tracking couples detector resolution, observation quality, latency, and downstream association. AC--MOT exposes detector operating-point selection as an online, auditable control layer. Resolution is the dominant compute control; confidence and NMS determine which observations reach the tracker; and the controller adapts these choices to changing scene statistics without retraining detector weights.

The paper reconstructs the original five-cue AC--MOT design, separates the historical V1/V2 profiles from the modern frozen v3 protocol, and reports results independently for ByteTrack and a controlled OATrack host. The modern evidence includes paired sequence bootstrap intervals, a no-retuning UAVDT transfer, matched-static and shuffled-timing attribution controls, and an explicit second-validation-read disclosure.

The real-time measurements report the complete operating path rather than detector throughput alone. On a Tesla T4 over 200 frames, v3 mean/P95 latency is 74.3238/100.6847 ms for ByteTrack and 65.4748/85.8969 ms for OATrack. Measured controller overhead is 4.2562 ms and 4.2010 ms, respectively. The corresponding measured throughputs are 13.4546 and 15.2731 frames/s. These results show compute shaping across frames and make the deployment trade-off visible, but the manuscript does not claim 30-FPS real-time operation or hard deadline compliance. It also reports that the baseline and v3 harnesses do not instrument image-read work identically, so the comparison is not presented as a perfectly matched speedup.

The evidence supports training-free detector operating-point control and tracker-dependent transfer behaviour. The matched-static and shuffled-timing controls prevent us from claiming that the precise scene-conditioned switching schedule provides a separate, consistent causal quality gain. This bounded interpretation is central to the manuscript's technical contribution and reproducibility claims.

The accompanying Online Resource 1 contains the extended historical tables, complete bootstrap details, state occupancy, full UAVDT transfer table, original algorithm, implementation trace, and provenance audit. The authors have not submitted this manuscript elsewhere. [Please confirm this statement and any journal-specific declarations before submission.]

Thank you for considering this work.

Sincerely,

Ahmed Gouda Ismail, Mohamed S. Mohamed, and Tarek Ahmed Mahmoud  \\
Computer Engineering and Artificial Intelligence Department, Military Technical College, Cairo, Egypt;\\
Computer Engineering Department, Egypt University of Informatics, Cairo, Egypt\\
Corresponding author: a7medgouda1@gmail.com

## Author confirmations before submission

- Funding statement.
- Competing-interests statement.
- Author-contribution statement.
- Acknowledgements statement.
- Originality and exclusive-submission confirmation.
- Exact AI tool/model identifier and usage record, if the submission system requests a more specific identifier than the manuscript disclosure.
