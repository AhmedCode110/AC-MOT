# AC-MOT v12 Colab

Open `AC_MOT_v12_Colab.ipynb` in Google Colab. All four Python modules are embedded; no additional source upload is required. Select a T4 for new inference. The default mode evaluates the existing September 5 recording without YOLO.

Modes in cell 1:

- `evaluate_saved`: run TrackEval against OLD_RUN and TEST_DATASET. Outputs an independent timestamped evaluation. Requires saved per-frame tracks and original annotations on Drive.
- `development`: set DEV_DATASET to an existing train/validation dataset. Cache six size/NMS combinations once, replay all 18 configurations, evaluate every candidate, and optionally freeze a jointly qualifying candidate. Full multiresolution caching costs more than one detector run initially.
- `final_test`: set FROZEN_CONFIG to a generated development selection. Runs A0, reference adaptive A3 and the candidate on all 17 sequences with three fresh T4 timing repeats. Evaluates every repeat separately.

Implemented: low-score recovery; resolution dwell and hysteresis; causal detector-based feedback and bounded observation-collapse probes; adaptive track-birth thresholds; fixed-resolution and non-scene scheduling controls; separate NMS bank tests; match/buffer controls; complete per-frame recording; content-hash cache resume; test-search guard; TrackEval metrics and correct sequence combination; exploratory source-cluster bootstrap; and synchronized whole-run FPS/p95 latency/peak memory.

The reference adaptive A3 is a controlled v12 reference, not a byte-identical reproduction of the legacy run: pinned Ultralytics, input class restriction and fixed NMS differ. Default ByteTrack refers to the pinned package. Compare inside this experiment. A3_v12 is experimental and does not replace the adopted final A3. All candidate outcomes are retained. A qualified development winner is not guaranteed. No ReID improvement is implemented or claimed.

The cache stores post-NMS detections at 0.01 confidence, sizes 640/736/832, NMS 0.45/0.55, max_det=1000. New NMS values require another bank. Replay measures the tracking workflow only; its speed is not deployment FPS. Live timing includes image read, scene analysis, detector and tracker, excludes warmup and serialization. First-time weights are hashed; final weights/environment must match development.

Research limitations: the preserved five-class GT filter is evaluated class-agnostically. COCO has no exact van output. Benchmark ignored-region/class-wise preprocessing is not implemented, so TrackEval outputs are official metric implementations under a custom protocol, not official VisDrone benchmark scores. External SOTA tracker comparisons, validated benchmark preprocessing and untouched cross-dataset evidence remain necessary for a strong submission. No Q1/Q2 acceptance or numeric gain is promised.

Validation: six local tests passed, including actual pinned ByteTrack low-score ID recovery, empty frames, causal resolution stability, synthetic cache/replay/resume and rejection of test-data search. Notebook code cells and embedded modules compiled. No T4 dataset inference or real-data score was produced during this build.

Primary references: https://github.com/JonathonLuiten/TrackEval ; https://github.com/VisDrone/VisDrone2018-MOT-toolkit ; https://github.com/ultralytics/ultralytics/tree/v8.3.200 . TrackEval revision and NumPy alias compatibility are embedded in evaluate.py. Full environment freeze is saved by the notebook.
