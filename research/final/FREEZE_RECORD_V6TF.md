# V6-TF FREEZE RECORD

- **Tag:** `universal-acmot-v6-freeze`. The frozen commit is the tagged
  commit (`git rev-list -n1 universal-acmot-v6-freeze`), recorded in
  research/context/FROZEN_VERSIONS.md by the follow-up commit.
- **Branch:** `universal-adapters-v1`; remote github.com/AhmedCode110/AC-MOT.
- **Policy:** `configs/universal_acmot_policy_v6tf.json` (development
  candidate X5; D-027). The runner alias `V6TF` in `tools/v6/dev.py` reads
  this file. A clean rerun is byte-identical to X5 on all 14 val-7 cells.
- **File hashes:** `research/V6TF_POLICY_LOCK.json` covers the policy file,
  pipeline, calibration primitives, adapters, live wrapper, evaluators and
  tests.
- **Detector weights (sha256):**
  - `yolov8n.pt`: f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
  - `rtdetr-l.pt`: 6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f
  - Faster R-CNN ResNet50-FPN v2 (transfer only):
    weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth (torchvision COCO),
    sha256 dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf
- **Detector interface:** emission floor 0.01 and max 1000 detections;
  detector-native suppression (YOLO NMS 0.7, RT-DETR none, Faster R-CNN
  0.5); COCO classes {person, car, bus, truck}; resolution 736.
- **Tracker:** Ultralytics BYTETracker (ultralytics 8.3.200), native
  buffer 30, base match 0.8, fuse_score on. The layer sets the per-frame
  association and birth thresholds to 0.5, low to 0.1, and the match
  tolerance through the motion rule. BoT-SORT (transfer): the Ultralytics
  BOTSORT adapter with the same generic controls.
- **Environment:**
  - Python 3.12.14, macOS 27.0 arm64 (Mac/MPS caches).
  - torch 2.14.0, ultralytics 8.3.200, numpy 2.2.6, scipy 1.18.1,
    opencv 4.11.0.
  - motmetrics 1.4.0, TrackEval 12c8791.
- **Randomness:** the layer is deterministic. Bootstrap uses 10,000 paired
  sequence resamples with seed 42. The split seed is 20260927
  (`research/TRAIN_SPLIT_V5.json`, annotation fingerprint 045620e4…).
- **Sequence lists:**
  - val-7 (development sandbox): uav0000086_00000_v, uav0000117_02622_v,
    uav0000137_00458_v, uav0000182_00000_v, uav0000268_05773_v,
    uav0000305_00000_v, uav0000339_00001_v.
  - development-40 (robustness check): `TRAIN_SPLIT_V5.json["development"]`.
  - confirmation-16 (PROTECTED until this tag, evaluated once after it):
    uav0000013_01073_v, uav0000072_05448_v, uav0000072_06432_v,
    uav0000099_02109_v, uav0000124_00944_v, uav0000140_01590_v,
    uav0000145_00000_v, uav0000150_02310_v, uav0000244_01440_v,
    uav0000248_00001_v, uav0000266_03598_v, uav0000266_04830_v,
    uav0000273_00001_v, uav0000281_00460_v, uav0000308_00000_v,
    uav0000323_01173_v.
- **Pre-freeze verification:**
  - tests/test_v6_adaptive_layer.py: 9 pass; legacy tests: 3 pass.
  - Live == replay parity: PASS, 80/80 frames.
  - Causality perturbation test: PASS.
  - Exact Platt invariance: PASS.
  - Config consistency (V6TF ≡ X5): PASS, 14/14 cells.
- **Protected data untouched before the tag:** no confirmation-16,
  test-dev, UAVDT, Faster R-CNN or BoT-SORT quality metric was computed for
  V6-TF. `tools/v6/dev.py` refuses them until the tag exists.
- **Not part of the freeze:**
  - Unknown-provenance WIP is kept in git stash 51a46971 (D-028).
  - The Amendment-5f T4 fidelity gate and official timing are deferred by
    the owner (Amendment 9 §5).
