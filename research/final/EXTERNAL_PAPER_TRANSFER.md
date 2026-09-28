# External published-system transfer (post-freeze; owner directive 2026-09-28)

Rule: freeze AC-MOT first (done: tag `universal-acmot-v6-freeze`), then
select and reproduce a published system (official code, weights, config,
split and evaluator), then attach the SAME frozen V6-TF layer through an
integration-only adapter, then evaluate once. No change to the frozen layer
for any external system.

## 1. Candidate survey (2026-09-28)
Criteria:
- peer-reviewed;
- MOT, tracking-by-detection (separable detector → tracker, so a
  detector-control layer is meaningful);
- aerial (VisDrone/UAVDT) preferred;
- official code and weights;
- reproducible evaluator;
- **executable on the available hardware** (Mac arm64, MPS/CPU, no CUDA).

| Candidate | Venue | Domain | Code | Weights | Separable? | Feasible here? | Assessment |
|---|---|---|---|---|---|---|---|
| MM-Tracker (Yao et al.) | AAAI 2025 | VisDrone, UAVDT | github.com/YaoMufeng/MMTracker | Baidu Net Disk only | yes (YOLOX + ByteTrack-style + Motion Mamba) | **no**: `mamba_ssm` is CUDA-only; Baidu download needs an account | best aerial fit scientifically; blocked by hardware and weight access |
| MOSAIC-Tracker | ISPRS J. Photogramm. Remote Sens. 2025 | VisDrone, UAVDT | github.com/aJanm/MOSAIC-Tracker (GPL-3.0) | **none**; training scripts only | yes (YOLOv8) | only after training YOLOv8 on VisDrone, i.e. not the authors' checkpoint | reproduction would not be of the published system |
| OATrack (Qi et al.) | Sensors 26(16):5222, 2026 | VisDrone val, UAVDT | not public ("on request") | no | yes (YOLO11m cache) | **no** | excluded (no code) |
| DroneMOT (Wang et al.) | ICRA 2024 (arXiv 2407.09051) | VisDrone, UAVDT | github.com/PenK1nG/DroneMOT | — | no (FairMOT JDE) | **no**: DCNv2 CUDA build, PyTorch 1.7 | excluded |
| UAVMOT (Liu et al.) | CVPR 2022 | VisDrone, UAVDT | github.com/LiuShuaiyr/UAVMOT | not listed | no (MCMOT JDE) | **no** | excluded |
| FOLT (Yao et al.) | ACM MM 2023 | VisDrone, UAVDT | no official repository found | — | yes | **no** | excluded |
| BoostTrack / BoostTrack++ (Stanojević & Todorović) | Machine Vision and Applications 2024 (doi 10.1007/s00138-024-01531-5); Filomat 39(16) 2025 | MOT17, MOT20 (pedestrian) | github.com/vukasin-stanojevic/BoostTrack | Google Drive (Deep OC-SORT YOLOX-X) | yes | likely (PyTorch; `--no_reid` option; detections cacheable) | **strongest feasible**: it boosts detection confidence from tracklets, a function adjacent to AC-MOT's candidate control |
| SparseTrack (Liu et al.) | IEEE TCSVT 2025 | MOT17, MOT20, DanceTrack | github.com/hustvl/SparseTrack | ByteTrack YOLOX weights | yes (pseudo-depth cascade association) | possible (Detectron2 + Boost/pbcvt compile) | second system with a different association design |
| OC-SORT (Cao et al.) | CVPR 2023 | MOT17, MOT20, DanceTrack | github.com/noahcao/OC_SORT | ByteTrack YOLOX weights | yes (observation-centric) | likely (pure PyTorch/numpy) | alternative second system |

No aerial candidate with public weights runs on this hardware. The feasible
external transfers are therefore pedestrian benchmarks (MOT17 val-half), and
that is a limitation for an aerial paper. The value of the experiment is
that the frozen layer meets a new detector (YOLOX-X, MOT17-trained), new
trackers and a new domain without any retuning.

## 2. Plan
- BoostTrack (primary) and SparseTrack or OC-SORT (secondary). They cover
  confidence-boosting and depth-cascade/observation-centric association.
- Integration: published detector output → adapter (format only) → frozen
  V6-TF layer (duplicate suppression, nested bands, ECDF, motion rule) →
  published tracker. The tracker adapter maps the layer's generic
  association/birth threshold (0.5) onto the tracker's own detection
  threshold argument.

## 3. Status
BLOCKED on owner permission for the downloads, per the session's safety
rules on file downloads:
- MOT17 (motchallenge.net, about 5.5 GB);
- the published weights (Google Drive, a few hundred MB);
- the repositories (GitHub).

Everything else in the paper package is complete.
Sections 4–20 (citation, reproduction vs paper, integration, before/after,
per-sequence, statistics, failure analysis, cost, claims, commits, commands)
are filled when the experiment runs.
