# ENVIRONMENT AND PATHS (no secrets here — never add tokens/passwords/keys)

## Repository
- GitHub: https://github.com/AhmedCode110/AC-MOT (remote `origin`)
- Working branch: `universal-adapters-v1` (legacy `main` = original AC-MOT)
- Mac checkout: `/Users/ahmedgouda/Desktop/Universal-ACMOT`
  (NOT `/Users/ahmedgouda/Desktop/new acmot/Universal-ACMOT`, an unrelated empty repo)
- Colab checkout: `/content/Universal-ACMOT` (notebooks/Colab_T4_gate_and_benchmark.ipynb)

## Python (Mac development, `.venv`, verified 2026-09-27)
python 3.12.14 · torch 2.14.0 · torchvision 0.29.0 · ultralytics 8.3.200 ·
numpy 2.2.6 · scipy 1.18.1 · motmetrics 1.4.0 · optuna 5.0.0 ·
opencv 4.11.0 · pandas 3.0.6. Device: Apple MPS (development only).
Run tools with `PYTHONPATH=. .venv/bin/python ...`; set
OMP/MKL/VECLIB_MAXIMUM_THREADS=1 for parallel replays.

## Evaluator
TrackEval commit 12c8791 at `/Users/ahmedgouda/Desktop/CUE_SELECTION/cue_ablation_tools/TrackEval`
(path set in tools/eval_local.py); motmetrics 1.4.0.

## Detector weights (frozen, COCO-pretrained; never retrained)
- Detector: YOLOv8n — `/Users/ahmedgouda/Desktop/CUE_SELECTION/yolov8n.pt` (sha256 f59b3d83…)
- Detector: RT-DETR-L — `rtdetr-l.pt` (repo root, git-ignored; sha256 6de60b10…)
- Detector: Faster R-CNN ResNet50-FPN v2 — `weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` (torchvision)

## Data (see DATASETS_AND_SPLITS.md)
- val: `/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val`
- train: `~/Library/CloudStorage/GoogleDrive-<user>/My Drive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-train`
- test-dev: Google Drive shortcut (full path in tools/mac_cache_queue_v5tf.sh)
- UAVDT: `…/AC-MOT-shared/UAVDT_EXTERNAL_GENERALIZATION`, view `outputs/uavdt_view`
- Colab results: Drive `My Drive/Universal-ACMOT-Results/`

## Generated artifacts (git-ignored)
`outputs/det_cache*` (NPZ detection caches), `outputs/v5/`, `outputs/v5tf_dev/`,
`outputs/heldout_v4/`, `outputs/opt_*`. Weights `*.pt`, `*.pth` ignored.

## Graphify
CLI `/Users/ahmedgouda/.local/bin/graphify` (uv tool `graphifyy`, Python 3.14);
global agent rules `~/AGENTS.md`, Codex hook `~/.codex/hooks.json`
(`graphify hook-check`). Repo graph: `graphify-out/` built by
`tools/build_context_graph.py` (see README.md).

## Hardware policy
Mac/MPS: development + caching only. Colab T4: official timing + fidelity gate.
