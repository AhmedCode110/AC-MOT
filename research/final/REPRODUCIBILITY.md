# REPRODUCIBILITY — V6-TF

## Code and freeze
- Repository: github.com/AhmedCode110/AC-MOT, branch `universal-adapters-v1`.
- Freeze: tag `universal-acmot-v6-freeze` → commit
  `2cff95f8a565fdff17f5b1fb06c23cf78e0ffed0`. The follow-up commits
  (b489113, 6f82ff7, …) add documentation, evaluation tooling and the
  transfer lock only. `tools/v6/dev.py` refuses any protected run when a
  file hash in `research/V6TF_POLICY_LOCK.json` differs.
- Policy: `configs/universal_acmot_policy_v6tf.json`. Method description:
  `FINAL_METHOD.md`. Freeze record: `FREEZE_RECORD_V6TF.md`.
- Unknown-provenance WIP that predates this work is preserved in git stash
  `51a46971` and is NOT part of any result.

## Environment (Mac development machine)
Python 3.12.14 · torch 2.14.0 · ultralytics 8.3.200 · numpy 2.2.6 · scipy
1.18.1 · opencv 4.11.0 · motmetrics 1.4.0 · TrackEval 12c8791 (local path in
`tools/eval_local.py`) · macOS 27.0 arm64. Detection caches were built on
Mac MPS. The Amendment-5f T4 fidelity gate and official timing are deferred
to the pre-paper step (Amendment 9 §5).

## Weights (sha256)
| Model | File | sha256 |
|---|---|---|
| YOLOv8n (COCO) | yolov8n.pt | f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 |
| RT-DETR-L (COCO) | rtdetr-l.pt | 6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f |
| Faster R-CNN R50-FPN v2 (COCO, torchvision) | weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth | dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf |

## Data
- VisDrone2019-MOT-val: 7 sequences, the V6-TF development sandbox.
- VisDrone2019-MOT-train: split `research/TRAIN_SPLIT_V5.json` (seed
  20260927, annotation fingerprint 045620e4…) into development-40 and
  confirmation-16.
- VisDrone2019-MOT-test-dev: 17 sequences, post-hoc.
- UAVDT test: 20 sequences, through `outputs/uavdt_view`
  (`tools/build_uavdt_view.py`; vehicles; ignore regions).

## Detection caches (commands)
`tools/cache_detections.py --weights W --dataset D --output-dir O --nms N --resolutions 736`
- The native caches (`outputs/det_cache_*_native`) hold detector-native
  suppression: YOLO 0.7, RT-DETR none, Faster R-CNN 0.5. They are used by
  V6-TF and E41.
- The 0.45 caches (`outputs/det_cache*`) are used by V4 and the static
  baselines. The Faster R-CNN 0.45 caches were built after the freeze by
  `tools/v6/cache_queue_frcnn045.sh` (caching only).
- Image cues: `visual_cues/` next to each cache (`scene_state.image_stats`).

## Commands (all from the repository root, `PYTHONPATH=.`, `.venv/bin/python`)
| Result | Command |
|---|---|
| val-7 development runs | `tools/v6/dev.py run <system...>` then `tools/v6/dev.py report <system...>` / `seq` / `official` |
| development-40 check | `V6_SPLIT=dev40 tools/v6/dev.py run V4 shared_static E41 X5` |
| stress / ablation variants | `tools/v6/dev.py run "X5@t:temp2" "X5@dedup_iou=0" "X5@gate_window=5" …` |
| confirmation-16 (one-way) | `V6_SPLIT=conf16 tools/v6/dev.py run V6TF V4 shared_static static_default E41`; `V6_SPLIT=conf16 tools/v6/dev.py official …`; `V6_SPLIT=conf16 tools/v6/confirm_report.py V6TF V4 shared_static static_default E41` |
| BoT-SORT | `V6_SPLIT=conf16 tools/v6/dev.py run "V6TF@trk:botsort" "V4@trk:botsort" "static_default@trk:botsort" "shared_static@trk:botsort"` |
| test-dev post-hoc | `V6_SPLIT=testdev tools/v6/dev.py run V6TF V4 shared_static static_default` |
| UAVDT | `V6_SPLIT=uavdt tools/v6/dev.py run V6TF V4 shared_static static_default`; report with `V6_PROTOCOLS=internal` |
| Faster R-CNN | `V6_DETS=fasterrcnn V6_SPLIT=<val7/testdev/uavdt> tools/v6/dev.py run V6TF V4 shared_static static_default` |
| live == replay parity | `tools/v6/live_replay_parity.py` |
| tests | `ACMOT_WEIGHTS=… RTDETR_WEIGHTS=… ACMOT_SEQUENCE=… ACMOT_FRAME=… pytest -q tests` |
| tables | `tools/v6/make_tables.py` → `research/final/TABLES/` |

## Determinism
- The layer has no randomness. A clean rerun of the frozen config is
  byte-identical to the development candidate on all 14 val-7 cells.
- Ultralytics track IDs come from a process-global counter; every run
  resets it (`BaseTrack.reset_id`) or runs in a fresh process.
- Bootstrap: 10,000 paired sequence resamples, `numpy.random.default_rng(42)`.

## Evaluation protocols (never mixed)
1. Internal class-agnostic (canonical for the project's history):
   `tools/eval_local.py` + `tools/seqstats.py`.
2. Official-compatible VisDrone Task-4b port: `tools/v6/eval_official.py`,
   with the sanity check that GT as the tracker output scores 100/100/100.
   It is class-aware; COCO detectors have no "van", so van GT is always FN.
   UAVDT uses its own ignore-region rule (`tools/seqstats.apply_ignore_regions`).
