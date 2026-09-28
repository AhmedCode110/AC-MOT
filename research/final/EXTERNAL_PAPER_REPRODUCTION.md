# EXTERNAL PAPER REPRODUCTION / FAITHFUL EXECUTION

Owner directive: faithful execution of the final published method with the
official code, weights and config, rather than chasing exact paper numbers.
Paper numbers are quoted for context.

## Common data and evaluation
- **Data.** MOT17 from the owner's Google Drive
  (`My Drive/AC-MOT-SparseTrack-NEW/data/MOT17`: train and test, 7 FRCNN
  sequences each; annotations train/train_half/val_half/test_frcnn_private).
- **Split.** MOT17 val-half. Verified frame-for-frame against BoostTrack's
  official split definition (`data/tools/convert_mot17_to_coco.py`, ByteTrack
  convention: frames n//2+2 … n per sequence, `frame_id` renumbered from 1):
  - 2,652 images and 53,890 annotations identical;
  - val-half GT derived from the raw `gt.txt` byte-identical to the authors'
    shipped evaluation GT (`results/gt/MOT17-val`) for all 7 sequences;
  - `val_half.json` sha256 ae7ac86d61df6cdba59e3b7a78ef7a3a8b7b56978d3a8322971ef6d8dfa5c126;
  - report: `acmot_external/reports/mot17_split_verification.json`, tool
    `tools/v6/external/verify_mot17_split.py`.

  SparseTrack uses the same ByteTrack conversion (val_half.json), so the
  split is shared.
- **Mirror.** A byte-identical local mirror of the 2,652 val-half frames plus
  annotations and GT (sha256 manifest `data_mirror/MOT17_mirror_manifest.json`)
  avoids re-streaming from Drive for each run. The Drive copy stays
  authoritative.
- **Detector checkpoint.** `bytetrack_ablation.pth.tar`, sha256
  26cb8d2808664e5068a4c812d53becbc948b47fd6eacf2b45db049ab40c48b1a. It is
  identical for SparseTrack, BoostTrack and TOPICTrack
  (`topictrack_ablation.pth.tar` has the same hash), and identical to the
  owner's Drive copy.
- **Evaluator.** TrackEval (pinned 12c8791), official MOTChallenge MOT17
  protocol, with the authors' shipped val-half GT. Sanity: GT as tracker
  output gives 100/100/100. Tool: `tools/v6/external/mot17_eval.py`.
  SparseTrack's in-repo motmetrics path is also available.
- **Hardware.** Apple Silicon Mac, PyTorch 2.14 MPS, fp16 detector as
  published. No CUDA.

## BoostTrack (supporting): Machine Vision and Applications 2024
- Official code: github.com/vukasin-stanojevic/BoostTrack @ fb5bfc3.
- README setting "Run BoostTrack": `--no_reid --btpp_arg_iou_boost
  --btpp_arg_no_sb --btpp_arg_no_vt`, plus `--s_sim_corr` (the corrected
  shape similarity whose MOT17-val numbers the authors re-reported in issue
  #8).
- Compatibility-only changes (git diff in the clone):
  - `.cuda()` → MPS device;
  - `torch.load(map_location='cpu', weights_only=False)`;
  - YOLOX `tensor.type('torch.mps.HalfTensor')` → `.to(dtype, device)`;
  - `run_mac.py`: spawn-safe entry and ReID-module stubs; the ReID modules are
    never called with `--no_reid`.

| Metric (MOT17 val-half) | Authors (issue #8, Table 3: SB+DCB, CMC) | Our execution | Δ |
|---|---:|---:|---:|
| HOTA, online | 68.371 | 68.492 | +0.12 |
| MOTA, online | 75.561 | 75.502 | −0.06 |
| IDF1, online | 81.354 | 81.413 | +0.06 |
| IDSW, online | 118 | 113 | −5 |
| HOTA, + LI | 71.066 | 71.474 | +0.41 |
| MOTA, + LI | 80.538 | 80.996 | +0.46 |
| IDF1, + LI | 83.809 | 84.128 | +0.32 |
| HOTA, + GBI | 71.326 | 71.725 | +0.40 |
| MOTA, + GBI | 80.549 | 81.032 | +0.48 |
| IDF1, + GBI | 83.839 | 84.163 | +0.32 |

Verdict: **credible reproduction.** The online setting is within ±0.12
HOTA/MOTA/IDF1. The replay driver used for the +V6 arm reproduces this
official output byte-for-byte on all 7 sequences.

## SparseTrack (headline): IEEE TCSVT 2025
- Official code: github.com/hustvl/SparseTrack @ 499844f (the owner's pinned
  commit).
- Released MOT17 ablation config `mot17_ab_track_cfg.py`:

  | Setting | Value |
  |---|---|
  | Detector input size | 800×1440 |
  | Detector conf / NMS | 0.01 / 0.7 |
  | Precision | fp16 |
  | `track_thresh` / buffer / `match_thresh` | 0.6 / 30 / 0.85 |
  | `min_box_area` | 100 |
  | GMC `down_scale` | 4 |
  | `depth_levels` / `depth_levels_low` | 1 / 8 |
  | `confirm_thresh` | 0.7 |
  | Tracker | `deep=True` (SparseTracker) |

  The paper text states 3 pseudo-depth levels for MOT17; the released config
  is executed as published.
- Paper-reported (Table VI, MOT17 val): HOTA 69.2, AssA 72.3, IDF1 81.4,
  MOTA 76.8. The owner's earlier deterministic Colab/T4 reproduction of the
  same code and checkpoint (motmetrics) gave MOTA 76.8658, IDF1 81.6698,
  IDS 118, FP 2794, FN 9555.
- Compatibility-only changes (git diff in the clone):
  - `register_data.py` dataset paths (the authors' paths pointed at MOT20);
  - device-agnostic tensor dtype in the evaluator (was `torch.cuda.*Tensor`);
  - YOLOX head dtype strings → `.to()`;
  - NumPy-2 aliases (`np.float` → builtin `float`, identical meaning);
  - DataLoader workers 0 (the pinned deterministic protocol's patch);
  - `KMP_DUPLICATE_LIB_OK` for the second OpenMP runtime brought in by OpenCV;
  - GMC: SparseTrack's `python_module.cpp::GMC` body copied verbatim
    (OpenCV videostab KeypointBasedMotionEstimator + RANSAC-L2 similarity),
    compiled against Homebrew OpenCV 5.0 and called through a C ABI (ctypes)
    instead of Boost.Python/pbcvt, which only converts arrays. The original
    runtime used Ubuntu OpenCV 4.6.
- Val-ablation instruction followed: the authors' comment says to invalidate
  the per-sequence overrides for ablation. They were disabled, so MOT17-05
  and MOT17-13 use the official buffer 30.
- Results: see EXTERNAL_PAPER_TRANSFER.md (execution vs paper table).
