# CODEX / CLOUD HANDOFF — Universal AC-MOT V7 cycle (2026-09-28)

## 1. Why this handoff exists
The V7 development cycle started in a Claude desktop session running on the
owner's LOCAL Mac, not in Claude Cloud:
- MacBook Neo, Apple A18 Pro (6 cores), 8 GB;
- no NVIDIA GPU, no CUDA.

The owner then instructed that development runs must NOT use the local
Mac/MPS and must run in Claude Cloud. Every run was stopped at that point.
This file lets a cloud session (or Codex) continue without repeating
completed work.

## 2. Authoritative state
- **Repository:** github.com/AhmedCode110/AC-MOT, branch
  `universal-adapters-v1`. The V7 commit is recorded by `git log`
  ("V7 development …"). If the push did not happen, the local Mac holds
  the only copy: see §9.
- **V6-TF:** FROZEN at `2cff95f` (tag `universal-acmot-v6-freeze`). Never
  edit the files in `research/V6TF_POLICY_LOCK.json`. That includes
  `tools/eval_local.py`, `tools/seqstats.py` and `tools/v6/eval_official.py`,
  which V7 reuses read-only.
- **V7:** NOT frozen. New files only:
  - `acmot_v7.py` (layer: `V7Spec`, `HostContract`, `V7Layer.step/observe`);
  - `tools/v7/systems.py` (named systems + `@k=v` parser);
  - `tools/v7/dev.py` (VisDrone/UAVDT runner: internal + official-compatible reports);
  - `tools/v7/external/{sparsetrack_v7,boosttrack_v7}.py` (MOT17 development hosts);
  - `tools/v7/collect.py` (writes `research/final/V7_DEV_RESULTS.json`);
  - diagnostics `tools/v7/{streams,diag_*,screen,censored_mixture}.py`.
- **Records:**
  - `research/final/V7_EXPERIMENT_LEDGER.md` — every experiment, including
    rejected ones;
  - `research/final/V7_CLOUD_RUNS.md` — execution record;
  - `research/final/V7_DEV_RESULTS.json` — all metrics.

## 3. V7 architecture (development version)
Per frame t, with host contract (assoc, birth, low, match) declared by the
tracker adapter:
1. **Stream statistics from frames < t only.** Nested exact Otsu on the pooled
   logits of frames t−10..t−1 gives t1 (background|foreground) and
   t2 (ambiguous|confident).
2. **Regime.** ρ = confident share of the foreground; the median of ρ over
   the last 100 frames decides it: clean iff ρ̄ ≥ ½. Frame 1 is "cold": the
   host acts natively.
3. **Clean regime (self-limiting).**
   - The host's own thresholds are kept. They are only lowered to t2 when the
     host would reject the stream's confident class (`clean=upper`).
   - Raw scores are passed through.
   - Only cross-class duplicates are removed (`dup_clean=xclass`: a candidate
     overlapping, IoU > ½, a stronger candidate with a different class label).
   - The host keeps its low stage.
4. **Noisy regime.**
   - V6 bands: primary ≥ t2; extension [t1, t2); discard < t1.
   - ECDF score remap (`scores=auto`).
   - Track-context duplicates (`dup=track`): an overlapping weaker candidate
     is removed unless a DIFFERENT track of frame t−1 claims it (IoU ≥ ½).
5. **Motion-conditioned IoU-match tolerance** (V6): V7c applies it always;
   V7d only in noisy frames.
6. **State updates** with frame-t data affect frame t+1 on only.

Pass-through identity holds: with the regime forced clean and every rule off
(`NATIVE`), outputs are byte-identical to the official SparseTrack and
BoostTrack replays.

## 4. Best candidates and status (details: ledger E1–E10)
| System | SparseTrack (baseline 68.88/77.85/81.97) | BoostTrack online (68.49/75.50/81.41) | VisDrone val-7 YOLO / RT-DETR | development-40 YOLO / RT-DETR | Faster R-CNN val-7 |
|---|---|---|---|---|---|
| V6-TF (frozen) | 64.72/71.71/77.49 | 62.61/66.64/75.19 | 18.6/34.3/38.6 · 25.0/41.6/48.1 | 25.5/35.8/43.5 · 29.6/39.9/47.7 | 20.2/37.0/44.0 |
| V7c | 68.81/77.99/81.77 | 68.16/75.31/81.07 | 18.4/34.5/38.6 · 23.5/41.6/48.0 | 25.6/36.0/43.6 · 29.4/40.8/48.6 | 18.6/37.4/44.2 |
| V7d | **68.93/77.93/82.13** | 68.37/75.17/81.19 | 18.4/33.9/37.8 · 23.4/41.5/47.9 | 25.6/36.0/43.5 · 29.5/40.8/48.6 | not run |

Tables give HOTA/MOTA/IDF1 for MOT17 and MOTA/HOTA/IDF1 for VisDrone.

Catastrophic cells (V7c):
- val-7: 0 (host native 5);
- development-40: 2 (native 17, V6 2);
- Faster R-CNN val-7: 0 (native 6).

**Rejected so far:**
- temporal-support precision proxy (D5b);
- censored Gaussian mixture (D7);
- host-domain Otsu alone (D6: catastrophic under temperature 2);
- scene-evidence duplicate rule `ctx` (E4);
- duplicate handling restricted to primaries (E5);
- track memory as a fix (E3: superseded).

**Status by system:**
- **SparseTrack:** severe V6 collapse removed (V7d ≈ +0.05 HOTA, n.s. not
  yet tested).
- **BoostTrack:** −0.12 HOTA online / −0.07 post-processed with V7d; no
  catastrophic negative transfer.
- **Faster R-CNN:** V7c keeps V6's gain (HOTA +0.4 vs V6), MOTA −1.6 vs V6.
- **RT-DETR:** development-40 HOTA +0.9 vs V6; val-7 MOTA −1.5 vs V6.
- **VisDrone:** ≈ V6; ID switches +10–32% vs V6 (open).
- **UAVDT:** NOT yet run for V7.

## 5. Exact next experiments (in order)
1. **E11 — motion rule conditioning.** Add a generic `HostContract` field
   `cmc: bool` (does the host compensate camera motion itself?):
   - ByteTrack: False;
   - BoT-SORT, SparseTrack (GMC), BoostTrack (ECC): True.

   Motion loosening applies only to hosts without compensation. Compare with
   V7c and V7d on val-7, development-40, SparseTrack and BoostTrack. This is
   a declared capability, like the native thresholds; not a tracker-name
   branch.
2. **E12 — VisDrone ID switches.** Frame-1 admission (`cold`) vs pass-through
   principle; check where the extra IDS arise per sequence (clean vs noisy
   frames).
3. **Stage D (development transfer)**, with `V7_SPLIT` set per split:
   - UAVDT: `V7_SPLIT=uavdt` for YOLO + RT-DETR, and `V7_DETS=fasterrcnn`;
   - test-dev: `V7_SPLIT=testdev`, same detectors;
   - BoT-SORT: `"<S>@trk:botsort"` on val-7 and development-40.

   Keep confirmation-16 UNTOUCHED (post-freeze internal check).
4. **Stress tests:**
   - floors `@floor=0.05`, `@floor=0.1`, `@floor=0.2`;
   - transforms `@t:temp2`, `@t:temp05`, `@t:pow3`, `@t:scale05` on val-7;
   - the same via `--system "V7x@t:temp2"` on SparseTrack and BoostTrack
     (both drivers support `t:` and `floor=`).
5. **Tests** — `tests/test_v7_adaptive_layer.py`:
   - causality: perturb frames ≥ k, and outputs < k and the frame-k
     thresholds are unchanged;
   - state reset;
   - pass-through identity;
   - no detector/tracker/dataset names in `acmot_v7.py`;
   - affine-logit invariance of the noisy-regime bands;
   - floor tests;
   - crowd/duplicate unit cases: two tracked people overlapping must both be
     kept; a near-coincident duplicate must be removed.
6. **Paired bootstrap** (10,000 resamples, seed 42): every host,
   V7 vs baseline and V7 vs V6, using `tools/v6/external/mot17_eval.py --boot`
   and the V6 `tools/v6/confirm_report.py` logic.
7. **Freeze V7:**
   - config `configs/universal_acmot_policy_v7.json`;
   - lock hashes `research/V7_POLICY_LOCK.json`;
   - tag `universal-acmot-v7-freeze`.
8. **Predeclare the external set BEFORE running frozen V7:**
   - TOPICTrack (IEEE TIP 2025; untouched; same YOLOX checkpoint;
     weights `topictrack_ablation`, `mot17_sbs_S50` already downloaded
     locally);
   - plus 1–3 more 2025/2026 Q1/Q2 systems with official code and weights,
     verified with sources (see `research/final/EXTERNAL_PAPER_SELECTION.md`
     for the V6 candidate matrix).
9. **Final real-time benchmark** on ONE fixed device, AFTER freeze only:
   - Baseline vs Baseline + frozen V7;
   - batch 1, sequential, full pipeline;
   - report hardware, precision, resolution, mean/P95 latency, FPS and
     V7 overhead in ms and %.

## 6. Commands (repo root, `PYTHONPATH=.`)
**VisDrone runner**
```
python tools/v7/dev.py run V7c V7d "V7c@t:temp2" …    # V7_SPLIT=val7|dev40|testdev|uavdt, V7_DETS=yolov8,rtdetr|fasterrcnn
python tools/v7/dev.py report NATIVE V6:X5 V7c …      # V6 runs are read from outputs/v6 as V6:<name>
python tools/v7/dev.py official V7c …                 # official-compatible VisDrone port
python tools/v7/dev.py seq V6:X5 V7c
```

**SparseTrack (external venv)**
```
python tools/v7/external/sparsetrack_v7.py --system V7d --name ST7_V7d    # --system BASELINE for the replay
```

**BoostTrack (external venv)**
```
python tools/v7/external/boosttrack_v7.py  --system V7d --name BT7_V7d
```

**MOT17 evaluation and bootstrap**
```
python tools/v6/external/mot17_eval.py <runs_root> ST7_BASELINE ST7_V7d
python tools/v6/external/mot17_eval.py --boot <runs_root> ST7_BASELINE ST7_V7d
```

**Collect all development results**
```
python tools/v7/collect.py
```

## 7. Data a cloud session needs (NOT in git)
Sizes are measured on the Mac.

| Item | Local path | Size | Public source / how to rebuild |
|---|---|---|---|
| VisDrone native detection caches | `outputs/det_cache_{val,train,testdev,uavdt}_native/` (+ `visual_cues/`) | 13 + 74 + 78 + 119 MB | Copy from the owner, or rebuild with `tools/cache_detections.py --weights W --dataset D --output-dir O --nms none --resolutions 736` (YOLOv8n `yolov8n.pt`, RT-DETR-L `rtdetr-l.pt`, Faster R-CNN torchvision v2, sha256 in `research/final/REPRODUCIBILITY.md`) |
| UAVDT view | `outputs/uavdt_view` | 10 MB (+ images) | `tools/build_uavdt_view.py` |
| VisDrone2019-MOT val / train / test-dev (annotations + images; the runner counts `*.jpg` per sequence, BoT-SORT reads images) | Google Drive `AC-MOT-shared/AC-MOT-data/…` (symlinked) | several GB | public VisDrone2019-MOT release |
| MOT17 val-half | `/Users/ahmedgouda/Desktop/acmot_external/data_mirror/MOT17` (byte-identical mirror, sha256 manifest) | 456 MB (val-half images + `annotations/*.json`) | MOT17 from motchallenge.net; split verified by `tools/v6/external/verify_mot17_split.py` (hashes in `vendor/mot17_split_verification.json`) |
| SparseTrack published detections | `…/acmot_external/runs/sparsetrack_A_official/published_detections.pkl` | 1.9 MB | or rerun the official detector (YOLOX-X, `bytetrack_ablation.pth.tar`, sha256 26cb8d28…) |
| BoostTrack detections + ECC cache | `…/acmot_external/BoostTrack/cache/` | 2.5 MB | BoostTrack's own cache |
| SparseTrack @499844f, BoostTrack @fb5bfc3 | `…/acmot_external/{SparseTrack,BoostTrack}` | — | clone at the pinned commits + `tools/v6/external/vendor/*_compat.diff`; SparseTrack GMC shim `vendor/gmc_shim.cpp` + `pbcvt.py` (OpenCV videostab) |
| MOT17 val-half GT | `…/acmot_external/BoostTrack/results/gt/MOT17-val` | small | shipped in the BoostTrack repo |
| TrackEval @12c8791 | `/Users/ahmedgouda/Desktop/CUE_SELECTION/cue_ablation_tools/TrackEval` | small | github.com/JonathonLuiten/TrackEval |

**Hard-coded Mac paths.** The locked V6 files (`tools/eval_local.py` TRACKEVAL)
and the V6/V7 tools use absolute Mac paths. The locked files must not be
edited. In the cloud, recreate the same absolute paths with symlinks:
```
mkdir -p /Users/ahmedgouda/Desktop && ln -s <cloud>/… …
```
Paths used:
- `/Users/ahmedgouda/Desktop/{Universal-ACMOT,acmot_external,CUE_SELECTION}`;
- the Google-Drive VisDrone paths in `tools/v6/dev.py` SPLITS.

## 8. Scientific constraints (unchanged)
- Training-free, online, causal: frame-t control uses frames < t plus
  frame-t image cues.
- No labels, and no detector, tracker, dataset or sequence names or branches.
- SparseTrack and BoostTrack are DEVELOPMENT hosts for V7, never external
  evidence.
- TOPICTrack must stay untouched until V7 is frozen and the external list is
  predeclared.
- confirmation-16 is reserved for a single post-freeze V7 check.
- No policy change after the freeze based on external results.
- Keep negative results.
- Report timing only with the exact hardware; the final speed comparison is
  Baseline vs Baseline + frozen V7 on one fixed device, after the freeze.

## 9. If the branch was not pushed
The V7 development commit exists only on the owner's Mac
(`/Users/ahmedgouda/Desktop/Universal-ACMOT`). Run from that folder:
```
git push origin universal-adapters-v1 && git push origin universal-acmot-v6-freeze
```
