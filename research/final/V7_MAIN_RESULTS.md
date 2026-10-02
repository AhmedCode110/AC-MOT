# V7 main results — published trackers: paper vs reproduction vs + AC-MOT

Frozen policy V7f (`configs/universal_acmot_policy_v7.json`, freeze commit
488df9a). The same configuration is used for every tracker; nothing is
retuned per tracker, detector or dataset. MOT17 val-half (7 sequences),
published YOLOX-X detections, TrackEval 12c8791. Δ = (tracker + AC-MOT) −
(our reproduction of the tracker). 95% CI from a paired sequence bootstrap
(10,000 resamples, seed 42). Machine-readable: `V7_MAIN_RESULTS.json`,
`V7_DEV_RESULTS.json`, `V7_EXTERNAL_RESULTS.json`.

Two roles:
- **Development baselines** (SparseTrack, BoostTrack, ByteTrack, OC-SORT):
  published trackers whose results were visible while AC-MOT was designed.
  Their gains are real measured gains over the reproduced baseline, but they
  are development evidence, not held-out evidence.
- **External systems** (PD-SORT, Hybrid-SORT): selected and declared before
  they were run (`V7_EXTERNAL_SELECTION.md`), evaluated once with the frozen
  policy.

## Table 1 — MOT17 val-half, HOTA / MOTA / IDF1

| Tracker (venue) | Role | Paper | Our reproduction | + AC-MOT (V7f) | ΔHOTA [95% CI] | ΔMOTA | ΔIDF1 |
|---|---|---|---|---|---|---|---|
| SparseTrack (IEEE TCSVT 2025) | dev | 69.2 / 76.8 / 81.4 | 68.876 / 77.849 / 81.974 | 68.931 / 77.927 / 82.130 | +0.055 [−0.011, +0.254] | +0.078 [−0.027, +0.304] | +0.156 [−0.015, +0.696] |
| BoostTrack (MVA 2024), online | dev | 68.371 / 75.561 / 81.354 | 68.492 / 75.502 / 81.413 | 68.492 / 75.502 / 81.413 | 0 (identical output) | 0 | 0 |
| BoostTrack (MVA 2024), + GBI | dev | 71.326 / 80.549 / 83.839 | 71.725 / 81.032 / 84.163 | 71.725 / 81.032 / 84.163 | 0 (identical output) | 0 | 0 |
| ByteTrack (ECCV 2022), floor 0.01 | dev | – / 76.6 / 79.3 | 67.698 / 77.604 / 79.471 | 67.684 / 77.662 / 79.440 | −0.014 [−0.058, +0.009] | +0.058 | −0.031 |
| ByteTrack (ECCV 2022), floor 0.1 | dev | – / 76.6 / 79.3 | 67.698 / 77.604 / 79.471 | 67.698 / 77.604 / 79.471 | 0 (identical output) | 0 | 0 |
| OC-SORT (CVPR 2023), floor 0.01 | dev | 66.5 / 74.9 / 77.7 | 66.428 / 74.672 / 78.052 | **67.041 / 75.940 / 78.708** | **+0.613 [+0.363, +1.241]** | **+1.267 [+0.173, +3.014]** | +0.656 [−0.248, +1.598] |
| OC-SORT (CVPR 2023), floor 0.1 | dev | 66.5 / 74.9 / 77.7 | 66.443 / 74.669 / 78.046 | **66.907 / 75.931 / 78.333** | **+0.464 [+0.266, +1.086]** | **+1.262 [+0.367, +3.343]** | **+0.287 [+0.003, +0.838]** |
| PD-SORT (IEEE TCE 2025) | external | see `V7_EXTERNAL_TRANSFER.md` | 68.011 / 75.185 / 81.032 | **68.624 / 76.313 / 81.850** | **+0.61 [+0.27, +1.65]** | **+1.13 [+0.27, +3.03]** | **+0.82 [+0.43, +1.94]** |
| Hybrid-SORT (AAAI 2024) | external | 67.1 / 75.8 / 78.0 | 66.698 / 75.517 / 77.556 | 66.698 / 75.517 / 77.556 | 0 (identical output) | 0 | 0 |

Bold: CI excludes 0. Paper sources: SparseTrack Table VI (MOT17 val);
BoostTrack corrected numbers from the authors (GitHub issue #8); ByteTrack
val-half ablation (HOTA not reported; the paper ran its own detector
inference); OC-SORT val-half; Hybrid-SORT README. Reproduction details:
`EXTERNAL_PAPER_REPRODUCTION.md` (SparseTrack, BoostTrack),
`V7_FALLBACK_VALIDATION.md` (ByteTrack, OC-SORT), `V7_EXTERNAL_TRANSFER.md`
(PD-SORT, Hybrid-SORT).

Note 1. SparseTrack reads the MOT17 frames (GMC and the AC-MOT motion cue).
Its runs were executed on a GitHub Actions runner (ubuntu-24.04, AMD EPYC
7763 × 4, OpenCV 4.6.0 for the GMC shim; workflow
`.github/workflows/sparsetrack_v7f.yml`, script `tools/v7/ci/sparsetrack_v7f.sh`)
at repo commit 79749c5. The runner downloaded MOT17 from motchallenge.net and
verified every file against `MOT17_mirror_manifest.json` (2669/2669 sha256
identical to the owner's Drive copy). Before running, it checked the V7
policy lock (10/10 files match). The runner's baseline is the same
reproduction as the earlier one: same tracks, identities and frames in all 7
sequences; box coordinates differ by at most 0.1 px (last written decimal,
GMC under OpenCV 4.6 vs 5.0); metrics identical. SparseTrack + V7f gives
exactly the V7d numbers. Record: `sparsetrack_v7f/` (results.json,
summary.txt, environment.txt, pip_freeze.txt, tracks.tar.gz).

## Table 2 — per-sequence ΔHOTA (V7f − reproduction), MOT17 val-half

| Tracker | 02 | 04 | 05 | 09 | 10 | 11 | 13 | improved |
|---|---|---|---|---|---|---|---|---|
| OC-SORT, floor 0.01 | +0.25 | +0.54 | +0.45 | +2.90 | +1.27 | +0.34 | +0.03 | 7/7 |
| OC-SORT, floor 0.1 | +0.42 | +0.21 | +0.37 | +1.40 | +1.14 | +0.71 | +1.48 | 7/7 |
| SparseTrack | +0.09 | 0.00 | 0.00 | 0.00 | +0.52 | 0.00 | −0.07 | 2/7 (5 unchanged) |
| ByteTrack, floor 0.01 | 0.00 | 0.00 | −0.01 | 0.00 | +0.04 | 0.00 | −0.21 | – |
| ByteTrack, floor 0.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | identical |
| BoostTrack online / + GBI | 0 | 0 | 0 | 0 | 0 | 0 | 0 | identical |
| PD-SORT | see `V7_EXTERNAL_RESULTS.json` | | | | | | | 7/7 |

## Table 3 — the same trackers under detector calibration shift (MOT17 val-half, floor 0.01)

The detector scores are transformed before the tracker (`pow3`: s³;
`scale05`: 0.5·s; `temp2` / `temp05`: logit temperature 2 / 0.5). The tracker
keeps its published thresholds. All cells are listed, including those with
no gain. HOTA, reproduction → + AC-MOT.

| Tracker | pow3 | scale05 | temp2 | temp05 |
|---|---|---|---|---|
| ByteTrack (official setting) | 60.92 → **65.81** | 0.00 → **66.17** | 65.84 → **66.95** | 67.30 → 67.29 |
| ByteTrack (ultralytics setting) | 66.31 → 66.31 | 66.61 → 66.48 | 63.97 → 63.97 | 66.53 → 66.53 |
| OC-SORT | 58.34 → **66.60** | 0.00 → **66.72** | 65.99 → **67.22** | 66.61 → **67.06** |
| BoostTrack online (floor 0.1) | 61.33 → **64.84** | – | 67.59 → 67.59 | – |

## Table 4 — the same trackers with a different detector (KITTI tracking training, HOTA car/pedestrian averaged)

| Tracker · detector | Reproduction HOTA / MOTA / IDF1 | ΔHOTA [95% CI] | ΔMOTA [95% CI] | ΔIDF1 [95% CI] |
|---|---|---|---|---|
| ByteTrack · RT-DETR-L | 50.73 / 47.27 / 64.98 | **+0.91 [+0.22, +1.55]** | **+5.98 [+1.16, +9.94]** | **+3.87 [+1.67, +5.06]** |
| BoT-SORT · RT-DETR-L | 53.91 / 48.25 / 67.74 | +0.61 [−0.39, +1.61] | **+7.86 [+2.21, +13.44]** | **+3.10 [+1.00, +4.57]** |
| OC-SORT · RT-DETR-L | 52.64 / 57.59 / 69.87 | +0.19 [−0.57, +1.19] | −0.57 [−4.36, +1.40] | +0.04 [−1.31, +1.78] |
| OC-SORT · YOLOv8n | 36.27 / 33.99 / 50.46 | **+7.91 [+5.47, +9.98]** | **+9.40 [+1.10, +12.30]** | **+9.60 [+5.61, +12.16]** |
| ByteTrack · YOLOv8n | 45.31 / 46.54 / 60.92 | +0.06 [−1.19, +1.24] | −1.87 [−6.25, +0.03] | +1.49 [−0.36, +3.27] |
| BoT-SORT · YOLOv8n | 49.21 / 49.89 / 64.41 | −1.02 [−2.43, +0.25] | **−2.43 [−7.43, −0.14]** | +0.35 [−2.10, +2.27] |

BoT-SORT excludes sequence 0020 (numerical failure of the unmodified tracker).

## Reading of the results
1. **Where the tracker's own operating point already fits a clean detector**
   (SparseTrack, BoostTrack, ByteTrack on YOLOX-X MOT17), AC-MOT leaves the
   output unchanged or within ±0.06 HOTA (CIs include 0). The frozen V6
   cost these same trackers 4–6 HOTA (SparseTrack −4.15, BoostTrack −5.89),
   so the absence of harm is itself the V6 → V7 result. These trackers are
   not reported as improved on clean MOT17 because they were not improved
   there.
2. **Where the tracker lacks a continuation stage** (OC-SORT, PD-SORT),
   AC-MOT improves HOTA, MOTA and IDF1, with CIs above 0 and gains on 7/7
   sequences. PD-SORT is held-out evidence.
3. **Where the detector's scores no longer match the tracker's fixed
   thresholds** (calibration shift, Table 3), AC-MOT restores the same
   published trackers, including collapsed ones (ByteTrack and OC-SORT from 0
   to 66–67 HOTA). This is where the two-stage trackers gain.
4. **With a noisy detector** (KITTI RT-DETR-L, Table 4), ByteTrack and
   BoT-SORT gain 6–8 MOTA and 3–4 IDF1.
5. **Known regression**: an under-confident detector with a low-threshold
   two-stage tracker (KITTI YOLOv8n with ByteTrack / BoT-SORT), MOTA −1.9 /
   −2.4, while ID switches fall by about half.
