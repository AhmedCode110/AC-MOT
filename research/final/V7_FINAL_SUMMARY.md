# Universal AC-MOT V7 — final summary (2026-09-28)

## What V7 is
A training-free, online, causal adaptive control layer that sits between a
frozen detector and a frozen multi-object tracker and decides, per frame,
which candidates reach the tracker, with which scores and at which operating
point — from the detector's own score stream, the tracker's previous output
and the tracker's declared operating point only. No detector, tracker, dataset
or sequence names. Frozen policy V7f (`V7_METHOD.md`), freeze commit 488df9a.

## Story
- V6-TF (frozen) fixed miscalibrated/noisy streams (VisDrone RT-DETR) but cost
  strong published trackers 4–6 HOTA on MOT17 (crowd-unsafe duplicate removal,
  over-strict thresholds).
- V7 learns when not to intervene (ρ regime, host-anchored clean regime,
  crowd-safe duplicates), estimates the regime from the whole stream, checks
  that its statistics are interpretable (background mode), and supplies a
  track-continuation stage to trackers that have none. Rejected mechanisms are
  kept in the ledger (`V7_EXPERIMENT_LEDGER.md`, `V7_ABLATION.md`).

## Main table
Paper vs reproduction vs + frozen AC-MOT for SparseTrack, BoostTrack,
ByteTrack, OC-SORT (development baselines) and PD-SORT, Hybrid-SORT
(external): `V7_MAIN_RESULTS.md`. SparseTrack + V7f is the one open cell
(needs the MOT17 frames).

## Development evidence (contaminated; `V7_STATISTICS.md`)
- Two-stage hosts on clean streams (ByteTrack ×4, BoostTrack): unchanged.
- Single-stage OC-SORT: +0.61 / +0.46 HOTA on MOT17 (CIs > 0); +7.9 on KITTI YOLOv8n.
- Noisy detector (KITTI RT-DETR-L): ByteTrack / BoT-SORT MOTA +6.0 / +7.9, IDF1 +3.9 / +3.1.
- Score recalibration stress: collapsed hosts (0–61 HOTA) restored to 65–67.
- Known regression: KITTI YOLOv8n with ByteTrack / BoT-SORT MOTA −1.9 / −2.4.

## External evidence (post-freeze, predeclared; `V7_EXTERNAL_TRANSFER.md`)
| Published system | Faithful reproduction HOTA/MOTA/IDF1 | + frozen AC-MOT | Δ HOTA [95% CI] |
|---|---|---|---|
| PD-SORT (IEEE TCE 2025) | 68.011 / 75.185 / 81.032 | **68.624 / 76.313 / 81.850** | **+0.61 [+0.27, +1.65]** (MOTA +1.13, IDF1 +0.82, CIs > 0; 7/7 sequences) |
| Hybrid-SORT (AAAI 2024) | 66.698 / 75.517 / 77.556 | identical | 0 |

## Runtime (`V7_REALTIME.md`)
Controller 2.95 ms mean / 4.39 ms P95 per frame on a 4-vCPU Xeon;
end-to-end +2.59 ms (+4.5%) with live YOLOv8n + ByteTrack.

## Status of the objective
Achieved: one frozen general policy; faithful reproductions; a significant
improvement of one recent published tracker without retuning; no harm to a
second; development evidence across 4 trackers × 3 detectors × 2 datasets;
tests (61), bootstrap, lock, runtime.
Not yet achieved (needs network access to motchallenge.net / Google Drive):
a second (and third) 2025/26 published system with an improvement; a second
dataset / test-set external result; VisDrone/UAVDT labelled checks and the
reserved VisDrone confirmation-16; the freeze tag on GitHub (owner push).
