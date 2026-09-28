# V7 FALLBACK VALIDATION MATRIX (cloud environment C1)

Why this file exists: the cloud environment's network policy denies
`motchallenge.net` and Google Drive (403). The block was recorded once in
`V7_CLOUD_RUNS.md` and is not retried. This file lists every
dataset/system cell, what is available, and whether an EXACT evaluation or
only a DIAGNOSTIC approximation is possible with reachable resources.

Definitions:
- **exact**: original detections/tracker code/GT, the tracker reads nothing
  that is unavailable (verified where possible by byte identity);
- **diagnostic**: some input is missing or replaced (e.g. no image motion
  cue); results are labelled evidence for THAT configuration only, and never
  presented as a reproduction of the original protocol;
- **label-free**: no GT; statistics of the output only (never called
  accuracy).

## Development vs reserved (strict list)
| DEVELOPMENT / CONTAMINATED (used before the V7 freeze) | RESERVED / UNTOUCHED (post-freeze only) |
|---|---|
| VisDrone val-7, development-40, test-dev (V6 post-hoc), UAVDT (V6) | VisDrone confirmation-16 (one post-freeze V7 check) |
| SparseTrack (IEEE TCSVT 2025), BoostTrack (MVA 2024) | TOPICTrack (IEEE TIP 2025) |
| ultralytics ByteTrack / BoT-SORT (VisDrone hosts) | 1–3 further 2025/2026 systems, predeclared in `V7_EXTERNAL_SELECTION.md` before the freeze |
| ByteTrack on MOT17 val-half (official MOT17 operating point and ultralytics default) — added in C1 | |
| OC-SORT (CVPR 2023, noahcao/OC_SORT @ 8462e7e) on MOT17 val-half — added in C1 | |
| KITTI tracking TRAINING split (21 sequences; YOLOv8n / RT-DETR-L native caches built in C1; ByteTrack host) — added in C1 | KITTI tracking testing split (no public GT; never used) |

## Matrix
| dataset / system | GT available? | images available? | cached detections? | motion cues? | tracker runnable? | exact evaluation possible? | diagnostic-only? | dev / reserved | source / provenance | blocker | next action |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT17 val-half · BoostTrack (post-proc. + GBI) | yes (BoostTrack repo `results/gt/MOT17-val`, verified identical to the half split) | no (motchallenge 403) | yes (BoostTrack published YOLOX-X dets, release tar) | host: ECC cache complete (7/7 seqs, frames−1 keys); V7 image cue: no | yes, pixel-free (only frame SHAPE read, from `val_half.json`) | **yes for BASELINE/NATIVE** (online + GBI byte-identical to the Mac reference tracks; `_post` identical as a row set, line order differs) | V7 arms run with motion=None (rule inactive). Mac evidence E10: `V7c@motion=0` = V7c on BoostTrack, so this matches the original within that evidence | development | release `v7-dev-assets-1` + BoostTrack @fb5bfc3 | none for this configuration | done: NATIVE, V6EMU, V7c, V7d, cold=none, pool=raw |
| MOT17 val-half · SparseTrack | yes | no | yes (published dets) | GMC needs pixels (not cached) | no (GMC reads images) | **no** | a GMC-off run would change the published tracker → not run | development | release + SparseTrack @499844f | MOT17 frames | none until frames are reachable |
| MOT17 val-half · ByteTrack official MOT17 setting (assoc 0.6, birth 0.7, match 0.8, writer filter) | yes | not needed | yes (two floors: 0.01 SparseTrack stream, 0.1 BoostTrack stream; same YOLOX-X) | V7 image cue: no | yes | **yes** (IoU+Kalman only; BASELINE MOTA 77.6 / IDF1 79.5 vs published ByteTrack val-half 76.6 / 79.3 with its own detector run) | V7 motion rule inactive | development (new) | release + ultralytics 8.3.200 BYTETracker | none | done: 9 systems × 2 floors |
| MOT17 val-half · ByteTrack ultralytics default (0.25/0.25/0.1/0.8) | yes | not needed | yes (two floors) | no | yes | **yes** | motion rule inactive | development (new) | as above | none | done: 9 systems × 2 floors |
| MOT17 val-half · OC-SORT official MOT17 args | yes | not needed | yes (two floors) | no | yes | **yes** (BASELINE 66.44 / 74.67 / 78.05 vs paper val-half 66.5 / 74.9 / 77.7) | motion rule inactive | development (new) | noahcao/OC_SORT @ 8462e7e | none | running: 9 systems × 2 floors |
| KITTI tracking training · ByteTrack (ultralytics default) · YOLOv8n / RT-DETR-L | yes (official `data_tracking_label_2.zip`, sha256 6ad1fa01…) | yes (official `data_tracking_image_2.zip`, training members read by HTTP range from the official S3 bucket; zip CRC-checked per member) | built in C1 with the ORIGINAL native-cache recipe (`tools/cache_detections.py --resolutions 736`, YOLO NMS 0.7, RT-DETR no NMS, floor 0.01) | yes (`tools/visual_cues.py`, same cue) | yes | **yes** (TrackEval Kitti2DBox, official KITTI HOTA/CLEAR/Identity; classes Car + Pedestrian; detector task classes COCO person→Pedestrian, car→Car, declared before any run) | — | development (new) | s3.eu-central-1.amazonaws.com/avg-kitti (official KITTI distribution) | CPU time for RT-DETR-L | in progress (image extraction → caches → runs) |
| VisDrone val-7 / dev-40 · ByteTrack | no (annotations only in the Google-Drive zips) | no | yes (release caches) | yes (cached cues) | yes | **no** (no GT) | label-free only (E12-LF, STRESS-LF) | development | release | Google Drive 403 | label-free diagnostics done; labelled cells wait for GT |
| VisDrone · BoT-SORT (E11) | no | no | yes | yes | no (BoT-SORT CMC reads frames) | no | — | development | — | Google Drive 403 | E11 moved to KITTI (real frames available): ByteTrack vs BoT-SORT with CMC on the same streams |
| VisDrone test-dev · Faster R-CNN | no | no | yes | yes | yes | no | label-free only | development | release | Google Drive 403 | — |
| UAVDT | no | no | yes (release cache) | yes | yes | no | label-free only | development | release | Google Drive 403 | — |
| VisDrone confirmation-16 | — | — | — | — | — | — | — | **RESERVED** | — | reserved until the V7 freeze | untouched |
| TOPICTrack | — | — | — | — | — | — | — | **RESERVED** | — | reserved until the V7 freeze | untouched (not cloned) |

## Alternative-source search (recorded once)
Reachable: GitHub (code, releases), PyPI, the official KITTI S3 bucket.
Denied (403, not retried): motchallenge.net, drive.google.com,
drive.usercontent.google.com, huggingface.co, zenodo.org, kaggle.com,
dl.fbaipublicfiles.com, download.pytorch.org, download.openmmlab.com,
opendatalab.com, cocodataset.org, the ETH / Oxford / UCF / DETRAC
institutional hosts, dancetrack.github.io, aiskyeye.com.
Consequences: DanceTrack, SportsMOT, MOT20 frames, UA-DETRAC, BDD100K-MOT
are unreachable. KITTI tracking is the one reachable official MOT benchmark
with public training GT; it adds a new regime (driving, moving camera,
vehicles + pedestrians, KITTI occlusion/truncation, DontCare regions).
The Faster R-CNN checkpoint (download.pytorch.org) is unreachable, so
KITTI uses YOLOv8n + RT-DETR-L only (checkpoints from the ultralytics GitHub
release, sha256 checked against ASSET_MANIFEST.json).
