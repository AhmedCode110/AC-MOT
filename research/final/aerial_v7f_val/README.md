# Frozen V7f on VisDrone2019-MOT-val (development split only)

Purpose: score the frozen V7f layer (policy lock `research/V7_POLICY_LOCK.json`, freeze commit 488df9a) against the
host alone (NATIVE) on VisDrone val-7, with cached YOLOv8n and RT-DETR-L detections and the ByteTrack host.
Status of the split: development (row P6 of `research/context/PROTECTED_EVALUATIONS.md`), so any number is
development evidence, not held-out. Confirmation-16 (P1) and test-dev (P5) are not run by this script, which
refuses any other split.

Not a comparison with the original AC-MOT of Paper 1: the host settings and detector operating points differ.

## Run record
| Run | Date (UTC) | Where | Result |
|---|---|---|---|
| 36910783384 | 2026-10-01 | GitHub Actions | policy lock 10/10 matched; official val zip not downloadable from the public Google Drive id (download quota, 4 attempts over 20 min); no metric computed |
| 36994862846 | 2026-10-02 | GitHub Actions | zip downloaded (sha256 e5357199…5705); failed in `dev.py run`: the macOS-made cache tar holds AppleDouble `._*` entries that GNU tar extracted as fake `.npz` files (pickle ValueError); fixed in 1b5bc79; no metric computed |
| 36995662755 | 2026-10-02 | GitHub Actions (AMD EPYC, 4 vCPU, no GPU), commit 1b5bc79 | OK: policy lock 10/10; results in this folder |

Result of run 36995662755 (internal protocol, paired bootstrap over the 7 sequences, 10,000 resamples, seed 42;
`bootstrap_val7.txt`): V7f − NATIVE HOTA +2.17 [+1.20, +3.58] and IDF1 +3.90 [+2.17, +6.40] with YOLOv8n,
HOTA +4.30 [+2.25, +6.42], IDF1 +6.13 [+2.81, +9.54] and MOTA +27.3 [+14.7, +40.4] with RT-DETR-L; HOTA higher
on 7/7 sequences for both detectors; catastrophic sequences (MOTA < 0) 5 → 0. Official-compatible protocol
(`official_val7.txt`): HOTA 28.9 → 30.3 (YOLOv8n) and 31.2 → 34.5 (RT-DETR-L).

Earlier, run 36617713245 of the broader workflow `v7_aerial.yml` failed at the same download for the same reason.

## Run it with your own copy of the official zip (Colab or any machine)
```
git clone -b universal-adapters-v1-y0zkeh https://github.com/AhmedCode110/AC-MOT.git && cd AC-MOT
export VISDRONE_VAL_ZIP=/path/to/VisDrone2019-MOT-val.zip
bash tools/v7/ci/aerial_v7f_val.sh
```
Results are written to this folder (`report_val7.txt`, `official_val7.txt`, `per_sequence_val7.txt`,
`bootstrap_val7.json`, `summary_val7.json`, `STATUS.txt`).
