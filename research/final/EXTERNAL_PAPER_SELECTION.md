# EXTERNAL PAPER SELECTION — candidate matrix (2026-09-28)

Goal: a strong, recent, independently published MOT method. Reproduce or
faithfully execute its final published method with official code and
weights. Attach the SAME frozen V6-TF (tag `universal-acmot-v6-freeze`,
2cff95f) and compare under identical conditions.

Selection priority (owner):
1. scientific compatibility with AC-MOT
2. venue strength
3. recency (2025–2026 headline; 2024 supporting)
4. reproducibility
5. official weights
6. architecture
7. dataset

Easiness to outperform is NOT a criterion.

## Ranking sources
SCImago (scimagojr.com), ooir.org and Resurchify are behind bot protection,
and bypassing it is not permitted. Rankings below are therefore quoted from
secondary pages that reproduce SCImago/JCR figures, with the source named.
Treat them as "secondary-source, not read directly from SCImago/Clarivate".

| Venue | Type | Ranking (source, year) |
|---|---|---|
| IEEE Trans. Circuits Syst. Video Technol. (TCSVT) | journal | SJR Q1; exaly.com 2-year impact 7.46 (2025); JCR Q1 per askbisht.com/editage (2024) |
| IEEE Trans. Image Processing (TIP) | journal | SJR Q1, SJR 2.502; IF 13.7 (2024); h-index 363 (researcher.life / askbisht.com) |
| ISPRS J. Photogramm. Remote Sens. | journal | SJR Q1 (secondary sources; well-established Q1) |
| Machine Vision and Applications (MVA) | journal | SJR Q2, 0.530 (2024); IF 3.0 (2024) (search-result summary of scimagojr/resurchify pages) |
| Filomat | journal | SJR Q2, 0.467 (Mathematics); IF 1.0, JCR Q2 (researcher.life / askbisht.com) |
| Sensors (MDPI) | journal | SJR Q2 (secondary) |
| AAAI / CVPR | conference | not journal-ranked (CORE A*) |

## Candidate matrix
Hardware available: Mac arm64 (Apple MPS/CPU), no CUDA.
Colab/T4 is the owner's resource; this autonomous session has no access to it.

| # | Paper (venue, year) | Task / data | Detector → tracker | Code | Weights | Runs here? | V6-TF insertion point | Decision |
|---|---|---|---|---|---|---|---|---|
| 1 | **SparseTrack**: Liu, Wang, Wang, Liu, Bai, "SparseTrack: Multi-Object Tracking by Performing Scene Decomposition based on Pseudo-Depth" (**IEEE TCSVT 2025**, Q1) | general MOT: MOT17/20, DanceTrack | YOLOX-X (ByteTrack weights) → pseudo-depth DCM + GMC, two-stage (high / low) association | github.com/hustvl/SparseTrack (@499844f) | ByteTrack `bytetrack_ablation.pth.tar` (sha256 26cb8d28…) | **yes**: detectron2 built from source; GMC (OpenCV videostab C++) compiled verbatim behind a C ABI; MPS | between detector output and the DCM association; generic controls → `track_thresh`, `det_thresh`, low 0.1 native, `match_thresh` | **HEADLINE** (owner-confirmed): Q1 2025, official code and checkpoint, separable, ByteTrack-lineage two-stage association where candidate control is meaningful; pinned deterministic runtime already in the owner's Drive |
| 2 | TOPICTrack: Cao et al., "TOPIC: A Parallel Association Paradigm for MOT under Complex Motions and Diverse Scenes" (**IEEE TIP 2025**, Q1) | MOT17/20, DanceTrack, GMOT-40, BEE24 | YOLOX-X (weights identical to ByteTrack ablation, sha256 26cb8d28…) → parallel motion/appearance association (OC-SORT + FastReID SBS-S50) | github.com/holmescao/TOPICTrack | Google Drive (downloaded: topictrack_ablation, mot17_sbs_S50) | likely (CUDA calls patchable; FastReID on MPS) | before OC-SORT/embedding association | secondary headline candidate; deferred by owner priority (SparseTrack first, BoostTrack supporting) |
| 3 | BoostTrack: Stanojević & Todorović (**Machine Vision and Applications 2024**, Q2); BoostTrack++ (Filomat 2025, Q2) | MOT17/20 | YOLOX-X (same checkpoint) → single-stage confidence-boosted association (DLO/DUO boosting), ECC | github.com/vukasin-stanojevic/BoostTrack (@fb5bfc3) | Google Drive | **yes** (reproduced) | before BoostTrack's confidence boost / `det_thresh` filter | **SUPPORTING** (owner): reproduced within ±0.12 of the authors' re-reported online numbers; frozen V6-TF result obtained (negative) |
| 4 | MOSAIC-Tracker (ISPRS JPRS 2025, Q1) | VisDrone, UAVDT (aerial) | YOLOv8 → association | github.com/aJanm/MOSAIC-Tracker | **none** (training scripts only) | only after training a VisDrone detector, which would not be the published checkpoint | — | rejected: no official weights |
| 5 | MM-Tracker (AAAI 2025) | VisDrone, UAVDT | YOLOX → Motion-Mamba | github.com/YaoMufeng/MMTracker | Baidu Net Disk only | **no**: `mamba_ssm` is CUDA-only; Baidu needs an account | — | rejected: hardware and weight access (conference venue) |
| 6 | OATrack (Sensors 26(16):5222, 2026, Q2) | VisDrone val, UAVDT | YOLO11m cache → progressive association | not public | no | no | — | rejected: no code |
| 7 | DroneMOT (ICRA 2024) | VisDrone, UAVDT | FairMOT JDE | public | — | no (DCNv2 CUDA build) | none (joint detection) | rejected |
| 8 | UAVMOT (CVPR 2022) | VisDrone, UAVDT | MCMOT JDE | public | not listed | no | none | rejected: old, JDE |
| 9 | LTTrack (TCSVT 2024) | MOT17/20, DanceTrack | TBD | github.com/linjiaping1/LTTrack | not checked | not attempted | — | not pursued (2024; TCSVT 2025 candidate available) |
| 10 | MOTIP (CVPR 2025), DiffMOT (CVPR 2024), Hybrid-SORT / UCMCTrack (AAAI 2024) | general MOT | end-to-end ID prediction / diffusion motion / weak cues / CMC | public | public | not attempted | varies (MOTIP not detector-separable) | not pursued: conference venues; journal Q1 candidates available |

## Decision
- **Headline: SparseTrack (IEEE TCSVT 2025, Q1).** Its final released method
  (pseudo-depth DCM + GMC, official `mot17_ab_track_cfg.py`) is executed with
  the official checkpoint. It is compared with the identical run + frozen
  V6-TF.
- **Supporting: BoostTrack (MVA 2024).**
- **Optional: TOPICTrack (IEEE TIP 2025)** if time permits.

All three candidates share the same YOLOX-X detector checkpoint (sha256
26cb8d28…). They differ in association design:
- SparseTrack: two-stage IoU with depth cascade;
- BoostTrack: single-stage confidence boosting;
- TOPICTrack: parallel motion/appearance.
