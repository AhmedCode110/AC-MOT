# V7 EXTERNAL SELECTION (predeclaration — DRAFT until the V7 freeze commit)

Purpose: after the V7 freeze, the EXACT frozen AC-MOT policy is added,
without any retuning, to recent strong published MOT systems reproduced as
faithfully as the accessible artefacts allow. For every system the table
reports: paper-reported result, our faithful reproduction, the reproduction +
frozen AC-MOT, Δ = (reproduction + AC-MOT) − reproduction, per-sequence
results, 10,000-sample paired bootstrap (seed 42) with 95% CIs, sequences
improved/degraded, failure cases.

Selection rules (fixed before any AC-MOT run on these systems):
- 2025/2026 publication (Q1 journal preferred, then Q2 / top venue);
- official code; official weights OR released tracker-input detections;
- reproducible protocol with reachable artefacts from this environment
  (GitHub, PyPI, official KITTI S3; motchallenge.net / Google Drive / arXiv /
  Hugging Face / Zenodo are unreachable here);
- AC-MOT inserted between the published detector output and the published
  tracker input only; the published tracker, detector, thresholds, evaluator
  and split are unchanged;
- never selected because AC-MOT helps it; all predeclared outcomes reported;
- NOT any development system (SparseTrack, BoostTrack, ByteTrack, OC-SORT,
  BoT-SORT, KITTI training, MOT17 val-half with those hosts).

## Candidate status (recorded once)
| System | Venue | Code | Artefacts reachable? | Status |
|---|---|---|---|---|
| PD-SORT (Wang et al.) | IEEE Trans. Consumer Electronics, 2025 | github.com/Wangyc2000/PD_SORT @ af21db6 | yes: tracker-input detections for MOT17 val-half (`res_mot/MOT17-val/yolox_x_ablation_results/*/MOT17-*_detections.txt`), CMC files (`cache/cmc_files`), the authors' own MOT17-val outputs and TrackEval summaries | CANDIDATE (baseline reproduction after the freeze; the final configuration is identified from the repository's code + released outputs because the paper PDF host arxiv.org is unreachable) |
| TOPICTrack (Cao et al.) | IEEE TIP, 2025 | github.com/holmescao/TOPICTrack @ e7b260f | needs YOLOX + FastReID on frames; MOT17/DanceTrack frames unreachable | RESERVED (kept untouched); infeasible in this environment unless frames become reachable |
| C-TWiX (Miah et al.) | Pattern Recognition, 2025 | github.com/Guepardow/TWiX @ 3cff9cc | detections + weights only on mehdimiah.com (403) | EXCLUDED (inaccessible artefacts) |
| TrackTrack (Shim et al.) | CVPR 2025 | github.com/kamkyu94/TrackTrack | needs detector + FastReID on frames (unreachable) | EXCLUDED (inaccessible artefacts) |
| CAMELTrack | 2025 | github.com/TrackingLaboratory/CAMELTrack | needs frames / appearance models | EXCLUDED (inaccessible artefacts) |

Search continues for 1–3 further 2025/2026 systems satisfying the rules; the
list is closed in the freeze commit.
