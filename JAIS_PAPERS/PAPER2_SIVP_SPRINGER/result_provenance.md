# Result provenance — Paper 2, Signal, Image and Video Processing version (same evidence as the AIAA-format version)

All numbers enter the manuscript through `scripts/evidence.py` (tables: `scripts/make_tables.py`; figures:
`scripts/make_figures.py`). Frozen policy: `configs/universal_acmot_policy_v7.json`, tag `universal-acmot-v7-freeze`
→ 488df9a, lock `research/V7_POLICY_LOCK.json`. Statistics: paired sequence bootstrap, 10,000 resamples, seed 42.

| Manuscript item | Source |
|---|---|
| MOT17 development hosts (Table 3, Figs. 3–4, Tracker Transfer): SparseTrack, BoostTrack, ByteTrack, OC-SORT | `research/final/V7_STATISTICS.md`, `V7_MAIN_RESULTS.md` Tables 1–2 (`V7_MAIN_RESULTS.json`, `V7_DEV_RESULTS.json`); SparseTrack run: `research/final/sparsetrack_v7f/` |
| Paper reference numbers of these hosts | `V7_MAIN_RESULTS.md` (sources listed under Table 1) |
| KITTI detector transfer (Table 4, Fig. 3, Detector Transfer) | `V7_STATISTICS.md` KITTI table |
| Confidence-shift cells (Fig. 5, Detector Transfer, Robustness) incl. worst cell −0.14 | `V7_EXPERIMENT_LEDGER.md` STRESS-L; `V7_MAIN_RESULTS.md` Table 3; `V7_STATISTICS.md` |
| PD-SORT and Hybrid-SORT (Table 5, Figs. 3–4, External: predeclared) | `V7_EXTERNAL_TRANSFER.md` S1/S2 (`V7_EXTERNAL_RESULTS.json`) |
| C-TWiX, TrackTrack, TOPICTrack (Table 5, Figs. 3 and 6, External, Cross-Dataset, Failure Cases) | `research/final/V7_RECENT_EXTERNAL_RESULTS.json` (read directly), records in `research/final/recent/{ctwix,tracktrack,topictrack}/`; classification: `V7_RECENT_EXTERNAL_SYSTEMS.md`, `V7_RECENT_EXTERNAL_FAILURES.md`; protocol: `V7_RECENT_EXTERNAL_PROTOCOL.md` |
| Regime shares (Fig. 6a, Cross-Dataset: 99.7 %, 99.9 %, 2.8 %) and KITTI sequence 0014 (45 of 106 noisy frames) | `research/final/recent/ctwix/audit_*.json`, `recent/tracktrack/audit_DanceTrack.json` |
| Prior design on SparseTrack/BoostTrack (Fig. 2, Development and Freeze Protocol): −4.15 [−5.54, −1.66], −5.89 [−7.76, −2.87], FP/FN changes | `research/final/EXTERNAL_PAPER_TRANSFER.md` §12, §14 |
| Ablation steps and rejected variants (Table 6, Ablation) | `research/final/V7_ABLATION.md`; ledger E15–E20 |
| Interpretability-check diagnosis (t1 ≈ 0.6, t2 ≈ 0.93) | `research/final/V7_FAILURE_EVOLUTION.md` item 4 |
| YOLOv8n regression (ρ ≈ 0.45, IDS −271 / −227) | `V7_FAILURE_EVOLUTION.md` item 6; `V7_STATISTICS.md` |
| Floor sensitivity 41–45 % (earlier variant, aerial) | `V7_EXPERIMENT_LEDGER.md` STRESS-LF |
| Runtime (Table 7, Fig. 7) | `research/final/V7_REALTIME.md`, `V7_REALTIME_yolov8n.json` |
| 61 tests | `tests/test_v7_adaptive_layer.py`, `tests/test_v7_bootstrap.py` (61 passed, 2026-09-29) |
| Method description (Architecture to Complexity sections) | `acmot_v7.py` (`V7Layer.step`), `research/final/V7_METHOD.md`, configuration file |
| Host contracts (Table 1) | adapters in `tools/v7/external/*.py`, `tools/v7/recent/*.py`; `V7_RECENT_EXTERNAL_PROTOCOL.md` §5 |

Not used: universal V1/V3/V4/V5-TF results; V6-TF results other than the SparseTrack/BoostTrack transfer; any legacy
scene-adaptive (Paper 1) result; the unfinished aerial V7f run (no result exists).
