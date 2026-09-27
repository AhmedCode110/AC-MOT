# RESULTS — canonical index

Rule: the machine-readable file is authoritative; this index only points to
it. Never copy a number here without checking the file. Conflicts are listed
explicitly, never silently reconciled. `outputs/` is git-ignored (local, Mac):
paths below are relative to the repo root on the development Mac.
Evaluator for every row: internal class-agnostic protocol, HOTA TrackEval
12c8791, others motmetrics 1.4.0 (NOT official VisDrone).

## Canonical result sets
| Set | Commit | Status | Authoritative file(s) | Human summary |
|---|---|---|---|---|
| V4 held-out VisDrone2019-MOT-test-dev (E31) | a55b138 code / 9d9ebeb results | HELD-OUT (evaluated once) | `outputs/heldout_v4/pooled_metrics.json`, `outputs/heldout_v4/bootstrap.json` | `research/RESULTS_TESTDEV_V4.md` |
| V4 nested LOSO selection (E29) | fc003bf | DEVELOPMENT (val) | `outputs/opt_v4/v4_selection.json`, `outputs/opt_v4/grid.json` | research/EXPERIMENT_REGISTRY.md E29 |
| V3 selection (E22) | c1e799d | DEVELOPMENT (val), SUPERSEDED | `outputs/opt_v3/selection_report.json`, `response_surface.csv` | E22 |
| Stress / invariance (E20, E23) | c1e799d | DEVELOPMENT (val) | `outputs/stress/` | E20, E23 |
| BoT-SORT tracker transfer on val (V3) | pre-fc003bf | DEVELOPMENT (val) | `outputs/tracker_transfer_val/` | — (NEEDS VERIFICATION: summary not in registry) |
| V5 S1 headroom (E32) | 62111e8 | DEVELOPMENT (val) | `outputs/v5/s1/headroom.json`, `outputs/v5/s1b/headroom.json` | context E32 |
| V5 S2 cue utility (E33) | 62111e8 | DEVELOPMENT (val) | `outputs/v5/s2_cue_utility.json` | context E33 |
| V5 S3 attempts (E34, E35) | 62111e8 | DEVELOPMENT (val), FAILED | `outputs/v5/s3/full/`, `outputs/v5/s3b/full/` | Amendments 5b/5c |
| V5-TF dev validation (E36) | 71faf44 + Amendment 7 | DEVELOPMENT-40, COMPLETE | `outputs/v5tf_dev/` PKLs, `outputs/v5tf_dev/family_choice.json` | F3 selected; F5 did not beat F3 or F5R; D-022 |
| V5-TF constant audit (E39) | Amendment 7 | DEVELOPMENT-40, COMPLETE | `outputs/v5tf_dev/constant_audit.json` | bins/window sensitive E; history/warm-up insensitive A; D-023 |
| V5-TF exact-Otsu family (E41) | 240eba9 + Amendment 8 | DEVELOPMENT-40, COMPLETE | `outputs/v5tf_dev/E41/` | 80/80 PKLs; 15 catastrophic cells; worst-detector relative gain vs V4 +0.69%; no protected data; D-025 |
| V5-TF live/replay parity (E40) | fd44a11 / 50f2a6f | DEVELOPMENT FIDELITY, PASS | `outputs/v5tf_dev/live_replay_parity_v2.json` | exact tracks + controls, 80/80 frames; v1 preserved as harness failure |
| Faster R-CNN, UAVDT, BoT-SORT (V5-TF) | — | PROTECTED | none may exist before V5-TF freeze | — |
| Official T4 timing | — | none official yet | will be `t4_benchmark*.json` from a T4 run | — |

## Headline held-out numbers (health-checked against pooled_metrics.json)
Tracker: ByteTrack. Dataset: VisDrone2019-MOT-test-dev (17 seq). Status HELD-OUT.
<!-- verify:outputs/heldout_v4/pooled_metrics.json -->
| tracker | system | detector | MOTA | HOTA | IDF1 |
|---|---|---|---:|---:|---:|
| bytetrack | V4_736 | yolov8 | 23.89 | 33.72 | 40.57 |
| bytetrack | V4_736 | rtdetr | 23.60 | 36.33 | 42.56 |
| bytetrack | V4_832 | yolov8 | 26.19 | 35.75 | 43.60 |
| bytetrack | V4_832 | rtdetr | 25.31 | 37.33 | 43.71 |
| bytetrack | shared_static_736 | yolov8 | 21.35 | 29.80 | 34.41 |
| bytetrack | shared_static_736 | rtdetr | 25.67 | 38.82 | 46.58 |
| bytetrack | default | yolov8 | 20.27 | 30.00 | 34.97 |
| bytetrack | default | rtdetr | 8.78 | 37.09 | 42.89 |
<!-- /verify -->
Reading (from RESULTS_TESTDEV_V4.md): YOLOv8n — V4 significantly better than
default and shared static at matched compute; RT-DETR-L — V4 removes the
default's collapse (6→0 catastrophic sequences) but is significantly below
shared static on HOTA (−2.49) and IDF1 (−4.02). Bootstrap: 10,000 resamples,
seed 42.

## Known conflicts / caveats
- E36 selected F3 over F5, but F3 has 17 catastrophic sequence-detector cells
  versus V4's 3 and worst-detector relative ½(HOTA+IDF1) gain −0.339%.
  This is development evidence, not protected confirmation; report honestly.
- E39's instruction to keep/report sensitive defaults does not override HARD
  C5's ban on category-E deployment constants. OTSU_BINS and Otsu window block
  the V5-TF lock/freeze until a predeclared training-free revision removes them.
- UAVDT "never evaluated by any system" (Amendment 5) vs legacy branch
  `origin/freeze/final-after-uavdt-2026-09-12` (legacy AC-MOT UAVDT run):
  both true only if "system" means Universal AC-MOT. See DATASETS_AND_SPLITS.md.
- E32 headroom: Amendment 5a quotes "0.4–2.3" per-sequence; `s1b/headroom.json`
  gives 0.44–2.25 (sensitivity/assoc/τ/resolution) — consistent after rounding;
  which of s1 vs s1b the amendment used: NEEDS VERIFICATION.
