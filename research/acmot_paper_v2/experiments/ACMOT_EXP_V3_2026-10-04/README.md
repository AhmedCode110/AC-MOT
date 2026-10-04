# ACMOT_EXP_V3_2026-10-04 — canonical experiment archive

Archive of the AC-MOT v3 experiment (YOLO11m + ByteTrack / OATrack, VisDrone2019-MOT, UAVDT frozen transfer).
Built by copying existing artifacts byte-for-byte. Nothing was rerun, retuned, recomputed or deleted.

**Status:** COMPLETE. The experiment ID stays `ACMOT_EXP_V3_2026-10-04` because the final experimental run was
initiated on 2026-10-04 (owner instruction), even though the UAVDT transfer evaluation and this final archival
freeze completed after midnight. See `EXPERIMENT_MANIFEST.json → timestamps.experiment_completed` for the actual
completion timestamp. Final tag: `acmot-v3-exp-2026-10-04-freeze`.

## Frozen controller (from `controller_search/FREEZE_MANIFEST_V3_ADAPTIVE.json`, freeze commit `309dca5`)

| level | resolution | NMS IoU | detector conf floor |
|---|---|---|---|
| LOW | 1536 | 0.70 | 0.10 |
| MEDIUM | 1280 | 0.45 | 0.40 |
| HIGH | 1088 | 0.60 | 0.40 |

`t_med = 0.29747709701135766`, `t_high = 0.5114928923560124`, cue subset `['crowd']`, SceneLayer otherwise frozen.
Baseline: `r1536_n70` (1536 px, NMS 0.70, no extra confidence filter). Detector sha256
`c6f98247e7c731a567aaf37ca7b808beff588b687623e028b25a2e1d846e96e7` (verified against the local file).

## Layout

| path | content |
|---|---|
| `EXPERIMENT_MANIFEST.json` | IDs, timestamps, splits, seeds, Kaggle kernel record, stage counts, result-table locations, known issues |
| `FREEZE_MANIFEST.json` | archive-level freeze: controller, baseline, detector, seeds, val reads, selection audit |
| `PAPER_READY_RESULTS.md` | every paper number with the exact source file and JSON key |
| `ARTIFACT_INDEX.csv` | one row per artifact (copied or hash-only external) |
| `GIT_STATE.txt`, `ENVIRONMENT.txt` | git provenance and recorded environment |
| `CONSISTENCY_REPORT.json` | results of `check_consistency.py` |
| `hashes/SHA256SUMS.txt` | sha256 of every file in this archive (`sha256sum -c` from this folder) |
| `artifacts/<repo path>` | verbatim copies (code, protocols, predeclarations, search outputs, results, reports, scripts) |
| `uavdt/` | Kaggle kernel provenance; cache pulled, frozen policy and baseline evaluated |
| `build_archive.py`, `write_manifests.py`, `check_consistency.py` | the scripts that built this archive |

External, hash-only (not in git, listed in `ARTIFACT_INDEX.csv` with `EXTERNAL:` paths): the detection caches
under `research/acmot_paper_v2/cache/`, the frozen checkpoint, and the UAVDT evaluation view (`outputs/uavdt_view`,
image folders are symlinks into Google Drive).

## Audit trail: paper result → source

| result | source file | key |
|---|---|---|
| Systems 1/2 (baseline) | `BASELINE_SYSTEMS_RESULT.json` | `YOLO11m+ByteTrack/OATrack.aggregate` |
| Systems 3/4 (AC-MOT v3), SECOND VAL READ | `SECOND_VAL_READ_V3_RESULT.json` | `AC-MOT-v3+YOLO11m+*.aggregate` |
| Deltas + 95% CI | `BOOTSTRAP_V3_RESULT.json` | `bytetrack_v3_vs_system1`, `oatrack_v3_vs_system2` |
| Action use / switching on val | `SECOND_VAL_READ_V3_RESULT.json` | `level_frac`, `switches_per_100_frames` |
| Calibration evidence | `controller_search/FREEZE_MANIFEST_V3_ADAPTIVE.json`, `V3_JOINT_SEARCH_RESULT.json` (trial 7) | `development_evidence` |
| Shuffled control | `controller_search/SHUFFLED_CONTROL_RESULT.json` | — |
| Runtime (T4) | `RUNTIME_BENCHMARK_RESULT.json` (baseline), `V3_RUNTIME_BENCHMARK_RESULT.json` (v3) | `results` |
| Static control (first val read) | `FINAL_ACMOT_SYSTEMS_RESULT.json`, `BOOTSTRAP_RESULT.json` | — |
| UAVDT transfer (frozen, no retuning) | `UAVDT_TRANSFER_RESULT.json` | `baseline_r1536_n70+*`, `v3_adaptive+*` |

All paths are under `research/acmot_paper_v2/` and copied under `artifacts/research/acmot_paper_v2/`.

## Disclosures (also in PAPER_READY_RESULTS.md)

1. VisDrone-val was read twice: static control (`bfa5caf`), then v3 (`0af0066`, labelled SECOND VAL READ).
2. The shuffled control (same action mix, random timing) scored at least as well as the causal schedule on calibration.
3. Selection-rule deviation: trial 7 is not the argmax of the predeclared rule (`FREEZE_MANIFEST.json → selection_audit`).
   By the implemented eligibility flag the argmax is trial 10; by the predeclared tuple rule it is trial 54.
4. Detector is third-party, trained on VisDrone2019-DET at 640 px.
5. UAVDT transfer (Table 6) is a single frozen-policy run per host, not a search; no bootstrap/CV applies. ByteTrack
   improves substantially and consistently with the VisDrone direction; OATrack's improvement is small and mixed
   (HOTA essentially flat, -0.09).
6. `FINAL_REPORT.md`'s §8 (runtime) and §11 (remaining work) were corrected at commit `87fd91a`, after the UAVDT
   transfer completed, to remove stale "not measured"/"blocked" language; archived as of that commit.

## Archive completion record

UAVDT cache generation (Kaggle kernel `ahmedgouda1111111/uavdt-v3-cache-gen`, verified exact match to the frozen
20-sequence/16,592-frame protocol) and the frozen no-retuning transfer evaluation (both trackers, both policies)
are complete and archived under `uavdt/` and `artifacts/research/acmot_paper_v2/UAVDT_TRANSFER_RESULT.json`. This
archive was rebuilt (`build_archive.py`, `write_manifests.py`, `check_consistency.py`), re-hashed
(`hashes/SHA256SUMS.txt`, 100% verified), and frozen at the final freeze commit under the annotated tag
`acmot-v3-exp-2026-10-04-freeze`. No scientific experiment was run during this final archival pass.
