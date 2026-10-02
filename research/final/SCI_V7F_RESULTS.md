# SCI + V7f and General AC-MOT G1 — results

Development split only (VisDrone2019-MOT-val, 7 sequences, row P6); no protected split was opened.
Full record: `SCI_V7F_EXPERIMENT_LEDGER.md`; generated tables: `SCI_V7F_TABLES.md`; failures:
`SCI_V7F_FAILURES.md`; runtime: `SCI_V7F_REALTIME.md`.

## Architecture and roles
frame → compute layer (generic request LOW / MEDIUM / HIGH) → detector adapter (native setting) →
frozen detector → canonical detections → frozen V7f → tracker adapter / host contract → frozen tracker.

| module | file | role |
|---|---|---|
| compute layer | `acmot_sci.py` (historical SCI), `acmot_sci_v7.py` (`LevelSource`) | chooses a generic compute level; never sees a score or a threshold |
| detector adapter | `adapters/detectors/compute_profile.py`, `configs/sci_v7_profiles.json` | maps the level to a native setting the detector supports; refuses unsupported ones |
| score layer | `acmot_v7.py` (frozen V7f, lock `research/V7_POLICY_LOCK.json`) | score bands, admission, host thresholds, clean/noisy regime, rescue, host preservation |
| host contract | `tools/sci_v7/hosts.py` | ByteTrack, BoT-SORT, OC-SORT behind one interface |

No detector, tracker or dataset name and no native setting occurs in the generic files (unit test).
Causality: scene decision for t from frames < t and the image of t, then detection, then V7f (state of
frames < t), then the tracker, then history for t+1 (trace test and future-perturbation test).

## Candidates
- **C1** (5a8502f): historical SCI rule (resolution part of `core.Controller`, copied, object cues from the
  previous frame's tracks) + V7f. Pre-registered; not adopted.
- **G1** (tag `general-acmot-g1-freeze` → 751c602, lock `research/GENERAL_ACMOT_G1_LOCK.json`): constant
  compute request (no scene policy passed the pre-registered cue rule) + V7f. Behaviourally V7f at 736 px.

## Four-way ablation (val-7, ByteTrack, internal protocol, pooled over 14 detector × sequence cells)
| comparison | ΔHOTA [95% CI] | ΔIDF1 | ΔMOTA | compute |
|---|---|---|---|---|
| V7f only − Native | +2.98 [+1.76, +4.47] | +4.54 | +14.17 | 1.000 vs 1.000 |
| SCI only − Native | −0.22 [−0.45, −0.01] | −0.27 | −0.31 | 0.949 / 1.026 vs 1.000 |
| SCI + V7f − V7f only | −0.17 [−0.56, +0.21] | −0.19 | −0.18 | 0.979 / 1.012 vs 1.000 |
| SCI + V7f − SCI only | +3.03 [+1.73, +4.62] | +4.62 | +14.30 | equal |
| SCI + V7f − shuffled schedule (3 seeds) | −0.03 / +0.04 / +0.05 (all CIs contain 0) | | | equal |
Official protocol: SCI + V7f − V7f only HOTA −0.26 [−0.59, +0.04], MOTA −0.77 [−1.51, −0.20].
Per-detector numbers, all metrics, DetA/AssA-free runs and per-sequence values: `SCI_V7F_TABLES.md`.

## Matched compute and headroom
- Static points (V7f, HOTA): YOLOv8n 27.9 → 38.0 and RT-DETR-L 39.0 → 41.9 from 512 to 960 px.
- GT oracle at the compute of fixed 736 px: +0.47 [+0.005, +0.97] (640/736/832) and +0.60 [−0.12, +1.56]
  (nine resolutions) HOTA; at the SCI's own level counts: −0.01.
- The benefit of more compute per segment is reproducible within a detector (ρ ≈ +0.55 across score
  layers) and not across detectors (ρ ≈ +0.10 / −0.15).
- 13 causal cues (image, canonical detections, tracks, probe): none passes the pre-registered selection rule.

## G1 across detectors and hosts (G1 − host alone, ΔHOTA [95% CI], internal / official)
| detector \ host | ByteTrack | BoT-SORT | OC-SORT |
|---|---|---|---|
| YOLOv8n | +2.17 [+1.20, +3.58] / — | +0.61 [−0.05, +1.54] / +0.34 [−0.28, +1.15] | +14.86 [+12.32, +19.50] / +12.09 [+10.94, +14.23] |
| RT-DETR-L | +4.30 [+2.25, +6.42] / — | +2.15 [+0.16, +4.50] / +1.31 [−0.50, +3.55] | +6.76 [+2.88, +11.82] / +3.75 [+0.38, +8.42] |
| RetinaNet (unseen, locked) | +4.01 [+2.25, +5.46] / +4.84 [+3.37, +6.11] | +2.94 [+1.28, +4.54] / +4.22 [+2.98, +5.30] | not run |
Pooled over the 14 development cells with ByteTrack, official protocol: +2.29 [+1.19, +3.59].
Faster R-CNN is not listed as unseen: it was a V7 development detector.

## Claims
| claim | status | evidence |
|---|---|---|
| The scene layer and the score layer can be separated and composed causally without changing V7f | supported | unit tests; V7f's gain is unchanged under resolution switching (D − B ≈ C − A) |
| V7f keeps its gain at every compute level tested (512–960 px) | supported | +1.8 to +2.3 (YOLOv8n), +3.9 to +4.4 (RT-DETR-L) HOTA |
| Historical SCI adds accuracy or saves compute on top of V7f | not supported | four-way ablation; shuffled controls; official MOTA −0.77 |
| A detector-independent scene index can target where compute pays off | not supported on this data | headroom ≤ ~0.6 HOTA; benefit detector-specific; no cue passes |
| G1 improves the host on an unseen detector without recalibration | supported (one unseen detector, development sequences) | RetinaNet, locked before evaluation |
| G1 preserves already well-matched hosts | partially supported | YOLOv8n + BoT-SORT: no measurable change; OC-SORT official protocol: 4 new catastrophic sequences |
| Works with any detector / any tracker | not claimed | tested: YOLOv8n, RT-DETR-L, RetinaNet × ByteTrack, BoT-SORT, OC-SORT |
| Real-time on a T4 | not measured here | CPU only; `notebooks/G1_T4_runtime.ipynb` |

## Reproduction
```
git checkout general-acmot-g1-freeze          # 751c602
# development caches: release v7-dev-assets-1 (native) and sci-v7f-sweep-1 (512-960 px)
export ACMOT_VISDRONE_VAL=<VisDrone2019-MOT-val> V7_SPLIT=val7
python tools/sci_v7/dev.py run NATIVE+MEDIUM V7f+MEDIUM       # V7f+MEDIUM = G1
python tools/sci_v7/dev.py boot NATIVE+MEDIUM V7f+MEDIUM
```
Cloud: `.github/workflows/sci_v7_val.yml` (immutable candidate worktree), `sci_unseen.yml` (RetinaNet,
BoT-SORT, runtime). Hashes: `research/GENERAL_ACMOT_G1_LOCK.json`, `research/TRANSFER_LOCK_RETINANET_G1.json`.
