# FROZEN VERSIONS

Historical meaning is never overwritten. Tag → commit mappings are verified
by tools/context_health_check.py (`git rev-list -n1 <tag>`).

| Name | Tag | Commit | Architecture | Purpose | What was frozen | Evaluation status | Superseded? |
|---|---|---|---|---|---|---|---|
| Legacy AC-MOT | v1.0.0-acmot-frozen | a6c1fa49fce1d402513c2df05b7d04b962a6e89e | handcrafted SCI + resolution/sensitivity controller, YOLOv8n + ByteTrack | original thesis method | SCI, controller, three-worker final test workflow | legacy 17-seq test-dev runs (exploratory); legacy UAVDT run on branch freeze/final-after-uavdt-2026-09-12 | yes (research line) |
| Version: V1 | universal-acmot-v1-freeze | e56c2f37793450aa3189a3734298f295194ffa77 | adapters + histogram normaliser + ratio gate ρ 0.6 + legacy SCI | first universal freeze | `configs/universal_acmot_policy.json` at that commit | val only; later run on test-dev as a baseline inside E31 | yes → V3 (Amendment 3) |
| Version: V3 | universal-acmot-v3-freeze | c1e799d3ba74d4c220b6be88f1659cdde1b20bf9 | ECDF + z-logit gate τ 0.75 + legacy SCI | calibration-invariant | policy + pipeline hashes in research/TESTDEV_LOCK.json (lock SUPERSEDED before any test-dev metric) | val; baseline row in E31 | yes → V4 (Amendment 4) |
| Version: V4 | universal-acmot-v4-freeze | fc003bf96621214dc09a0e55f720128093972927 | compute-budget only, ECDF + z-gate, global τ .75 / s .4 / assoc .10 / NMS .45 / tracker 45,.86 | latest frozen Universal system | hashes in research/TESTDEV_LOCK_V4.json | held-out test-dev once (E31); transfer locks FRCNN/UAVDT not yet evaluated | no — BASELINE/ABLATION ONLY, never the final system (Amendment 7, C0); contains category-E constants under Amendment 6 |
| Version: V5-TF | (planned) universal-acmot-v5tf-freeze | — | training-free online self-calibration + scene-state control | FINAL TARGET, current development | nothing frozen yet | none | — |

Never-to-be-created tag: `universal-acmot-v5-scene-generalized-freeze`
(the learned V5 final fit it would have tagged is forbidden by Amendment 6).

Reproducibility notes: policy/pipeline file hashes for V3/V4 are in the lock
JSONs; detector weights: yolov8n.pt sha256 f59b3d83…, rtdetr-l.pt sha256
6de60b10… (full values in TESTDEV_LOCK_V4.json). Evaluator: TrackEval
12c8791, motmetrics 1.4.0. Cache replay fidelity: E05, E30.
