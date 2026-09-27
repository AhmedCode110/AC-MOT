# PROTECTED EVALUATIONS — consult BEFORE evaluating anything

"Quality metric" = any of HOTA, IDF1, MOTA, IDS, FP, FN, precision, recall,
DetA/AssA, or anything computed with GT. "Freeze" = tag
universal-acmot-v5tf-freeze exists (check `git tag`). Until it exists, every
PROTECTED status below holds.

| # | Target | Current status | Allowed before freeze | Forbidden before freeze | After freeze |
|---|---|---|---|---|---|
| P1 | Split: VisDrone2019-MOT-train confirmation-16 | PROTECTED, untouched | caching detections/cues | any quality metric; replaying any system with GT; inspecting GT; design choices based on them | evaluate ONCE: V5-TF vs V4 at 736, Amendment-5a rule, paired bootstrap; no redesign after |
| P2 | Detector: Faster R-CNN ResNet50-FPN v2 (unseen detector) | PROTECTED (no quality metric ever computed) | weights download, adapter, caching, detection-level fidelity (tools/fidelity_gate.py compare_det), T4 timing | tracking metrics; `tools/audit/nms_audit_frcnn.py` (computes HOTA/IDF1 — untracked, never run); any tuning from its outputs | evaluate under TRANSFER_LOCK_FASTERRCNN (V4) and a V5-TF transfer lock written before running; no retuning |
| P3 | Tracker: BoT-SORT (tracker transfer for V5-TF) | PROTECTED for V5-TF (V3/V4 BoT-SORT results exist: val transfer, E31) | timing; tracker adapter code | V5-TF quality metrics with BoT-SORT; any V5-TF design decision informed by BoT-SORT | evaluate once, no retuning |
| P4 | Dataset: UAVDT test (unseen dataset) | PROTECTED (no Universal quality metric) | caching (`outputs/det_cache_uavdt*`), view building | any quality metric; tuning | evaluate under TRANSFER_LOCK_UAVDT(_FRCNN) and a V5-TF lock; no retuning |
| P5 | Split: VisDrone2019-MOT-test-dev | USED ONCE (V4, E31) | nothing for V5-TF design | any V5-TF design/tuning use | post-hoc only, always labelled "post-hoc" |
| P6 | Split: VisDrone2019-MOT-val | not clean (development) | — for V5-TF design (Amendment 5d: secondary only) | using val to choose V5-TF rules | secondary check after freeze |
| P7 | Amendment-5f T4 fidelity gate | PENDING | running the gate (detection-level + declared replays) | relaxing thresholds after seeing results; tuning V5-TF to hardware | FAIL → rebuild dev caches on T4 |
| P8 | Official timing | none yet | T4 benchmark (timing only) | reporting Mac/MPS timing as results | report scene analyzer, normaliser, AC decision, detector, tracker, total, P95, FPS, GPU memory, AC overhead % |
| P9 | Any future unseen detector/dataset | PROTECTED by default | caching | quality metrics before a written lock | lock first, then evaluate once |

Before running ANY evaluation command, an agent must state which row applies
and confirm the action is in "Allowed". If none applies: stop and ask.
