# EXPERIMENT REGISTRY — index

Authoritative detail:
- E01–E31: `research/EXPERIMENT_REGISTRY.md` (do not duplicate; append there
  only for V4-era corrections).
- E32+: THIS file is authoritative (V5 / V5-TF era).
Evaluation protocol for all rows unless stated: internal class-agnostic
protocol (GT classes {1,4,5,6,9}, score==1, trunc<2, occ<2, IoU 0.5) —
NOT official VisDrone. HOTA = TrackEval 12c8791; others = motmetrics 1.4.0.

## E01–E31 by theme (summary; see authoritative file)
| IDs | Theme | Version | Outcome |
|---|---|---|---|
| E01–E03 | candidate explosion / density budgets | V1, V2b, V2c | FAIL (see FAILED_EXPERIMENTS.md) |
| E04–E05 | evaluator and cache-replay fidelity | infra | PASS (exact) |
| E06–E14 | reliability, persistence, continuity, closed-loop trust | V2d–V2f | mostly FAIL; leader-relative score identified |
| E15–E17 | leader-relative gate; signal audit | V2g/V1g | gate is the core fix; temporal persistence rejected |
| E18–E19 | V1 selection grid + baselines | V1 | V1 frozen; shared-static competitive at 832 |
| E20–E23 | score-calibration stress; V3 invariant design | V3 | V3 exactly Platt-invariant |
| E24–E26 | SCI / cue audits at matched compute | V3→V4 | legacy SCI ≤ random → removed |
| E27–E30 | NMS audit, sensitivity, V4 selection, checks | V4 | V4 frozen |
| E31 | held-out VisDrone test-dev (once) | V4 | RESULTS_TESTDEV_V4.md |

## E32+ (V5 / V5-TF)
| ID | Commit | Hypothesis | Config | Data | Result | Status |
|---|---|---|---|---|---|---|
| E32 | 62111e8 | S1: which control targets have adaptation headroom | fixed-value runs around V4 at 736 (`tools/v5_s1_headroom.py`) | VisDrone2019-MOT-val × {YOLOv8n, RT-DETR-L} | headroom for gate τ, association offset, resolution, sensitivity (per-sequence ≈0.4–2.3 MOTA pts); retention ≈0 (`outputs/v5/s1/headroom.json`, `outputs/v5/s1b/headroom.json`) | DEVELOPMENT (val) |
| E33 | 62111e8 | S2: which cues predict headroom | single-cue LOSO + permutation null (`tools/v5_s2_cues.py`) | val-7 × 2 | candidates det_gap, det_count, trk_survival, trk_match, img_motion, img_motion_resp; edges/brightness/blur negative (`outputs/v5/s2_cue_utility.json`) | DEVELOPMENT (val, diagnostic) |
| E34 | 62111e8 | S3 attempt 1: learned stump controller | cost FP+FN+IDSW, nested LOSO (`outputs/v5/s3/full/`) | val-7 × 2 | ½(HOTA+IDF1) vs V4: YOLO −2.42 [−4.86,−0.61], RT-DETR −2.64 [−8.30,+0.57] (Amendment 5b text) | FAILED |
| E35 | 62111e8 | S3 attempt 2: identity-aligned cost | (FP+FN)+(IDFP+IDFN) (`outputs/v5/s3b/full/`) | val-7 × 2 | YOLO −2.26 [−4.77,−0.38]; RT-DETR −0.60 [−2.15,+1.10] (Amendment 5c text) | FAILED |
| E36 | 71faf44 + Amendment 7 | V5-TF families F1, F2, F3, F5 (+R-res) and control F5R vs references (static, shared-static, V4) at budget 736; choice among F3/F5 | `tools/v5tf_dev.py run/report` | VisDrone2019-MOT-train development-40 × {YOLOv8n, RT-DETR-L}, ByteTrack | — | PENDING (waits for 736 + 640/832 caches) |
| E37 | — | Discovery: S1/S2/S3 on development-40 (research upper bound D, headroom, cue stability) | `tools/v5_train.py s1/s2/s3` (NOT `final`) | development-40 | — | PLANNED, optional |
| E39 | Amendment 7 | Constant audit on the chosen family (OTSU_BINS, Otsu window, RobustHistory window, warm-up; E28 criterion) | `tools/v5tf_dev.py sens` | development-40 × 2 detectors | — | PENDING (after E36) |
| E38 | feff14f | Amendment 5f T4 fidelity gate | `tools/fidelity_gate.py`, `notebooks/Colab_T4_gate_and_benchmark.ipynb` | 4 predeclared dev sequences | — | PENDING (needs T4 session) |

Numbers in E34/E35 are quoted from the committed protocol text
(research/OPTIMIZATION_PROTOCOL.md Amendments 5b/5c); they were not re-derived
from `outputs/v5/s3*/` during the context build → re-derive before citing in a paper.
