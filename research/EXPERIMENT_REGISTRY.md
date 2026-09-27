# Universal AC-MOT — Experiment Registry

Evaluation protocol for every row (unless stated): custom class-agnostic
internal protocol (NOT official VisDrone): GT classes {1,4,5,6,9}, score==1,
truncation<2, occlusion<2, IoU 0.5, class ignored. HOTA = TrackEval;
MOTA/IDF1/IDS/FP/FN/Precision/Recall = motmetrics 1.4.0
(`tools/eval_local.py`, verified identical to the Colab evaluator on all
14 V1 rows: max |diff| = 0.0).

Dataset: VisDrone2019-MOT-val (7 sequences, 2846 frames, 67,345 GT boxes).
Test-dev: untouched.

Detectors: YOLOv8n (sha256 f59b3d83…), RT-DETR-L (sha256 6de60b10…).
Tracker: ByteTrack (ultralytics 8.3.200). Dev hardware: Mac MPS
(development only; no timing from Mac is reported).

"Cache replay" = `tools/cache_detections.py` + `tools/run_policy_validation.py`.
Fidelity: YOLO V1/V2b full-7 and all 0137/0268 V1/V2b/V2cA/B/C rows reproduce
Colab exactly; RT-DETR 0137/0268 rows reproduce Colab (one row differs by
~1 FP of 24,636: MPS vs CUDA arithmetic); live-vs-cache RT-DETR V2cA 0137 is
byte-identical. Valid only while NMS is fixed at 0.45 (guarded in code).

| ID | Code / policy | Hypothesis | Data | Result | Conclusion → next |
|---|---|---|---|---|---|
| E01 | V1 (Colab) | Online percentile normalization is enough | val-7, both | YOLO MOTA −2.19 / HOTA 31.98; RT-DETR MOTA −112.49 / HOTA 29.80 | Equal percentage ≠ equal count → candidate explosion |
| E02 | V2b (Colab) | Density budget + Top-K | val-7, both | YOLO 11.83/27.60, RT 18.39/33.31 | Explosion fixed, recall/HOTA destroyed (budget applied twice) |
| E03 | V2c-A/B/C (Colab) | Which threshold carries the damage | 0137, 0268 | A: 0137 good, RT-0268 −217; B: 0268 good, 0137 bad; C: in between | Additive-offset density thresholds are the over-suppressor |
| E04 | eval_local | Mac evaluator == Colab evaluator | V1 tracks, val-7 | max diff 0.0 on MOTA/HOTA/IDF1/IDS | Reference protocol preserved |
| E05 | cache replay | Replay == live | 0137/0268 all V1–V2c; full-7 YOLO V1,V2b | exact (see above) | Cache usable for fixed-NMS policies |
| E06 | offline sim | Track reliability (persist3×survival) separates failure | V2cA audits, 4 cases | RT-0268 0.53 vs others 0.76–0.87 | Promising signal; budget-only effect predicted weak |
| E07 | V2d A/B/C + combos (7 variants) | Reliability-scaled budget / birth / persistent-track SCI feedback | 0137, 0268, both | best RT-0268 MOTA −91.5 | FAIL. Budget floor 12 + weak linear scaling; births in dead zone |
| E08 | GT FP attribution (offline) | 0268 FPs are protocol artifacts | V2cA tracks | RT-0268: 89% unexplained background; 0137: ~50–57% FPs are excluded real classes (people/bicycle/motor) | Explosion is real; protocol limitation noted for 0137 |
| E09 | per-track continuity (offline) | False tracks are less continuous | V2cA tracks | FP-track continuity 0.56 vs TP 0.85 on RT-0268; gating would cut 24–30% TP | Not a clean separator |
| E10 | det-stream persistence (offline) | Raw-stream persistence separates failure | cache, 2 seqs | 0137≈0.85, 0268≈0.76 for BOTH detectors | Scene property, not failure signal → rejected (only 2 seqs; re-measure on val-7: E15) |
| E11 | V2e (proportional thresholds) ± reliability | Additive offsets are the bug; scale tails proportionally | 0137, 0268 | V2e-ACb: 3/4 criteria; RT-0268 MOTA −55 | Structural fix helps 0137 (IDS 48→10) but 0268 still fails |
| E12 | per-rank precision (offline) | Min budget 12 too large for sparse clutter | cache | RT-0268 precision ≤0.35 beyond rank 4; 0137 ≥0.9 to rank 12 | Need a rank-quality signal, not a count signal |
| E13 | V2f closed-loop trust (r*=0.80/0.85/0.90) | Integrate reliability to shrink budget | 0137, 0268 | RT-0268 MOTA 18 but 2 tracks/frame; YOLO-0268 HOTA 33.5→27 | FAIL for all r*: reliability conflates hard scene with false candidates (windup) |
| E14 | per-rank det persistence + score profile (offline) | Which within-frame signal tracks precision | cache, 2 seqs | persistence high for FP clutter (0.92–0.96); score/leader ratio tracks precision for both detectors | Normalization erased the leader-relative score drop-off |
| E15 | V2g (leader-relative gate) ρ∈{0.4,0.5,0.6}, demote vs drop; V1g ablation | Demote candidates far below the detector's recent leader | 0137, 0268 | V2gA ρ=0.5: all 4 diagnostic criteria pass; ρ 0.4–0.6 all pass; drop fails YOLO-0268; V1+gate also passes | Gate is the core fix. ρ NOT frozen — must be selected by the cross-sequence protocol (E17+) |
| E16 | full-7 YOLO: V1,V2b,V2cA,V2gA,V1g | Frozen candidates on all sequences | val-7 YOLO | V1 −2.19/31.98, V2b 11.83/27.60 (both = Colab), V2cA 10.28/32.85, V2gA 17.46/31.85, V1g 16.91/32.66 | RT-DETR pending |
| E17 | signal audit val-7 × 2 detectors (tools/signal_audit.py) | Which online signal separates TP from background FP everywhere | cache, all 14 cells | leader-relative score AUC 0.69–0.90 (≥ rank in 12/14); det-stream persistence/chains 0.53–0.67 everywhere | Temporal persistence rejected as primary signal (all sequences, both families) |
| E18 | 28-config grid + LOSO (tools/optimize_protocol.py, OPTIMIZATION_PROTOCOL.md) | Select architecture/ρ/κ jointly across detectors | val-7 × 2 | J2/J4 → gate_r0.4 (RT-DETR MOTA 3.3; 3 RT cells MOTA<0); J1/J3 → dens_k2_r0.5 (7/7 folds); Amendment 2 lexicographic → gate_r0.6 (L-J3 = L-J4) | v1 freeze (tag universal-acmot-v1-freeze, e56c2f3); J2 degenerates to weakest detector (documented) |
| E19 | baselines: default, per-detector oracle, shared-static (tools/baselines.py) | How much is lost vs detector-specific tuning; is a single static setting enough | val-7 × 2 | Oracle LOSO YOLO 21.05/34.10/37.62, RT 25.40/41.20/47.44; shared-static (raw 0.5, res 832) LOSO YOLO 21.10/33.33/36.04, RT 23.58/41.48/48.05 ≥ universal v1 | NEGATIVE: on these two detectors a shared static raw threshold at 832 is competitive. At matched compute (736) universal v1 wins HOTA/IDF1 on both |
| E20 | score-calibration stress test (tools/stress_scores.py) | Which frozen system survives a detector recalibration | val-7 × 2 × 5 transforms | shared-static collapses under scale×0.5 (0 tracks); universal v1 not invariant to temperature (YOLO HOTA 25.7 at T=0.5; RT MOTA 7.9 at T=2) | v1 superseded → V3 invariant design (Amendment 3) |
| E21 | z-logit gate audit | Is a Platt-invariant gate statistic still discriminative | cache val-7 × 5 transforms | AUC z 0.77–0.80 (exactly invariant under temperature) vs ratio 0.80 | Accept small AUC loss for exact invariance |
| E22 | V3 grid + LOSO (ACMOT_GRID=v3) | Select τ within the invariant family | val-7 × 2 | lexicographic L-J3 = L-J4 → τ=0.75, 7/7 folds agree; YOLO 16.51/32.00/34.07 IDS 108; RT 25.49/39.07/42.34 IDS 58; ECDF+ratio ablation ≈ v1 | V3 frozen (configs/universal_acmot_policy.json) |
| E23 | V3 stress test | Exact invariance at dataset level | val-7 × 2 × 5 | identical metrics for T=2, T=0.5; stable under scale/pow (≤2 YOLO cells MOTA<0) | Calibration-invariance claim supported for the Platt family |

Test-dev annotation fingerprint (sha256 of per-file sha256 list): 89d9dccb…c24; 4 files restored from the official zip in 2026-09-06 (logs in dataset root).
| E24 | controller ablation on V3 (ACMOT_GRID=audit) | Does the legacy SCI add value on top of V3 | val-7 × 2 | SCI resolution ≈ uniform 640/736 mix at equal pixel cost; fixed 736 ≥ SCI; SCI sensitivity ≈ constant 0.4 | SCI unjustified by evidence |
| E25 | per-frame cue→resolution-benefit audit (tools/cue_benefit_audit.py) | Which causal cue predicts where 832 helps | cache val-7 × 2 | within-sequence |ρ|<0.3, signs flip across sequences and detectors | No reliable cue |
| E26 | matched-compute single-cue vs RANDOM allocation (ACMOT_GRID=cues) | End-to-end cue usefulness | val-7 × 2 | random 33.06/39.71 HOTA ≥ every cue incl. legacy SCI (selection 0/2 for all) | SCI removed from decision path (Amendment 4) |
| E27 | NMS 0.45 vs detector-native 0.7 (tools/audit/nms_audit.py) | Is the legacy suppression request justified | YOLO val-7 at 736/832 | 0.45 wins 6/7 sequences at both levels (+0.6/+1.4 HOTA, +1.4/+2.0 IDF1) | Keep 0.45 (category C) |
| E28 | V4 memory-length / resolution sensitivity (ACMOT_GRID=v4sens) | Are memory lengths arbitrary-but-harmless | val-7 × 2 | all within ±0.4 HOTA, 0 catastrophic cells; 832 > 736 > 640 for YOLO | Memory lengths kept as structural (documented) |
| E29 | V4 nested LOSO (360 configs, tools/v4_select.py) | Select shared global parameters | val-7 × 2 | τ .75 (5/7), s .4 (5/7), assoc .10 (5/7), birth flat, tracker AC (7/7); outer-CV YOLO 15.23/31.84/34.05, RT 20.82/37.91/40.94 | V4 frozen |
| E30 | V4 checks | live == replay; temperature invariance; separation tests | RT-DETR 0137 live; 0268 both | byte-identical; identical hashes; 3/3 PASS | Ready for held-out |
