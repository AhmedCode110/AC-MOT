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
