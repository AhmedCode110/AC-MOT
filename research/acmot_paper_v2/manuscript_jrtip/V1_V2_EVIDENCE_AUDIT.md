# V1/V2 Evidence Audit for the JRTIP Restructure

Status: pre-edit audit for the `jrtip-manuscript` branch. No experiments were
run for this audit, and no scientific result was changed.

## 1. Scope and chronology

The main paper should describe the historical sequence as:

1. an early heuristic prototype used only to identify the controllable detector
   variables and the detector--tracker causal interface;
2. V1, the first formal optimization-derived AC--MOT configuration;
3. V2, an identity-aware multi-objective extension;
4. cross-dataset and cross-pipeline studies;
5. the modern frozen AC--MOT v3 validation.

The heuristic prototype must not be described as the primary AC--MOT
generation, and the current long `Original AC--MOT` section is therefore
appropriate for Online Resource 1 rather than the main paper. The sentence
below is supported by the audit and is safe for the transition:

> An early heuristic prototype was used to identify the controllable detector
> variables and causal interface. We define V1 as the first formal
> optimization-derived AC--MOT configuration evaluated under the frozen
> experimental protocol.

## 2. V1: quality-oriented optimization

### Source records

- Frozen configuration: `research/paper_split/evidence/legacy/FROZEN_DEFENSIBLE_ACMOT_CONFIG.json`.
- Freeze narrative and held-out results: `research/paper_split/evidence/legacy/ACMOT_FINAL_SCIENTIFIC_FREEZE_2026-09-12.md`, Sections 7--9.
- Canonical optimizer: `scripts/optuna_sci_empirical_portable_colab.py`.
- V1 search/development evidence: `research/paper_split/evidence/legacy/V1_DEVELOPMENT_*` and the files mapped in `EVIDENCE_MAP.md`.

### Exact design

The canonical V1 search used `VisDrone2019-MOT-val` only; test data were not
used during search. The detector was fixed to pretrained YOLOv8n and the
tracker to the tuned ByteTrack configuration. The frozen temporal settings
were `W=7` and analysis stride `10`. The three resolution levels were
`512, 928, 960`, selected by the preceding validation resolution screen. The
supported confidence values were
`0.25, 0.30, 0.35, 0.40, 0.45`; the supported NMS values were
`0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70`.

The variables actually learned by the canonical joint study were:

- five non-negative SCI cue weights, normalized to sum to one;
- confidence endpoints at SCI 0 and 1, selected from the supported values;
- NMS endpoints at SCI 0 and 1, selected from the supported values; and
- two SCI switching thresholds, sorted into `mid` and `high`.

The five cue weights were `crowd=0.12949277455301997`,
`tiny=0.22174766876599927`, `edge=0.43371337893805056`,
`night=0.05355765312756694`, and `blur=0.16148852461536325`.
The selected confidence endpoints were `easy=0.30` and `hard=0.40`; both
NMS endpoints were `0.35`; and the thresholds were
`mid=0.13534938199219218` and `high=0.28728676236279177`.

The optimizer used a seeded Optuna TPE sampler and recorded the objectives
`maximize MOTA`, `minimize IDS`, and `maximize FPS`. The final constrained
selection rule was validation-only: require the FPS gate and `IDS <= Old-A3`,
then choose the highest MOTA, breaking ties by lower IDS, HOTA, IDF1, and FPS.
The maximum trial budget was 50 and the selected trial was Trial24. Therefore
V1 may be described as quality-oriented optimization, but not as hand-tuned
threshold selection and not as an optimization of variables that were fixed by
the preceding screens or frozen temporal configuration.

### V1 numbers and permitted interpretation

The Trial24 validation record is MOTA `0.23038087460093548`, IDS `270`, HOTA
`0.3611015551055777`, IDF1 `0.4075780119438273`, and FPS
`37.16857745688161`.

On the held-out record, baseline versus V1 is:

| profile | MOTA | HOTA | IDF1 | IDS | FPS |
|---|---:|---:|---:|---:|---:|
| baseline | 19.729 | 28.430 | 32.724 | 1235 | 36.528 |
| V1 | 26.948 | 33.835 | 41.546 | 1184 | 38.985 |

The archived 5000-resample seed-42 bootstrap supports V1--baseline gains of
`+7.2195` MOTA (95% CI `[5.4362, 9.2934]`), `+5.4051` HOTA (CI
`[4.1273, 6.9022]`), and `+8.8220` IDF1 (CI `[6.9005, 11.0537]`). The IDS
reduction is `51` with CI `[-114, 219]`, so it must not be presented as a
statistically established identity improvement. V1 has lower IDS than the
baseline in this held-out record, but V1 is not the lowest-IDS profile in the
historical comparison and must not be described as such.

## 3. V2: identity-aware multi-objective optimization

### Source records

- Canonical implementation at the historical source commit
  `2b400347584512ebc09527a3e0e01dad82299329`:
  `scripts/optuna_sci_v2_multiobjective_validation.py`.
- Development description: `docs/V2_DEVELOPMENT_STATUS.md`.
- Freeze narrative: `research/paper_split/evidence/legacy/ACMOT_FINAL_SCIENTIFIC_FREEZE_2026-09-12.md`, Sections 10--13.
- Post-selection technical testdev record:
  `research/paper_split/evidence/legacy/V2_TRIAL22_TESTDEV_RESULT.json`.

V2 reuses the frozen V1 validation design, cue calibration, temporal settings,
supported operating points, and controller parameterization. It changes the
optimization and selection target. The validation study explicitly maximizes
MOTA and minimizes IDS, imposes only `FPS >= 25`, and removes the V1
`IDS <= Old-A3` feasibility gate. It uses Optuna `TPESampler(seed=42)` with a
50-trial target and no testdev access during search.

The implementation explicitly constructs a feasible Pareto front: a trial
dominates another when its MOTA is no lower and its IDS is no higher, with at
least one strict improvement. It then records the highest-MOTA, lowest-IDS,
and balanced feasible Pareto choices. The balanced score is the documented
equal-weight normalized MOTA/IDS score, with tie-breaks by HOTA, IDF1, FPS,
and lower trial index. The official selected configuration is Trial22. The
archive records completed Trials 0--48 (49 complete trials), so the paper must
not imply that 50 trials completed.

### V2 numbers and permitted interpretation

The selected Trial22 validation record is MOTA `0.1933031405449551`, HOTA
`0.31651410689705645`, IDF1 `0.3422606845656919`, IDS `168`, and FPS
`51.406277608143924`. The frozen V2 profile has cue weights
`crowd=0.16464526567145857`, `tiny=0.17462652795045444`,
`edge=0.5076112530333374`, `night=0.12069564693725379`, and
`blur=0.03242130640749567`; confidence endpoints `0.40/0.40`; NMS endpoints
`0.60/0.35`; thresholds `mid=0.2927135841069045` and
`high=0.6661671600900015`; and the same `512/928/960`, `W=7`, stride-10
operating design.

The separate post-selection technical testdev rerun records MOTA `23.7919`,
HOTA `31.2184`, IDF1 `37.8702`, IDS `919`, and FPS `46.0240`. It must be
labelled as a post-hoc/secondary testdev record, not as the validation result
that selected Trial22. The main narrative may call V1 a quality-oriented
profile and V2 an identity/runtime-oriented profile, but it must not claim that
V1 had worse IDS than the baseline: the archived held-out values are V1 `1184`
versus baseline `1235`.

## 4. Historical UAVDT transfer

The historical V1/V2 UAVDT transfer is recorded in
`research/paper_split/evidence/legacy/UAVDT_FINAL_COMPARISON.json` and
`UAVDT_PER_SEQUENCE.csv`, as mapped by `EVIDENCE_MAP.md`. It should be a
compact pre-v3 transfer paragraph/table and must retain its documented
no-retuning and protocol boundary. It must not be presented as evidence that
the modern v3 controller was trained or retuned on UAVDT.

## 5. U2MOT cross-pipeline study

Source: `research/transfer_legacy/U2MOT_SPARSETRACK_EVIDENCE_AUDIT.md`,
package `FINAL_U2MOT_ACMOT_FREEZE_2026-09-14`.

The host is the published U2MOT implementation at commit
`7411211d17cb893f5fcd6be39cd4e5f91cfe1586`, with YOLOX-X and the documented
VisDrone configuration. Development used VisDrone validation (7 sequences,
2846 frames); held-out evaluation used VisDrone test-dev (17 sequences,
6635 frames). The frozen transfer kept the V1 SCI weights/transforms and did
not adapt the tracker. Because the V1 SCI thresholds would put all U2MOT
frames in the hard region, the action boundaries were recalibrated from the
U2MOT validation distribution using unsupervised q33/q67 boundaries, with
actions disabled and no ground-truth metric optimization. The U2MOT action
tuples were hand-set and have no selection record.

The held-out custom evaluator reported baseline MOTA `53.9`, IDF1 `69.8`, IDS
`1239`, FP `41385`, and FN `63241`; the transferred controller reported MOTA
`53.76679`, IDF1 `69.84939`, IDS `1152`, FP `40155`, and FN `64801`. Thus the
defensible interpretation is approximately unchanged accuracy, 87 fewer IDS,
1230 fewer FP, and 1560 more FN. No HOTA, confidence interval, or matched
baseline speed claim is supported. The study must not be called UAVDT and must
not be written as a universal quality win.

## 6. SparseTrack cross-pipeline study

Source: `research/transfer_legacy/U2MOT_SPARSETRACK_EVIDENCE_AUDIT.md`, frozen
manifest `FROZEN_CONTROLLER_MANIFEST.json`.

The host is SparseTrack commit `499844f32c5bb2332f9811f26cd70cf4e517d4e7`,
evaluated on MOT17 `val_half` using seven FRCNN sequences and 2652 frames. The
controller is the minimal Adaptive Edge V1 transfer: it computes an edge cue
every tenth frame, smooths it over seven frames, and changes only NMS from
`0.70` to `0.80`; confidence, resolution, detector, and tracker settings are
unchanged. The threshold came from a leave-one-sequence-out stump study on
the same seven development sequences, not an external test set.

Static versus adaptive matched evaluation was MOTA `76.8881` versus `76.9252`,
IDF1 `81.4943` versus `81.5299`, and IDS `136` versus `123`. The MOTA change
was `+0.0371` percentage points with bootstrap CI `[-0.041,+0.182]`; the
manifest explicitly concludes that superiority is not established. This is a
separate negative/near-neutral validation study, not evidence of broad
cross-dataset generalization, and no MOT17 test-server result is available.

## 7. Claim boundaries for the rewrite

- Preserve the modern v3 scientific content and its current frozen evidence.
- Add OATrack as a relevant recent UAV tracking host and state that the
  cross-host experiment tests portability of the same detector-control policy,
  not published superiority over OATrack.
- Keep matched-static and shuffled-timing controls as concise limitations or
  supplementary evidence. Do not claim a separate timing benefit for v3.
- Do not claim that V1 optimized resolution, `W`, or stride in the canonical
  Optuna study; those were established/frozen by prior screens and the design.
- Do not call the U2MOT evaluation UAVDT, do not claim universal transfer, and
  do not imply that the SparseTrack near-neutral result proves superiority.
- Do not invent optimization values, test metrics, confidence intervals,
  bibliographic metadata, or completed-trial counts.

This audit is the evidence gate for the new main-paper structure. Any wording
not supported here or by the existing v3 evidence map remains supplementary or
must be omitted.
