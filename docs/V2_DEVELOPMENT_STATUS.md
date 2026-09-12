# AC-MOT V2 Development

This working branch was created from AC-MOT V1 frozen commit:

`a6c1fa49fce1d402513c2df05b7d04b962a6e89e`

## V1 immutable reference

Branch:

`acmot-final-frozen-2026-09-11`

Tag:

`v1.0.0-acmot-frozen`

V1 must never be modified.

## V2 branch

`experiment/multiobjective-mota-ids-v2`

V2 is the editable post-V1 research branch.

## Canonical Google Drive storage policy

All AC-MOT Colab accounts use the same shortcut root:

`/content/drive/MyDrive/AC-MOT-shared`

For AC-MOT project work:

- search for project files only inside `AC-MOT-shared`
- read project inputs only from `AC-MOT-shared`
- write all new project artifacts only inside `AC-MOT-shared`
- do not write new artifacts elsewhere in MyDrive
- treat `defensible_acmot_3workers` as frozen V1 evidence and do not overwrite it

Canonical shared root Drive folder ID:

`1Ry6tnO69Fpx9FFn2nKTPkRxy2n8o9Ah1`

Canonical data root:

`/content/drive/MyDrive/AC-MOT-shared/AC-MOT-data`

Drive folder ID:

`1aGTEWIz-EzVGguknIIrgI71dhxlzFd3i`

Validation dataset:

`/content/drive/MyDrive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-val`

Drive folder ID:

`1TCRyp9yxdoYyDF9McTHcd8wb8T3sG1xm`

Held-out test-dev dataset, recorded for provenance only and forbidden for V2 tuning/debugging/selection:

`/content/drive/MyDrive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-test-dev`

Frozen V1 results/reference root:

`/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers`

Drive folder ID:

`1RRjzBy1Q_K4415A1x4FCD7B4xb60QYBs`

Default V2 write root:

`/content/drive/MyDrive/AC-MOT-shared/V2_MULTI_OBJECTIVE_MOTA_IDS`

The canonical cross-account launcher is:

`scripts/run_v2_colab_shared.py`

The canonical path registry is:

`config/V2_COLAB_SHARED_PATHS.json`

Use the launcher for Colab runs so the shared-root policy is enforced automatically.

## Scientific distinction from V1

V1 used a multi-value Optuna study that recorded MOTA, IDS, and FPS, but its frozen candidate-selection rule was constrained and MOTA-prioritized:

- FPS >= 25
- IDS <= Old-A3 validation IDS
- then select the highest MOTA
- tie-break by lower IDS, then HOTA, IDF1, FPS

The frozen V1 selected candidate was Trial 24.

V2 changes the optimization/selection formulation while reusing the frozen V1 validation design.

V2 objectives:

- Maximize MOTA
- Minimize IDS

V2 feasibility gate:

- FPS >= 25

The V1 hard gate `IDS <= 271` is intentionally NOT used in V2 because IDS is now a primary optimization objective.

## Controlled V2 inputs

V2 reuses the frozen V1 inputs rather than rerunning earlier selection stages:

- `SCIENTIFIC_SEARCH_SPACE.json`
- `FROZEN_TEMPORAL_CONFIG.json`
- `DETECTOR_DERIVED_CUE_CALIBRATION.json`
- `FROZEN_DEFENSIBLE_ACMOT_CONFIG.json`

These are read from:

`/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers`

The controlled temporal design remains:

- smoothing window = 7
- analysis stride = 10

The V1 sampler family and seed are retained for comparability:

- `TPESampler`
- seed = 42
- full validation budget = 50 trials

## V2 implementation

V2 implementation script:

`scripts/optuna_sci_v2_multiobjective_validation.py`

The script creates an Optuna study with:

```python
directions=["maximize", "minimize"]
```

for:

1. MOTA
2. IDS

FPS is recorded for every trial and used only as a feasibility gate.

The V2 script explicitly does not use `study.best_trial` or `study.best_value`.

## Candidate selection

After the study completes, V2 constructs the Pareto front from feasible trials only.

A trial A dominates trial B when:

- A.MOTA >= B.MOTA
- A.IDS <= B.IDS
- at least one comparison is strict

V2 reports three candidates:

1. highest-MOTA feasible Pareto trial
2. lowest-IDS feasible Pareto trial
3. balanced feasible Pareto trial

Balanced score:

`0.5 * MOTA_norm + 0.5 * IDS_good_norm`

Tie-breakers:

1. higher HOTA
2. higher IDF1
3. higher FPS
4. lower trial number

## Safety rules

V2 is validation-only.

Do not access `VisDrone2019-MOT-test-dev` for optimization, debugging, candidate selection, or smoke testing.

The V2 optimizer contains explicit rejection for a `test-dev` validation path.

The canonical launcher additionally enforces that V2 writes stay under `AC-MOT-shared` and never inside the frozen V1 root.

## Smoke test

Run a small smoke study first on a CUDA/T4 Colab runtime.

After cloning the V2 branch and mounting Drive:

```python
import os
os.environ["ACMOT_V2_SMOKE_TEST"] = "1"
os.environ["ACMOT_V2_OPTUNA_TRIALS"] = "3"
%run /content/AC-MOT/scripts/run_v2_colab_shared.py
```

Smoke outputs are stored under:

`/content/drive/MyDrive/AC-MOT-shared/V2_MULTI_OBJECTIVE_MOTA_IDS/SMOKE`

Smoke results must not be reported as scientific results.

Smoke mode does not create the final `V2_DONE.json`.

## Full validation run

Only after smoke validation succeeds:

```python
import os
os.environ["ACMOT_V2_SMOKE_TEST"] = "0"
os.environ["ACMOT_V2_OPTUNA_TRIALS"] = "50"
%run /content/AC-MOT/scripts/run_v2_colab_shared.py
```

Full results are written under:

`/content/drive/MyDrive/AC-MOT-shared/V2_MULTI_OBJECTIVE_MOTA_IDS`

## Expected full-run outputs

- `V2_PROTOCOL.json`
- `V2_SEARCH_SPACE.json`
- `V2_INPUT_HASHES.json`
- `V2_MULTI_OBJECTIVE_OPTUNA.db`
- `V2_ALL_TRIALS.csv`
- `V2_PARETO_FRONT.csv`
- `V2_HIGHEST_MOTA_CANDIDATE.json`
- `V2_LOWEST_IDS_CANDIDATE.json`
- `V2_SELECTED_CANDIDATE.json`
- `V1_VS_V2_VALIDATION_COMPARISON.csv`
- `V2_STUDY_SIGNATURE.json`
- `V2_DONE.json`
- `plots/`
- `logs/`

## Scientific interpretation

V2 is a post-V1 validation experiment. It does not replace, relabel, or reopen the frozen V1 held-out final result.

If V2 is later evaluated on the already-exposed VisDrone test-dev split, that evaluation must be described as post-hoc/secondary rather than a new unbiased held-out final test.

## Current status

V2 code has been implemented on the V2 branch.

Canonical cross-account Drive paths and shared-root-only storage policy are now recorded in the project.

No V2 scientific trial has been run yet.

No held-out test was accessed.
