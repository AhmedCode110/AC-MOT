# AC-MOT V2 Development

This working branch was created from AC-MOT V1 frozen commit:

`a6c1fa49fce1d402513c2df05b7d04b962a6e89e`

## V1 immutable reference

Branch:

`acmot-final-frozen-2026-09-11`

Tag:

`v1.0.0-acmot-frozen`

## V2 branch

`experiment/multiobjective-mota-ids-v2`

V2 is editable.

V1 must never be modified.

## Intended next experiment

Multi-objective Optuna validation study

Objectives:

- Maximize MOTA
- Minimize IDS

Constraint:

- FPS >= 25

Do not access held-out test-dev for optimization or candidate selection.
