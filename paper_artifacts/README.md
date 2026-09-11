# Paper artifacts

This directory contains the publication-facing evidence for the locked AC-MOT study.

## `final/`

Held-out `VisDrone2019-MOT-test-dev` artifacts for the three predeclared systems:

- `FINAL_TEST_3WORKER_PROTOCOL.json`
- `FINAL_TEST_WORKER_1_BASELINE_DEFAULT.json`
- `FINAL_TEST_WORKER_2_OLD_ACMOT_FROZEN.json`
- `FINAL_TEST_WORKER_3_NEW_ACMOT_FROZEN.json`
- `FINAL_TEST_COMPARISON_3WORKER.csv`
- `FINAL_TEST_RESULTS_3WORKER.json`
- `FINAL_TEST_DONE.json`

The held-out results are exposed and locked. Do not retune on this test set and later describe another run as the same unbiased final test.

## `config/`

Frozen validation-selected configuration used to define `New_ACMOT_Frozen`.

## `validation/`

Validation-only component ablations and historical comparison evidence. These files are not held-out results.

## Hashes

`CANONICAL_SHA256SUMS.txt` records hashes from the verified Google Drive source artifacts. The final protocol also embeds hashes for its required prerequisite artifacts.

## Evaluation scope

The reported metrics use a custom class-agnostic AC-MOT TrackEval protocol, not the official VisDrone leaderboard preprocessing/evaluation protocol.
