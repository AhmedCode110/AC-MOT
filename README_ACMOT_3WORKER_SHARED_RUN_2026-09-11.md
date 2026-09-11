# AC-MOT 3-Worker Shared-Drive Run — 2026-09-11

This is an **optional distributed execution wrapper** around the existing defensible AC-MOT pipeline.

The original complete one-command pipeline is intentionally left unchanged.

Existing full pipeline:

```text
scripts/run_defensible_acmot_pipeline_colab.py
```

New distributed runner:

```text
scripts/run_defensible_acmot_3workers_shared.py
```

Shared status / merge checker:

```text
scripts/merge_defensible_acmot_3workers_shared.py
```

## Important provider note

Use multiple runtimes/accounts only where this is permitted by the compute provider's terms. This design is a generic multi-worker execution layout; it should not be used to evade account or quota restrictions.

---

# 1. Why this version exists

The full defensible AC-MOT validation pipeline contains three major stages:

```text
Stage 1
Resolution + Confidence + NMS operating-point ablations

Stage 2
Temporal ablation

Stage 3
Empirical joint Optuna
```

Stage 1 and Stage 2 are independent and can run at the same time.

Stage 3 depends on outputs from both Stage 1 and Stage 2.

Therefore the scientifically safest split across three workers is:

```text
Worker 1 -> Stage 1
Worker 2 -> Stage 2
Worker 3 -> waits, then Stage 3
```

This keeps the existing scientific methodology unchanged instead of splitting the adaptive TPE Optuna study into independent incompatible sub-studies.

---

# 2. Why Optuna is not split into three independent studies

The full optimizer uses Optuna TPE.

TPE is adaptive: later trials depend on earlier completed trials.

If 50 trials were naively divided into three completely separate databases, the result would no longer be the same search process as the original 50-trial study.

Therefore this distributed version preserves:

```text
one joint Optuna study
one study database
one final selection rule
```

Worker 3 runs that study only after the validation search space and temporal configuration are available.

This preserves the original methodology.

---

# 3. Shared folder requirement

All three workers must point to the **same underlying Google Drive shared folder**.

Recommended visible path on each worker:

```text
/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers
```

If the shared folder appears under a different path for a worker, set:

```python
import os
os.environ["ACMOT_SHARED_RESULT_ROOT"] = "/content/drive/MyDrive/YOUR_SHARED_FOLDER/defensible_acmot_3workers"
```

The important rule is that all three paths must resolve to the same shared folder contents.

Do not let each worker write to a different private Drive result folder.

---

# 4. Hardware rule

For thesis FPS comparison, use the same hardware class for all scientific runs.

Preferred:

```text
NVIDIA T4
```

Before starting a worker:

```python
import torch
print(torch.cuda.get_device_name(0))
```

If the scientific comparison is defined on T4, do not mix T4 FPS results with a different GPU and call them directly comparable.

---

# 5. Repository setup on every worker

Each worker should mount Drive first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then clone or synchronize the repository:

```python
from pathlib import Path
import subprocess

REPO = Path('/content/AC-MOT')

if REPO.exists():
    subprocess.run(['git', '-C', str(REPO), 'fetch', 'origin', 'main'], check=True)
    subprocess.run(['git', '-C', str(REPO), 'reset', '--hard', 'origin/main'], check=True)
else:
    subprocess.run([
        'git', 'clone',
        'https://github.com/AhmedCode110/AC-MOT.git',
        str(REPO),
    ], check=True)
```

All workers must use the same current repository revision for a scientific run.

---

# 6. Worker 1 — operating-point ablations

Worker 1 runs:

```text
Resolution sweep
Confidence sweep
NMS IoU sweep
```

Full mode:

```python
import os

os.environ['ACMOT_WORKER_ID'] = '1'
os.environ['ACMOT_SMOKE_TEST'] = '0'
os.environ['ACMOT_SHARED_RESULT_ROOT'] = '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'

%run /content/AC-MOT/scripts/run_defensible_acmot_3workers_shared.py
```

Expected important outputs include:

```text
OPERATING_RESOLUTION_SWEEP.csv
OPERATING_CONFIDENCE_SWEEP.csv
OPERATING_NMS_SWEEP.csv
SCIENTIFIC_SEARCH_SPACE.json
OPERATING_ABLATION_REPORT.json
WORKER1_OPERATING_DONE.json
```

Worker 1 can run at the same time as Worker 2.

---

# 7. Worker 2 — temporal ablation

Worker 2 runs the full 25-combination temporal experiment:

```text
Smoothing windows = 1, 3, 5, 7, 9
Analysis strides  = 1, 5, 10, 15, 20
```

Full mode:

```python
import os

os.environ['ACMOT_WORKER_ID'] = '2'
os.environ['ACMOT_SMOKE_TEST'] = '0'
os.environ['ACMOT_SHARED_RESULT_ROOT'] = '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'

%run /content/AC-MOT/scripts/run_defensible_acmot_3workers_shared.py
```

Expected important outputs include:

```text
TEMPORAL_ABLATION_FULL.csv
FROZEN_TEMPORAL_CONFIG.json
WORKER2_TEMPORAL_DONE.json
```

Worker 2 does not need Stage 1 to finish first.

---

# 8. Worker 3 — joint Optuna

Worker 3 runs the empirical joint SCI optimization.

It waits until these files exist in the shared folder:

```text
SCIENTIFIC_SEARCH_SPACE.json
FROZEN_TEMPORAL_CONFIG.json
```

Then it launches the existing joint optimizer.

Full mode:

```python
import os

os.environ['ACMOT_WORKER_ID'] = '3'
os.environ['ACMOT_SMOKE_TEST'] = '0'
os.environ['ACMOT_OPTUNA_TRIALS'] = '50'
os.environ['ACMOT_SHARED_RESULT_ROOT'] = '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'

%run /content/AC-MOT/scripts/run_defensible_acmot_3workers_shared.py
```

Worker 3 may be started after Workers 1 and 2, or it may be started earlier and left waiting.

It polls the shared folder until prerequisites exist.

Expected important outputs include:

```text
DETECTOR_DERIVED_CUE_CALIBRATION.json
EMPIRICAL_OPTUNA.db
EMPIRICAL_OPTUNA_TRIALS.csv
FROZEN_DEFENSIBLE_ACMOT_CONFIG.json
WORKER3_JOINT_DONE.json
```

The original Optuna resume mechanism remains active.

---

# 9. Smoke mode across three workers

Run the distributed smoke test before the full distributed run.

Worker 1:

```python
import os
os.environ['ACMOT_WORKER_ID'] = '1'
os.environ['ACMOT_SMOKE_TEST'] = '1'
os.environ['ACMOT_SHARED_RESULT_ROOT'] = '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'
%run /content/AC-MOT/scripts/run_defensible_acmot_3workers_shared.py
```

Worker 2:

```python
import os
os.environ['ACMOT_WORKER_ID'] = '2'
os.environ['ACMOT_SMOKE_TEST'] = '1'
os.environ['ACMOT_SHARED_RESULT_ROOT'] = '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'
%run /content/AC-MOT/scripts/run_defensible_acmot_3workers_shared.py
```

Worker 3:

```python
import os
os.environ['ACMOT_WORKER_ID'] = '3'
os.environ['ACMOT_SMOKE_TEST'] = '1'
os.environ['ACMOT_OPTUNA_TRIALS'] = '2'
os.environ['ACMOT_SHARED_RESULT_ROOT'] = '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'
%run /content/AC-MOT/scripts/run_defensible_acmot_3workers_shared.py
```

Smoke mode still does not create a scientific frozen final configuration.

---

# 10. Shared-result merge / status check

After all three workers finish, run on any worker:

```python
import os

os.environ['ACMOT_SMOKE_TEST'] = '0'
os.environ['ACMOT_SHARED_RESULT_ROOT'] = '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'

%run /content/AC-MOT/scripts/merge_defensible_acmot_3workers_shared.py
```

It writes:

```text
MERGED_3WORKER_FULL_STATUS.json
```

This file verifies that all expected artifacts from all three stages are present.

For smoke mode it writes:

```text
MERGED_3WORKER_SMOKE_STATUS.json
```

---

# 11. What "merge" means here

There is no unsafe merging of three competing metric databases.

The workflow merges naturally through the shared folder:

```text
Worker 1 outputs
        +
Worker 2 outputs
        ↓
Worker 3 consumes both
        ↓
One frozen AC-MOT configuration
```

The final status script only verifies the shared artifacts and writes one combined manifest.

This is intentionally safer than letting three workers write simultaneously to one SQLite Optuna database over Google Drive.

---

# 12. Resume behavior

Worker 1 retains the original operating-sweep CSV resume behavior.

Worker 2 retains the original temporal CSV resume behavior.

Worker 3 retains the original Optuna SQLite resume behavior.

Therefore if a runtime disconnects, reconnect that worker to the same shared result folder and rerun the same worker command.

Do not use force-reset environment variables unless intentionally restarting an experiment.

---

# 13. Validation data on each worker

Each independent runtime must be able to access `VisDrone2019-MOT-val`.

Workers 1 and 2 already contain portable validation-download logic in their underlying scripts.

Worker 3's new wrapper also checks for the validation set and downloads the portable public validation copy if needed.

The validation dataset itself does not have to be shared between workers as long as it is the same dataset content.

The **result folder must be shared**.

---

# 14. Scientific equivalence to the original full version

This wrapper preserves the main methodology because it calls the existing stage implementations unchanged:

```text
scientific_operating_ablation_portable_colab.py

temporal_ablation_portable_colab.py

optuna_sci_empirical_portable_colab.py
```

The split changes where stages execute, not the scientific selection logic.

It deliberately does not split the adaptive Optuna study into three independent optimizers.

---

# 15. Final held-out test is still separate

This distributed validation workflow does **not** run test-dev.

After:

```text
Worker 1 complete
Worker 2 complete
Worker 3 complete
MERGED_3WORKER_FULL_STATUS.json says complete=true
```

review the frozen configuration first.

Only then, if the final protocol is approved, run the existing separate final-test script:

```text
scripts/run_frozen_final_test_portable_colab.py
```

The final held-out test remains one locked evaluation and must not be distributed into parameter-selection runs.

---

# 16. Recommended execution order

```text
Create one shared Google Drive result folder
        ↓
Mount that same shared folder on all three workers
        ↓
Run distributed smoke test
        ↓
Verify MERGED_3WORKER_SMOKE_STATUS.json
        ↓
Clear/use a fresh full-result folder if desired
        ↓
Worker 1 full + Worker 2 full in parallel
        ↓
Worker 3 waits then runs 50-trial joint Optuna
        ↓
Run merge/status checker
        ↓
Review FROZEN_DEFENSIBLE_ACMOT_CONFIG.json
        ↓
Only then consider the one-shot held-out final test
```

## Current status

The original complete pipeline remains available and unchanged.

The new three-worker shared-result wrapper and merge/status checker have been added as separate files so either execution mode can be used without overwriting the other.
