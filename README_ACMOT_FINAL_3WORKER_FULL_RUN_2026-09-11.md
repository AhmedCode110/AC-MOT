# AC-MOT Final 3-Worker Full Validation Run — 2026-09-11

This is the final full-validation execution guide for the three-worker shared-result workflow.

The distributed runner is:

```text
scripts/run_defensible_acmot_3workers_shared.py
```

Worker 1 now uses:

```text
scripts/scientific_operating_ablation_fair_timing.py
```

The Stage-1 fair-timing launcher keeps the existing scientific search grid and selection logic unchanged while fixing two FPS-isolation issues:

1. It disables unused visual SceneAnalyzer computation during static operating-point screens.
2. It explicitly warms the exact tested static resolution before measured inference, including the full 512..960 resolution grid.

Warm-up remains excluded from reported processing FPS. Stages 1-3 remain validation-only; test-dev is not accessed.

## Shared folder

All three workers must see the same underlying writable Google Drive folder at:

```text
/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers
```

Do not create three independent folders with the same name.

## Common setup on every account/runtime

### Cell 1 — Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2 — Verify the shared folder

```python
from pathlib import Path

SHARED = Path('/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers')
print('Exists:', SHARED.exists())
print('Path:', SHARED)
assert SHARED.exists(), 'Shared folder is not visible on this account.'
```

### Cell 3 — Clone/synchronize the exact latest repository

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

subprocess.run(['git', '-C', str(REPO), 'log', '-1', '--oneline'], check=True)
```

For this finalized version, the latest commit should be at least the commit that switched the distributed runner to the timing-fair Stage 1 (`10c4644...`) or any later intentional commit.

### Cell 4 — Verify the scientific GPU class

```python
import torch
assert torch.cuda.is_available(), 'CUDA GPU is not available.'
print(torch.cuda.get_device_name(0))
```

For direct thesis FPS comparability, use NVIDIA Tesla T4 for every worker.

---

# Account 1 — Worker 1 — FULL

Runs the full operating-point validation screens:

```text
15 resolutions: 512..960 step 32
10 confidence values: 0.05..0.50 step 0.05
11 NMS IoU values: 0.30..0.80 step 0.05
```

Run after the four common setup cells:

```python
import os

os.environ['ACMOT_WORKER_ID'] = '1'
os.environ['ACMOT_SMOKE_TEST'] = '0'
os.environ['ACMOT_SHARED_RESULT_ROOT'] = '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'

%run /content/AC-MOT/scripts/run_defensible_acmot_3workers_shared.py
```

Expected final Worker-1 marker:

```text
WORKER1_OPERATING_DONE.json
```

Important search-space output:

```text
SCIENTIFIC_SEARCH_SPACE.json
```

---

# Account 2 — Worker 2 — FULL

Runs the full temporal validation ablation:

```text
smoothing windows = [1, 3, 5, 7, 9]
analysis strides  = [1, 5, 10, 15, 20]
25 combinations total
```

Run after the four common setup cells:

```python
import os

os.environ['ACMOT_WORKER_ID'] = '2'
os.environ['ACMOT_SMOKE_TEST'] = '0'
os.environ['ACMOT_SHARED_RESULT_ROOT'] = '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'

%run /content/AC-MOT/scripts/run_defensible_acmot_3workers_shared.py
```

Expected final Worker-2 marker:

```text
WORKER2_TEMPORAL_DONE.json
```

Important frozen temporal output:

```text
FROZEN_TEMPORAL_CONFIG.json
```

Worker 1 and Worker 2 may run in parallel.

---

# Account 3 — Worker 3 — FULL

Worker 3 runs the single joint Optuna TPE study after Worker 1 and Worker 2 finish.

It consumes:

```text
SCIENTIFIC_SEARCH_SPACE.json
FROZEN_TEMPORAL_CONFIG.json
```

Then it runs 50 trials and freezes the selected validation configuration.

Run after the four common setup cells and after Workers 1 and 2 have finished:

```python
import os

os.environ['ACMOT_WORKER_ID'] = '3'
os.environ['ACMOT_SMOKE_TEST'] = '0'
os.environ['ACMOT_OPTUNA_TRIALS'] = '50'
os.environ['ACMOT_SHARED_RESULT_ROOT'] = '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'

%run /content/AC-MOT/scripts/run_defensible_acmot_3workers_shared.py
```

Expected final Worker-3 marker:

```text
WORKER3_JOINT_DONE.json
```

Important final validation output:

```text
FROZEN_DEFENSIBLE_ACMOT_CONFIG.json
```

The Optuna resume mechanism remains active. If the runtime disconnects, reconnect the same worker to the same shared result folder, synchronize the repository again, and rerun the same Worker-3 command. Do not force-reset the study unless intentionally restarting the experiment.

---

# Merge/status check

After all three workers finish, run on any one account/runtime:

```python
import os

os.environ['ACMOT_SMOKE_TEST'] = '0'
os.environ['ACMOT_SHARED_RESULT_ROOT'] = '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'

%run /content/AC-MOT/scripts/merge_defensible_acmot_3workers_shared.py
```

Expected output:

```text
MERGED_3WORKER_FULL_STATUS.json
```

Review the merged status and `FROZEN_DEFENSIBLE_ACMOT_CONFIG.json` before any held-out test.

## Final-test rule

Do not run `scripts/run_frozen_final_test_portable_colab.py` until the full validation outputs have been reviewed and the frozen configuration is accepted. The held-out test remains one separate locked evaluation and is not used for parameter selection.

## Resume rule

If Worker 1 or Worker 2 disconnects, rerun the same worker command against the same shared folder. Their existing CSV resume behavior is retained. Worker 3 retains the Optuna SQLite resume behavior.

Do not delete the shared result folder during the full experiment. Do not set force-reset environment variables unless deliberately restarting the scientific experiment.
