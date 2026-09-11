# AC-MOT Locked Final Test — 3 Worker / 3 Account Workflow

This workflow is for the final held-out `VisDrone2019-MOT-test-dev` evaluation **after all validation work is finished and frozen**.

It compares exactly three predeclared systems:

1. `Baseline_Default`
2. `Old_ACMOT_Frozen`
3. `New_ACMOT_Frozen`

The goal is to compare the project baseline, the historical heuristic AC-MOT, and the new validation-optimized AC-MOT under one locked protocol.

> This is the custom class-agnostic AC-MOT research protocol used by this repository. It is **not** the official VisDrone leaderboard protocol.

## Scientific rules

Before any account accesses test-dev:

- Worker 1/2/3 validation stages must be complete.
- `FROZEN_DEFENSIBLE_ACMOT_CONFIG.json` must already exist.
- `DETECTOR_DERIVED_CUE_CALIBRATION.json` must already exist.
- `NEW_ACMOT_COMPONENT_ABLATION_DONE.json` must already exist.
- No parameter may be selected or changed using test-dev.
- All three final workers should use the same GPU class, preferably NVIDIA T4.
- Do not inspect a partial test result and then alter the remaining systems.

The test set is accessed only after the protocol has been frozen.

## Shared result folder

All three accounts must resolve the same underlying Google Drive folder at:

```text
/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers
```

Use account sharing/shortcuts only where allowed by the provider. The purpose is collaborative execution of the same frozen experiment, not quota circumvention.

## 1. Mount Drive

Run on each Colab account:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Clone or update the repository

```python
from pathlib import Path
import subprocess

REPO = Path('/content/AC-MOT')
if REPO.exists():
    subprocess.run(['git','-C',str(REPO),'fetch','origin','main'], check=True)
    subprocess.run(['git','-C',str(REPO),'reset','--hard','origin/main'], check=True)
else:
    subprocess.run([
        'git','clone','https://github.com/AhmedCode110/AC-MOT.git',str(REPO)
    ], check=True)

subprocess.run(['git','-C',str(REPO),'log','-1','--oneline'], check=True)
```

## 3. Confirm T4 on all three accounts

```python
import torch
print(torch.cuda.get_device_name(0))
```

For comparable FPS, all three final workers should report a T4.

## 4. Freeze the final-test protocol once

Do this **after Worker 3 and the new A0-A3 validation ablation are complete**, but **before any account runs test-dev**.

```python
import os
os.environ['ACMOT_RESULT_ROOT'] = \
    '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'

%run /content/AC-MOT/scripts/prepare_final_test_3workers_protocol.py
```

This writes:

```text
FINAL_TEST_3WORKER_PROTOCOL.json
```

The preparation script does **not** access test-dev.

## 5. Run the three locked workers

### Account 1 — Baseline

```python
import os
os.environ['ACMOT_FINAL_WORKER_ID'] = '1'
os.environ['ACMOT_RESULT_ROOT'] = \
    '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'
os.environ['ACMOT_TEST_DIR'] = \
    '/content/drive/MyDrive/VisDrone2019-MOT-test-dev'

%run /content/AC-MOT/scripts/run_final_test_3workers_locked.py
```

System:

```text
Baseline_Default
static conf=0.25
static NMS IoU=0.45
static imgsz=640
default ByteTrack profile
```

### Account 2 — Historical Old AC-MOT

```python
import os
os.environ['ACMOT_FINAL_WORKER_ID'] = '2'
os.environ['ACMOT_RESULT_ROOT'] = \
    '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'
os.environ['ACMOT_TEST_DIR'] = \
    '/content/drive/MyDrive/VisDrone2019-MOT-test-dev'

%run /content/AC-MOT/scripts/run_final_test_3workers_locked.py
```

System:

```text
Old_ACMOT_Frozen
original heuristic SCI
original adaptive confidence/NMS/resolution mapping
W=7
stride=10
tuned ByteTrack profile
```

### Account 3 — New Frozen AC-MOT

```python
import os
os.environ['ACMOT_FINAL_WORKER_ID'] = '3'
os.environ['ACMOT_RESULT_ROOT'] = \
    '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'
os.environ['ACMOT_TEST_DIR'] = \
    '/content/drive/MyDrive/VisDrone2019-MOT-test-dev'

%run /content/AC-MOT/scripts/run_final_test_3workers_locked.py
```

System:

```text
New_ACMOT_Frozen
validation-selected temporal design
normalized optimized SCI weights
optimized confidence mapping
optimized NMS mapping
validation-selected resolution levels
optimized SCI switching thresholds
tuned ByteTrack profile
```

Each account writes one immutable worker result plus its TrackEval summary/timing files in the shared folder.

## 6. Merge only after all three workers finish

Run on any one account:

```python
import os
os.environ['ACMOT_RESULT_ROOT'] = \
    '/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers'

%run /content/AC-MOT/scripts/merge_final_test_3workers_locked.py
```

Expected final artifacts:

```text
FINAL_TEST_COMPARISON_3WORKER.csv
FINAL_TEST_RESULTS_3WORKER.json
FINAL_TEST_DONE.json
```

The merged comparison contains:

- MOTA
- HOTA
- IDF1
- IDS
- FN
- FP
- Processing FPS

for all three systems, plus deltas for:

- New AC-MOT minus Baseline
- New AC-MOT minus Old AC-MOT
- Old AC-MOT minus Baseline

## Final comparison table

The paper/thesis can then report the directly compatible held-out table:

| System | MOTA | HOTA | IDF1 | IDS | FPS |
|---|---:|---:|---:|---:|---:|
| Baseline Default | final | final | final | final | final |
| Old AC-MOT Frozen | final | final | final | final | final |
| New AC-MOT Frozen | final | final | final | final | final |

## Important final rule

Once any held-out results have been exposed, do not change the controller, detector settings, tracker settings, temporal settings, Optuna search, or selection rule and then present a later test-dev run as an unbiased final result.
