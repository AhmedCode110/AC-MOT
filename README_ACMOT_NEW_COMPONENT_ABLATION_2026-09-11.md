# New Optimized AC-MOT Component Ablation

This is a **validation-only** attribution study for the frozen optimized AC-MOT controller.

Run it **after Worker 3 finishes** and creates `FROZEN_DEFENSIBLE_ACMOT_CONFIG.json`, but **before** the one-shot held-out final test.

## Why this ablation exists

The old results should remain in the thesis/presentation as the historical research path. This new ablation answers a different question:

> Which adaptive detector-control components contribute to the final optimized AC-MOT result when the detector and tracker are held fixed?

YOLOv8n and tuned ByteTrack are fixed across A0-A3 so the controller contribution is not confounded by tracker changes.

## Systems

| Stage | Definition |
|---|---|
| **A0 — Study Baseline** | Static validation-selected anchor: fixed confidence, fixed NMS IoU, fixed resolution. |
| **A1 — + Adaptive Confidence** | New empirical SCI + frozen temporal design + optimized confidence mapping. NMS and resolution remain fixed. |
| **A2 — + Adaptive NMS** | A1 + optimized NMS IoU mapping. Resolution remains fixed. |
| **A3 — Full New AC-MOT** | A2 + adaptive resolution using the three validation-selected resolution levels and optimized SCI thresholds. |
| **HIST — Old AC-MOT** | Original heuristic controller with W=7 / stride=10. This is a contextual historical comparator and is **not** part of the A0-A3 numbering. |

This nested structure avoids making `A3` the old system. `A3` is always the most complete new system.

## Important protocol rule

This script uses **VisDrone validation only** and never accesses test-dev.

It also refuses to run if `FINAL_TEST_DONE.json` already exists. This is deliberate: the ablation should be completed before exposing the held-out test, so later analysis cannot become post-test retuning.

## Run command

Use the same shared result folder used by the 3-worker pipeline:

```python
import os

os.environ["ACMOT_RESULT_ROOT"] = \
    "/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers"

%run /content/AC-MOT/scripts/new_acmot_component_ablation_validation.py
```

By default the historical old AC-MOT comparator is included. To omit it:

```python
os.environ["ACMOT_INCLUDE_HISTORICAL"] = "0"
```

Do not set `ACMOT_FORCE_NEW_ABLATION=1` unless an intentional rerun is scientifically justified.

## Expected outputs

The shared result folder will receive:

```text
NEW_ACMOT_COMPONENT_ABLATION.csv
NEW_ACMOT_COMPONENT_ABLATION_REPORT.json
NEW_ACMOT_COMPONENT_ABLATION_DONE.json
NEW_ABLATION_A0.json
NEW_ABLATION_A1.json
NEW_ABLATION_A2.json
NEW_ABLATION_A3.json
NEW_ABLATION_HIST.json          # when historical comparator is enabled
```

The CSV reports:

- MOTA
- HOTA
- IDF1
- IDS
- FN
- FP
- FPS
- p95 latency
- mean resolution
- mean confidence
- mean NMS IoU
- real-time >=25 FPS status
- deltas versus A0

## Timing fairness

The script explicitly warms the exact resolution(s) requested by each system before measured inference.

For A0, unused SceneAnalyzer visual computation is disabled because the baseline controller is static.

For A1-A3, empirical SCI visual analysis remains included because it is part of the real adaptive controller cost.

## Presentation story

Keep the old experimental sequence as the historical research journey:

```text
Baseline -> Tuned ByteTrack -> Original AC-MOT -> IDS problem -> optimization motivation
```

Then show the new attribution study separately:

```text
A0 Static
  -> A1 + Adaptive Confidence
  -> A2 + Adaptive NMS
  -> A3 + Adaptive Resolution / Full New AC-MOT
```

Finally compare:

```text
Historical Old AC-MOT vs A3 Full New AC-MOT
```

The new ablation demonstrates component contribution. The old results demonstrate how the research evolved. They serve different purposes and should not be mixed into one numbering scheme.

## Final step

After reviewing the validation-only ablation, do **not** retune A3 from these component results. The optimized configuration was already frozen by Worker 3.

Then run the one-shot held-out test:

```text
Baseline_Default vs Full_ACMOT_Frozen
```

using `scripts/run_frozen_final_test_portable_colab.py`.
