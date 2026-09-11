# AC-MOT Algorithm & Experimental Status — 2026-09-11

> **Frozen status date:** 2026-09-11  
> **Repository:** `AhmedCode110/AC-MOT`  
> **Branch:** `main`  
> **Purpose of this file:** give any new researcher, reviewer, supervisor, or AI agent enough context to understand exactly what AC-MOT is doing now, why the current experimental workflow exists, what is already implemented, and what must happen next without relying on previous chat history.

---

## 1. Current research goal

AC-MOT is being evaluated as an **adaptive multi-object tracking controller** around a fixed detector and fixed tracker.

The current thesis question is **not**:

> “Can we retrain YOLO or retune ByteTrack until the final score becomes higher?”

The current question is:

> “Can a lightweight Scene Complexity Index (SCI) adapt detector-side operating parameters online and improve tracking quality while keeping real-time speed, without retraining the detector or changing the tracker during the SCI study?”

Therefore, the current experimental design deliberately isolates the AC-MOT controller contribution.

---

## 2. Components that are fixed during the SCI study

### Detector

- Model: **YOLOv8n**
- Weights: pretrained `yolov8n.pt`
- Inference: FP16 on NVIDIA GPU when available
- The detector is **not retrained** during this experiment.

### Tracker

- Tracker family: **ByteTrack**
- Tracker configuration: fixed tuned ByteTrack profile already used by the project
- Tracker parameters are **not included in the SCI Optuna search**.

This is intentional. If detector-control parameters and ByteTrack association parameters were optimized together, it would become difficult to attribute any improvement specifically to AC-MOT.

The tuned ByteTrack values therefore require their own separate evidence / tracker-tuning experiment if a supervisor asks why a specific tracker value was chosen.

---

## 3. Evaluation protocol

Current optimization uses:

- `VisDrone2019-MOT-val`
- 7 validation sequences
- fixed YOLOv8n
- fixed tuned ByteTrack
- pinned TrackEval revision used by this repository
- real-time constraint: **processing FPS >= 25**

### Important protocol limitation

The repository currently evaluates with the project's **custom class-agnostic AC-MOT TrackEval protocol**.

It must **not** be described as the official VisDrone leaderboard preprocessing protocol.

Any report, paper, slide, or AI agent should preserve this distinction.

---

# 4. AC-MOT algorithm concept

At runtime, the system processes each frame approximately as follows:

```text
Input frame
   ↓
Scene analysis
   ↓
Scene cues
   ↓
Scene Complexity Index (SCI)
   ↓
Temporal smoothing
   ↓
Adaptive detector controller
   ├── Confidence
   ├── NMS IoU
   └── Input resolution
   ↓
YOLOv8n detections
   ↓
Fixed ByteTrack
   ↓
Tracked objects {ID, box}
```

The five SCI cues are conceptually:

```text
crowding
small/tiny objects
edge / scene complexity
low-light / night
blur
```

The important methodological change is that the final SCI coefficients and detector-control operating values should no longer be defended as hand-picked constants.

The current workflow is designed to replace arbitrary “magic numbers” with validation evidence.

---

# 5. Why the new experimental workflow exists

Earlier AC-MOT versions contained hand-selected values such as:

```text
SCI weights
confidence base/slope
NMS IoU base/slope
resolution values
SCI switching thresholds
smoothing window
analysis stride
```

A supervisor can correctly ask:

- Why this exact confidence?
- Why this exact NMS IoU?
- Why 640 / 736 / 832?
- Why smoothing window 7?
- Why analyze every 10 frames?
- Why these SCI weights?

The new workflow answers these questions using staged validation experiments.

The methodology is now:

```text
Scientific operating-point screening
        ↓
Temporal ablation
        ↓
Empirical joint Optuna optimization
        ↓
Freeze all parameters
        ↓
One held-out final test
```

---

# 6. Stage 1 — Scientific operating-point screening

Implemented in:

```text
scripts/scientific_operating_ablation_portable_colab.py
```

This stage has three independent sweeps.

The purpose is **not** to claim that one-factor-at-a-time screening finds the globally optimal system.

The purpose is to establish broad, defensible, validation-supported operating choices before joint optimization.

---

## 6.1 Resolution sweep

Full candidate set:

```text
512
544
576
608
640
672
704
736
768
800
832
864
896
928
960
```

Equivalent definition:

```text
512 → 960
step = 32 pixels
```

### Why step 32?

The values are regular, stride-compatible image sizes for the YOLO-style detector pipeline.

### What is fixed during this sweep?

```text
Confidence = 0.25
NMS IoU    = 0.70
YOLOv8n    = fixed
ByteTrack  = fixed
```

### CRITICAL: why 0.25 and 0.70 are used here

`Confidence = 0.25` and `NMS IoU = 0.70` are **not being declared as the final AC-MOT parameters**.

They are used only as **fixed static detector reference settings** while the resolution variable is changed.

The experimental logic is:

```text
Change one factor: resolution
Keep confidence fixed
Keep NMS IoU fixed
Keep tracker fixed
Keep detector weights fixed
```

This isolates the effect of input resolution during the screening stage.

Correct explanation to a supervisor:

> “Confidence 0.25 and NMS IoU 0.70 were used only as fixed detector reference settings to isolate the effect of input resolution. They were not assumed to be optimal AC-MOT parameters; confidence and NMS IoU were evaluated separately afterwards.”

### Metrics recorded for each resolution

```text
MOTA
HOTA
IDF1
IDS
FN
FP
FPS
processing latency
```

### Output of the resolution sweep

The code selects three distinct resolution operating levels:

```text
Low resolution
Medium resolution
High resolution
```

These do **not** have to be the old `640 / 736 / 832` values.

They are selected from validation results under the real-time constraint.

---

## 6.2 Confidence sweep

After the resolution screening, the code selects an anchor resolution and performs a confidence sweep.

Full candidate set:

```text
0.05
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
0.50
```

Equivalent definition:

```text
0.05 → 0.50
step = 0.05
```

During this sweep:

```text
Resolution = selected anchor resolution
NMS IoU    = 0.70 static reference
YOLOv8n    = fixed
ByteTrack  = fixed
```

The result is not necessarily one final confidence value.

The goal is to identify **empirically supported confidence values** that are safe candidates for the later joint search.

Current screening rule:

```text
FPS >= 25
and
MOTA >= static-reference MOTA
```

If too few values satisfy the rule, the script retains at least the strongest feasible values according to the documented ranking.

---

## 6.3 NMS IoU sweep

Full candidate set:

```text
0.30
0.35
0.40
0.45
0.50
0.55
0.60
0.65
0.70
0.75
0.80
```

Equivalent definition:

```text
0.30 → 0.80
step = 0.05
```

During this sweep:

```text
Resolution = selected anchor resolution
Confidence = selected confidence anchor
YOLOv8n    = fixed
ByteTrack  = fixed
```

The result is a set of **empirically supported NMS IoU values** for the later joint optimization.

Again, `0.70` is not assumed to be final simply because it was used as a static reference earlier.

---

## 6.4 Stage-1 output files

Expected files include:

```text
OPERATING_RESOLUTION_SWEEP.csv
OPERATING_CONFIDENCE_SWEEP.csv
OPERATING_NMS_SWEEP.csv
SCIENTIFIC_SEARCH_SPACE.json
OPERATING_ABLATION_REPORT.json
```

The most important file for the next stage is:

```text
SCIENTIFIC_SEARCH_SPACE.json
```

It records the validation-supported detector-control search space.

---

# 7. Stage 2 — Temporal ablation

Implemented in:

```text
scripts/temporal_ablation_portable_colab.py
```

The previous AC-MOT temporal design used:

```text
smoothing_window = 7
analysis_stride  = 10
```

Those exact numbers were originally engineering choices and therefore should not be presented as inherently optimal.

The new temporal ablation tests them experimentally.

### Smoothing-window candidates

```text
1, 3, 5, 7, 9
```

### Analysis-stride candidates

```text
1, 5, 10, 15, 20
```

Full grid:

```text
5 × 5 = 25 temporal configurations
```

The old setting:

```text
W = 7
S = 10
```

is retained as the reference configuration.

Current feasibility condition:

```text
FPS >= 25
and
IDS <= IDS of the W=7 / S=10 reference
```

Among feasible candidates, ranking is:

```text
1. highest MOTA
2. lower IDS
3. higher HOTA
4. higher IDF1
5. higher FPS
```

The full run writes:

```text
FROZEN_TEMPORAL_CONFIG.json
```

The selected window and stride are then passed to the joint SCI optimization stage.

---

# 8. Stage 3 — Empirical joint SCI optimization

Implemented in:

```text
scripts/optuna_sci_empirical_portable_colab.py
```

This stage consumes:

```text
SCIENTIFIC_SEARCH_SPACE.json
FROZEN_TEMPORAL_CONFIG.json
```

The joint optimization is where AC-MOT's adaptive controller is actually tuned.

## 8.1 Parameters learned by Optuna

Optuna learns the five non-negative SCI weights:

```text
weight_crowd
weight_tiny
weight_edge
weight_night
weight_blur
```

The weights are normalized so that:

```text
sum(weights) = 1
```

Optuna also learns the adaptive detector-control behavior, including:

```text
confidence behavior across SCI
NMS IoU behavior across SCI
SCI threshold for low → medium resolution
SCI threshold for medium → high resolution
```

The detector-control values are restricted to operating choices supported by Stage 1 rather than arbitrary old hand-picked values.

## 8.2 Important change: adaptation direction is learned

The final methodology does not blindly assume:

```text
SCI ↑ → confidence must always decrease
SCI ↑ → NMS IoU must always decrease
```

The joint search is allowed to determine which direction and magnitude works better inside the empirically supported operating space.

This is important because strong NMS suppression in crowded scenes can potentially remove valid overlapping detections and hurt identity continuity.

---

# 9. Scene-cue calibration and deployment realism

A previous prototype used validation ground-truth object counts to calibrate crowding.

That is not desirable for a deployable controller because ground truth does not exist at runtime.

The empirical portable optimizer is designed so that crowding calibration comes from **fixed-detector output counts**, not ground truth.

Therefore, the runtime controller uses information available from the detector itself.

Visual cues such as brightness, blur, and edge complexity are calibrated from validation-frame statistics.

The small-object boundary of `32 × 32` is retained as a documented small-object proxy rather than pretending that it was learned by Optuna.

---

# 10. Joint-search feasibility and selection rule

The joint Optuna stage uses validation only.

A candidate is considered feasible only if it satisfies the real-time / identity constraints defined by the script, including:

```text
FPS >= 25
IDS <= reference IDS constraint
```

Among feasible candidates, the intended ranking is:

```text
1. highest MOTA
2. lower IDS
3. higher HOTA
4. higher IDF1
5. higher FPS
```

This is deliberately not “maximize MOTA at any cost.”

The reason is that MOTA can improve even if identity switches become worse, because MOTA also depends strongly on false negatives and false positives.

---

# 11. Full validation experiment size

The current full workflow is approximately:

```text
Resolution sweep   = 15 runs
Confidence sweep   = 10 runs
NMS IoU sweep      = 11 runs
Temporal ablation  = 25 runs
Joint Optuna       = 50 trials
--------------------------------
Total              = 111 validation runs
```

Each operating-point / temporal run is evaluated across the validation sequences.

These are validation experiments, not final held-out test runs.

---

# 12. Smoke test versus full experiment

Before spending time on the full workflow, the code must first pass a smoke test.

The smoke test is only a pipeline check.

It must **not** be used as thesis evidence or as the final parameter-selection result.

## Smoke operating-point values

### Resolution

```text
640, 736, 832
```

### Confidence

```text
0.15, 0.25, 0.35
```

### NMS IoU

```text
0.45, 0.60, 0.75
```

### Temporal smoke grid

```text
Smoothing windows: 1, 7
Analysis strides:  1, 10
```

Therefore:

```text
2 × 2 = 4 temporal combinations
```

### Optuna smoke

```text
2 trials
```

The smoke test checks only that:

- the validation dataset can be found,
- YOLO loads,
- ByteTrack runs,
- TrackEval runs,
- output files are written,
- one stage correctly hands its results to the next,
- test-dev is not accessed.

---

# 13. Portable workflow entry point

Main portable runner:

```text
scripts/run_defensible_acmot_pipeline_colab.py
```

Its order is:

```text
Stage 1: scientific operating-point ablation
        ↓
Stage 2: temporal ablation
        ↓
Stage 3: empirical joint Optuna
```

All three stages are **validation-only**.

---

# 14. Current exact project status on 2026-09-11

As of this dated snapshot:

```text
[IMPLEMENTED] Portable operating-point ablation
[IMPLEMENTED] Portable temporal ablation
[IMPLEMENTED] Portable empirical joint Optuna
[IMPLEMENTED] One-command validation runner
[IMPLEMENTED] Separate frozen final-test runner

[NEXT] Run smoke test in Colab
[WAITING FOR] Smoke-test output / errors
[DO NOT YET] Start the full 111-run validation experiment until smoke passes
[DO NOT YET] Run test-dev final evaluation
```

The next action is therefore **not parameter interpretation** and **not final testing**.

The next action is simply:

> Run the portable smoke test and verify that the complete validation workflow executes successfully.

---

# 15. Exact smoke-test commands

## Cell 1 — mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 2 — clone or synchronize repository

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
        str(REPO)
    ], check=True)

subprocess.run(['git', '-C', str(REPO), 'log', '-1', '--oneline'], check=True)
```

## Cell 3 — run smoke test

```python
import os

os.environ['ACMOT_SMOKE_TEST'] = '1'
os.environ['ACMOT_OPTUNA_TRIALS'] = '2'

%run /content/AC-MOT/scripts/run_defensible_acmot_pipeline_colab.py
```

Expected logical order:

```text
Operating-point smoke
        ↓
Temporal smoke
        ↓
2-trial empirical joint Optuna smoke
        ↓
Smoke complete
        ↓
Test-dev not accessed
```

---

# 16. After the smoke test passes

Only after the smoke test is clean should the full validation workflow be launched.

Typical full-run setup:

```python
import os

os.environ['ACMOT_SMOKE_TEST'] = '0'
os.environ['ACMOT_OPTUNA_TRIALS'] = '50'

%run /content/AC-MOT/scripts/run_defensible_acmot_pipeline_colab.py
```

The full run should produce the validation-selected frozen configuration.

Before final testing, review:

- selected resolution levels,
- supported confidence values,
- supported NMS values,
- selected smoothing window,
- selected analysis stride,
- Optuna convergence,
- selected SCI weights,
- final validation MOTA / HOTA / IDF1 / IDS / FPS,
- reproducibility/audit outputs.

---

# 17. Freeze rule

After validation selection is complete, all selected parameters must be frozen.

Conceptually:

```text
Resolution levels
Confidence adaptation
NMS IoU adaptation
Smoothing window
Analysis stride
SCI weights
SCI switching thresholds
↓
FREEZE EVERYTHING
```

No parameter may be changed after inspecting held-out final-test performance if the final test is to remain scientifically defensible.

---

# 18. Final held-out test

Implemented in:

```text
scripts/run_frozen_final_test_portable_colab.py
```

This script is intentionally separate from the validation workflow.

It is meant to compare, in one locked final evaluation:

```text
Baseline_Default
vs
Full_ACMOT_Frozen
```

No tuning, searching, or parameter selection should happen in this stage.

The final-test script uses a lock file such as:

```text
FINAL_TEST_DONE.json
```

to prevent accidental repeated final-test use.

If this file already exists, inspect it before doing anything. Do not blindly delete it.

---

# 19. Optional future cross-dataset experiment

After the AC-MOT configuration has been frozen on VisDrone validation, a stronger generalization experiment can be performed on another dataset.

Strongest version:

```text
Calibrate / optimize once on VisDrone validation
        ↓
Freeze AC-MOT
        ↓
Run another dataset
        ↓
No YOLO retraining
No new Optuna
No parameter retuning
```

If performance remains useful, then the system can support a stronger statement such as:

> “The AC-MOT controller was calibrated once and transferred to an unseen target dataset without detector/tracker retraining or target-dataset parameter retuning.”

If a new dataset requires fresh validation calibration, then the correct statement is weaker but still useful:

> “Retraining-free, validation-calibrated adaptive MOT.”

Do not claim cross-dataset zero-retuning generalization until it has actually been tested.

---

# 20. Rules for any future AI agent working on this repository

A new agent should follow these rules unless the researcher explicitly changes the experimental protocol:

1. **Do not retrain YOLOv8n during the SCI experiment.**
2. **Do not optimize ByteTrack together with SCI.** Keep tracker contribution isolated.
3. **Do not use test-dev to choose parameters.**
4. **Do not call the current evaluation the official VisDrone leaderboard protocol.**
5. **Do not interpret smoke-test winners as final scientific results.**
6. **Do not hard-code the old 640/736/832 values after the new resolution sweep exists.**
7. **Do not hard-code old confidence or NMS values after the empirical screens exist.**
8. **Do not use ground-truth object counts as a runtime crowding requirement.** Runtime SCI must use deployable cues.
9. **Do not change frozen parameters after final-test inspection.**
10. **Preserve all CSV / JSON / convergence / audit outputs.** They are evidence for thesis defense.
11. **Keep fixed-hardware comparisons clearly identified.** FPS from a non-T4 GPU is not directly comparable with prior T4 measurements.
12. **Explain every important number using one of four sources:** standard definition, validation ablation, validation statistics, or optimization.

---

# 21. Short methodology summary for a supervisor

The full methodology can be summarized as:

> “The detector and tracker were held fixed to isolate the AC-MOT contribution. Detector-control operating points were first screened on validation data using controlled resolution, confidence, and NMS IoU experiments. Temporal parameters were then selected through a dedicated smoothing-window and analysis-stride ablation. Using only the empirically supported operating space, joint Optuna optimization selected the SCI cue weights and adaptive detector-control behavior subject to real-time and identity-stability constraints. All parameters were then frozen before a one-shot held-out final evaluation.”

---

# 22. One-line current status

**Current status on 2026-09-11:** the defensible portable AC-MOT validation pipeline is implemented; the immediate next step is to run and verify the smoke test before launching the full validation experiment.
