# AC-MOT Complete Method & Decision Log — 2026-09-11

> **Status date:** 2026-09-11  
> **Repository:** `AhmedCode110/AC-MOT`  
> **Branch:** `main`  
> **Purpose:** this is the canonical continuation document for a new researcher, supervisor, reviewer, or AI agent. It records not only the current algorithm, but also *why* the current methodology exists, what earlier design choices were challenged, what pilot evidence motivated the redesign, exactly what each validation stage does, what is fixed, what is optimized, what is still unresolved, and what the next action must be.

---

# 1. Current thesis question

AC-MOT is being treated as a lightweight adaptive controller around a fixed detector and fixed tracker.

The current scientific question is:

> Can a Scene Complexity Index (SCI) adapt detector-side operating parameters online and improve multi-object tracking quality while preserving real-time speed, without retraining the detector and without changing the tracker during the SCI study?

The experiment is **not** allowed to improve the final result by silently retraining YOLO or co-optimizing ByteTrack with SCI. That would confound the source of the improvement.

---

# 2. Fixed components during the SCI study

## 2.1 Detector

- Detector: `YOLOv8n`
- Weights: pretrained `yolov8n.pt`
- FP16 on CUDA when available
- Detector weights remain fixed
- No detector retraining during the SCI study

The detector is deliberately kept small because the thesis contribution is the adaptive controller rather than a larger detector architecture.

## 2.2 Tracker

Tracker family: `ByteTrack`.

The project currently contains two tracker profiles in `core_v17.py`:

### Default profile

```text
track_high_thresh = 0.25
track_low_thresh  = 0.10
new_track_thresh  = 0.25
track_buffer       = 30
match_thresh       = 0.80
fuse_score         = True
```

### Tuned profile used during the SCI study

```text
track_high_thresh = 0.18
track_low_thresh  = 0.04
new_track_thresh  = 0.20
track_buffer       = 45
match_thresh       = 0.86
fuse_score         = True
```

These tuned ByteTrack values are **frozen during SCI optimization**.

Important attribution rule:

> ByteTrack parameters must not be optimized in the same Optuna search as SCI, because then any improvement could come from tracker retuning rather than AC-MOT.

If a supervisor asks why `match_thresh=0.86`, `buffer=45`, etc. were chosen, the answer must come from a **separate tracker-tuning / tracker-ablation experiment**. The current SCI experiment does not justify those tracker numbers by itself.

---

# 3. Current evaluation protocol

Current validation split:

```text
VisDrone2019-MOT-val
7 sequences
```

Current final held-out split:

```text
VisDrone2019-MOT-test-dev
17 sequences
```

Current real-time requirement:

```text
processing FPS >= 25
```

The current defensible workflow uses **25 FPS as the single active requirement**. Older project configs may contain a 20-FPS acceptable threshold, but that must not be mixed into the new methodology unless the protocol is intentionally changed and documented.

TrackEval revision is pinned by the scripts to:

```text
12c8791b303e0a0b50f753af204249e622d0281a
```

## 3.1 Current ground-truth research filter

The repository evaluation is custom and class-agnostic. The current filter used by the new scripts is:

```text
categories      = [1, 4, 5, 6, 9]
score           = 1
occlusion       < 2
truncation      < 2
```

## 3.2 Important protocol limitation

The current evaluator is **not the official VisDrone leaderboard preprocessing protocol**.

Correct wording:

> Custom class-agnostic AC-MOT TrackEval research protocol.

Incorrect wording:

> Official VisDrone benchmark result.

This distinction must be preserved in the thesis, slides, code comments, and future AI-agent work.

## 3.3 Class-space caution that must not be forgotten

The fixed COCO YOLO detector uses classes:

```text
[0, 2, 5, 7]
= person, car, bus, truck
```

The custom VisDrone GT filter includes category IDs `[1,4,5,6,9]` and is class-agnostic. The repository historically included a VisDrone `van` category in the filtered GT while YOLOv8n does not have a distinct `van` output class.

This is a methodological point that can affect false negatives. Therefore:

- do not describe this custom protocol as an official 5-class VisDrone detector benchmark;
- if publication-quality class-specific evaluation is later required, revisit the class mapping explicitly;
- do not silently change the GT filter in the middle of the current experiment.

---

# 4. Runtime AC-MOT concept

The intended runtime flow is:

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
   ├── confidence
   ├── NMS IoU
   └── input resolution
   ↓
Fixed YOLOv8n
   ↓
Fixed ByteTrack
   ↓
{id, box}
```

The five scene cues are:

```text
crowding
tiny-object fraction
edge / scene complexity
low-light / night
blur
```

The new methodology retains this concept but replaces hand-picked constants with validation evidence wherever possible.

---

# 5. Historical SCI design and the exact “magic numbers” that triggered the redesign

The previous presentation-aligned controller in `core_v17.py` used the following scene analyzer.

## 5.1 Previous cue definitions

Let `previous` be the previous detector boxes.

### Crowding

```text
n = number of previous boxes
crowd = min(n / 30, 1)
```

### Tiny-object cue

A box is considered tiny when:

```text
box area < 32 × 32 pixels
```

and:

```text
tiny = fraction of previous boxes satisfying that condition
```

### Edge complexity

The old code used the visual edge measure and normalized it by:

```text
edge_normalized = min(edge / 0.14, 1)
```

### Night

```text
brightness < 80
```

### Blur

```text
Laplacian variance < 180
```

## 5.2 Previous SCI formula

The old raw SCI was approximately:

```text
SCI_raw =
    0.30 * crowd
  + 0.30 * tiny
  + 0.20 * normalized_edge
  + 0.10 * night
  + 0.05 * blur
```

Important detail:

```text
0.30 + 0.30 + 0.20 + 0.10 + 0.05 = 0.95
```

Therefore the old coefficients are **not a normalized weighted average summing to one**. They were heuristic design priorities.

The old design then clipped the value and smoothed it with a rolling mean.

## 5.3 Previous temporal design

```text
smoothing_window = 7
analysis_stride  = 10
```

Those exact values were engineering choices, not learned values.

At a nominal 30-FPS tracker clock:

```text
stride 10 ≈ one scene analysis every 10/30 = 0.333 s
```

For `window=7`, the oldest-to-newest sampled history spans approximately:

```text
(window - 1) * stride / 30
= 6 * 10 / 30
≈ 2.0 s
```

This gives an engineering interpretation, but it does **not** prove that `7` and `10` are optimal. That is why a temporal ablation was added.

## 5.4 Previous SmartCalibrator mapping

The old adaptive mapping was:

```text
conf = 0.245 - 0.050 * SCI
nms  = 0.490 - 0.050 * SCI
```

Additional scene offsets were:

```text
if scene in {crowded, tiny, night}:
    conf -= 0.012

if scene == blur:
    nms -= 0.012
```

Then clipping:

```text
conf ∈ [0.19, 0.28]
nms  ∈ [0.40, 0.52]
```

Old resolution switching:

```text
if SCI > 0.60 or tiny > 0.50:
    resolution = 832
elif SCI > 0.35 or scene in {crowded, tiny}:
    resolution = 736
else:
    resolution = 640
```

Other old scene thresholds included values such as:

```text
tiny > 0.50
crowd > 0.65
edge > 0.13
brightness < 80
blur < 180
```

These values are exactly the kind of numbers a strict supervisor can ask about:

- Why `0.245`?
- Why slope `0.050`?
- Why `0.490`?
- Why `0.012`?
- Why `0.35` and `0.60`?
- Why `640 / 736 / 832`?
- Why `7 / 10`?
- Why those SCI weights?

The new experiment exists to avoid defending those exact values as if they were inherently optimal.

---

# 6. Important NMS interpretation that changed the methodology

For standard NMS:

- a **lower NMS IoU threshold** means stronger suppression;
- a **higher NMS IoU threshold** allows more overlapping boxes to survive.

The old controller reduced NMS IoU as SCI increased.

In a crowded scene, stronger suppression can remove real overlapping objects, which can create missed detections, broken tracks, and new IDs.

Therefore the new design does **not** assume that:

```text
SCI ↑  =>  NMS IoU must decrease
```

The direction of NMS adaptation is learned by the joint optimizer.

The same is true for confidence: the final method does not blindly assume a particular monotonic direction before validation.

---

# 7. Why MOTA alone is not enough

A key observation from the project is that MOTA may improve while identity switches get worse.

Conceptually:

```text
MOTA ≈ 1 - (FN + FP + IDSW) / GT
```

If false negatives or false positives decrease enough, MOTA can rise even while `IDS` increases.

Therefore the current selection strategy does **not** simply maximize MOTA without constraints.

Identity stability is explicitly constrained using IDS, and HOTA / IDF1 are also reported.

---

# 8. Historical pilot evidence that motivated the redesign

These are **pilot / development results**, not final thesis results.

## 8.1 Two-trial SCI-weights-only pilot

Old A3 validation reference in that pilot:

```text
MOTA = 18.164674%
HOTA = 33.064013%
IDF1 = 36.296242%
IDS  = 271
FPS  ≈ 48.04
```

Trial 000:

```text
MOTA = 18.256738%
IDS  = 281
FPS  ≈ 47.75
```

Trial 001:

```text
MOTA = 18.237434%
IDS  = 282
FPS  ≈ 48.00
```

Interpretation:

> Optimizing SCI weights alone slightly improved MOTA but increased identity switches.

This was the main reason to stop treating SCI-weight optimization alone as sufficient.

## 8.2 Later two-trial joint smoke pilot

A later joint smoke test showed that lowering IDS was possible, but quality trade-offs remained.

Old A3 reference in that run:

```text
MOTA = 18.164674%
IDS  = 271
FPS  ≈ 49.67
```

Trial 000:

```text
MOTA = 17.605%
HOTA = 32.331%
IDF1 = 35.772%
IDS  = 250
FPS  ≈ 47.00
```

Trial 001:

```text
MOTA = 18.168%
HOTA = 32.301%
IDF1 = 35.088%
IDS  = 234
FPS  ≈ 49.71
```

Interpretation:

- identity switches can improve while HOTA/IDF1 decline;
- two smoke trials are nowhere near enough to claim a final optimum;
- the optimization method therefore needs explicit feasibility constraints and a larger declared trial budget.

These pilot results must never be presented as the final selected thesis configuration.

---

# 9. New defensible workflow — high-level order

The current methodology is:

```text
Fixed YOLOv8n + Fixed tuned ByteTrack
        ↓
Stage 1: Resolution / Confidence / NMS screening
        ↓
Stage 2: Temporal ablation
        ↓
Stage 3: Empirical joint Optuna
        ↓
Freeze every selected parameter
        ↓
One held-out final test
        ↓
Optional cross-dataset transfer test
```

The full validation workflow is launched by:

```text
scripts/run_defensible_acmot_pipeline_colab.py
```

All three tuning stages are validation-only.

---

# 10. Stage 1 — scientific operating-point screening

Implementation:

```text
scripts/scientific_operating_ablation_portable_colab.py
```

The stage performs three controlled sweeps:

1. Resolution sweep
2. Confidence sweep
3. NMS IoU sweep

This is a **one-factor-at-a-time screening stage**, not the final joint optimization.

Why use one-factor-at-a-time screening first?

- it makes each effect interpretable to a supervisor;
- it provides an empirically supported region for the later joint search;
- it avoids giving Optuna completely arbitrary hand-written bounds.

Limitation:

> One-factor-at-a-time screening can miss interactions between variables.

That limitation is why Stage 3 later performs joint optimization.

---

# 11. Stage 1A — resolution sweep

## 11.1 Full candidate values

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

Equivalent form:

```text
512 → 960
step = 32
```

Total:

```text
15 full resolution runs
```

Each run is evaluated over all 7 validation sequences.

## 11.2 Why step 32?

The candidate sizes are regular, stride-compatible image sizes for the YOLO-style pipeline.

## 11.3 Why the range 512–960?

This range is a **predeclared screening range**, not a learned result.

Its purpose is to cover:

- values below the standard 640 operating point;
- the standard 640 reference;
- values clearly above 640 to trade more compute for more detail;
- a range that is still intended to be compatible with the real-time requirement.

The exact outer bounds `512` and `960` are design bounds for the screening budget. They must **not** be described as data-derived optimal values.

If a future supervisor requires even stronger justification of the outer bounds, a separate speed-feasibility pre-scan could be introduced, but that is not part of the currently frozen 2026-09-11 workflow.

## 11.4 What stays fixed during the resolution sweep?

```text
Confidence = 0.25
NMS IoU    = 0.70
YOLOv8n weights = fixed
ByteTrack       = fixed tuned profile
```

## 11.5 Critical explanation of 0.25 and 0.70

These two numbers are **reference settings only**.

They are *not* being claimed as the final AC-MOT confidence or NMS values.

The purpose is controlled isolation:

```text
change resolution only
hold confidence constant
hold NMS IoU constant
hold detector constant
hold tracker constant
```

Correct supervisor answer:

> “Confidence 0.25 and NMS IoU 0.70 were used only as fixed static detector reference settings to isolate the effect of input resolution. They were not assumed to be optimal AC-MOT parameters; confidence and NMS IoU were evaluated separately afterwards.”

This point is essential and must not be lost.

## 11.6 Metrics recorded per resolution

```text
MOTA
HOTA
DetA
AssA
IDF1
IDS
FN
FP
processing FPS
processing p95 latency
```

## 11.7 Exact resolution-selection logic in the current code

The code first keeps feasible candidates satisfying the active FPS gate when possible.

Then it forms three distinct operating levels:

- a low-compute feasible level;
- a high-quality feasible level using the documented quality key;
- a distinct/intermediate high-quality level.

The quality key used by Stage 1 is effectively:

```text
higher MOTA
then lower IDS
then higher HOTA
then higher IDF1
then higher FPS
```

The three selected resolution levels are sorted ascending and later interpreted as:

```text
low / medium / high
```

They do not have to be `640 / 736 / 832`.

---

# 12. Stage 1B — confidence sweep

After the resolution sweep, the script selects an anchor resolution and tests confidence values:

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

Equivalent:

```text
0.05 → 0.50
step = 0.05
```

Total:

```text
10 full confidence runs
```

During this stage:

```text
Resolution = selected anchor resolution
NMS IoU    = 0.70 reference
YOLOv8n    = fixed
ByteTrack  = fixed
```

The purpose is **not** necessarily to choose one final confidence value. It is to identify a supported set of confidence operating values for the later joint optimizer.

## 12.1 Exact current support rule

The code finds the row nearest the static reference value `0.25` and uses its MOTA as the screening reference.

A confidence value is supported if:

```text
FPS >= 25
and
MOTA >= reference-row MOTA
```

If fewer than 3 values satisfy the rule, the script retains the top 3 feasible values using the same quality ordering:

```text
higher MOTA
lower IDS
higher HOTA
higher IDF1
higher FPS
```

The strongest supported confidence is also used as the anchor for the following NMS sweep.

---

# 13. Stage 1C — NMS IoU sweep

Candidate values:

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

Equivalent:

```text
0.30 → 0.80
step = 0.05
```

Total:

```text
11 full NMS runs
```

During this stage:

```text
Resolution = selected anchor resolution
Confidence = selected confidence anchor
YOLOv8n    = fixed
ByteTrack  = fixed
```

The support rule mirrors the confidence screen:

```text
FPS >= 25
and
MOTA >= MOTA at the reference NMS row nearest 0.70
```

If fewer than 3 values survive, keep the top 3 feasible values by the documented quality key.

The result is a set of supported NMS IoU values for the joint optimizer.

Again, `0.70` is a screening reference, not a forced final AC-MOT value.

---

# 14. Stage-1 outputs

Full-run outputs include:

```text
OPERATING_RESOLUTION_SWEEP.csv
OPERATING_CONFIDENCE_SWEEP.csv
OPERATING_NMS_SWEEP.csv
SCIENTIFIC_SEARCH_SPACE.json
OPERATING_ABLATION_REPORT.json
```

`SCIENTIFIC_SEARCH_SPACE.json` is the key machine-readable output consumed by Stage 3.

It records:

- all screened resolution values;
- selected three resolution levels;
- all screened confidence values;
- supported confidence values;
- all screened NMS values;
- supported NMS values;
- the anchor resolution/confidence/NMS used for detector-derived cue calibration;
- the active FPS gate;
- the selection rules;
- the custom-protocol warning.

---

# 15. Stage 2 — temporal ablation

Implementation:

```text
scripts/temporal_ablation_portable_colab.py
```

The purpose is to answer two supervisor questions experimentally:

1. Why analyze the scene every N frames?
2. Why average SCI over W measurements?

The old values `W=7`, `S=10` are retained as the reference but are not assumed optimal.

## 15.1 Candidate smoothing windows

```text
1, 3, 5, 7, 9
```

Interpretation:

- `1` = no rolling smoothing;
- larger values = more temporal averaging and potentially less oscillation;
- too much smoothing can introduce control lag.

## 15.2 Candidate analysis strides

```text
1, 5, 10, 15, 20
```

At nominal 30 FPS, these correspond to approximately:

```text
1  → 0.033 s
5  → 0.167 s
10 → 0.333 s
15 → 0.500 s
20 → 0.667 s
```

## 15.3 Full temporal grid

```text
5 windows × 5 strides = 25 configurations
```

The reference `W=7 / S=10` is evaluated first so its IDS can be used as the identity-stability gate.

## 15.4 Temporal feasibility rule

```text
FPS >= 25
and
IDS <= IDS of W=7 / S=10 reference
```

Among feasible candidates:

```text
1. highest MOTA
2. lower IDS
3. higher HOTA
4. higher IDF1
5. higher FPS
```

## 15.5 Output

The full temporal stage writes:

```text
FROZEN_TEMPORAL_CONFIG.json
```

The selected values are then read by the full joint optimizer.

Important smoke behavior:

> Temporal smoke mode does **not** freeze a temporal configuration. Therefore the joint Optuna smoke stage intentionally falls back to `W=7 / S=10` only to test pipeline execution. A smoke-stage temporal “winner” is not passed forward as a scientific result.

This distinction matters.

---

# 16. Stage 3 — empirical joint Optuna

Implementation:

```text
scripts/optuna_sci_empirical_portable_colab.py
```

Inputs:

```text
SCIENTIFIC_SEARCH_SPACE.json
FROZEN_TEMPORAL_CONFIG.json   # required for full run
```

The full joint optimizer uses:

```text
TPESampler(seed=42)
```

Default full budget:

```text
50 trials
```

The number `50` is a declared computational budget, not a theoretically guaranteed optimum. Convergence evidence must be reviewed. If the best curve is still improving strongly at the end, the researcher may later justify a larger predeclared budget in a new experiment, but must not secretly extend only after seeing test results.

---

# 17. Exact parameterization learned by Optuna

## 17.1 SCI weights

Raw values are sampled for:

```text
crowd
tiny
edge
night
blur
```

with raw sampling in `[0,1]`, then normalized:

```text
weight_k = raw_k / sum(raw)
```

so the final weights satisfy:

```text
weights >= 0
sum(weights) = 1
```

## 17.2 Confidence adaptation

The optimizer selects:

```text
conf_easy
conf_hard
```

from the **supported confidence values produced by Stage 1**.

Runtime interpolation is linear in SCI:

```text
conf(SCI) = conf_easy + SCI * (conf_hard - conf_easy)
```

Because `conf_easy` and `conf_hard` are both learned, the direction is not assumed in advance.

## 17.3 NMS adaptation

Similarly:

```text
nms_easy
nms_hard
```

are selected from the supported NMS values.

Runtime interpolation:

```text
NMS(SCI) = nms_easy + SCI * (nms_hard - nms_easy)
```

Again, the direction can increase or decrease with SCI depending on validation results.

## 17.4 Resolution switching

The three resolution levels come from Stage 1:

```text
r0 < r1 < r2
```

Optuna samples two continuous values in the mathematical SCI domain `[0,1]`, sorts them, and forms:

```text
threshold_mid
threshold_high
```

Runtime rule:

```text
if SCI >= threshold_high:
    resolution = r2
elif SCI >= threshold_mid:
    resolution = r1
else:
    resolution = r0
```

Thus the *resolution values* come from ablation, while the *SCI switching points* are learned jointly.

---

# 18. New scene-cue calibration — no GT crowd-count leakage

This was an important methodological correction.

An earlier prototype calibrated crowding using validation GT object counts. That is not deployment-realistic because GT does not exist at runtime.

The new empirical optimizer calibrates crowding from **fixed YOLOv8n detector output counts**.

## 18.1 Calibration anchor

The detector-derived calibration uses the anchor operating point stored in Stage 1:

```text
anchor resolution
anchor confidence
anchor NMS IoU
```

and fixed detector classes:

```text
[0, 2, 5, 7]
```

## 18.2 Visual cue extraction

For sampled validation frames, the code:

1. downsamples the image to `0.25 ×` scale;
2. converts to grayscale;
3. records mean brightness;
4. records Laplacian variance as blur/sharpness information;
5. computes Sobel X/Y gradients and mean gradient magnitude as edge complexity;
6. runs the fixed YOLO detector at the calibration anchor and records the number of detector boxes as the crowd-count sample.

The resulting sorted empirical distributions are saved to:

```text
DETECTOR_DERIVED_CUE_CALIBRATION.json
```

## 18.3 Runtime cue normalization

At runtime:

```text
crowd = empirical percentile rank of current previous-box count
edge  = empirical percentile rank of edge statistic
night = 1 - empirical percentile rank of brightness
blur  = 1 - empirical percentile rank of Laplacian variance
```

The tiny cue remains:

```text
fraction of previous detector boxes with area < 32 × 32
```

The `32×32` boundary is intentionally documented as an external small-object proxy; it is not falsely claimed to be learned by Optuna.

---

# 19. Temporal SCI behavior in the new controller

SCI is recomputed only on analysis frames:

```text
frame == 1
or
(frame - 1) % analysis_stride == 0
```

Between analysis frames, the latest SCI is reused.

On an analysis frame:

1. obtain deployable scene cues;
2. compute weighted raw SCI;
3. clip SCI to `[0,1]`;
4. append to the rolling history;
5. use the mean of the history as smoothed SCI.

This keeps the runtime lightweight and avoids frame-to-frame parameter oscillation, while the temporal ablation determines how much smoothing/subsampling is justified.

---

# 20. Joint-search feasibility and selection rule

The full Optuna trial evaluates MOTA, IDS, and FPS as a multi-objective study.

A trial is marked feasible when:

```text
FPS >= 25
and
IDS <= old-A3 reference IDS
```

Important improvement over an earlier implementation:

> The old-A3 cache is keyed by the selected temporal parameters:

```text
OLD_A3_VALIDATION_W{window}_S{stride}.json
```

This prevents accidentally reusing an IDS reference measured with a different smoothing window or analysis stride.

Among feasible trials, final selection is:

```text
1. highest MOTA
2. lower IDS
3. higher HOTA
4. higher IDF1
5. higher FPS
```

The selected validation configuration is written to:

```text
FROZEN_DEFENSIBLE_ACMOT_CONFIG.json
```

---

# 21. Optuna resume and reproducibility behavior

The empirical optimizer uses SQLite persistence.

Local DB:

```text
/content/acmot_empirical.db
```

Drive copy:

```text
EMPIRICAL_OPTUNA.db
```

After each trial, the local DB is copied to Drive.

Therefore a later Colab session/account can continue from completed trials rather than restarting from zero, provided the result folder is preserved.

A study signature hashes:

- the Stage-1 search space;
- selected temporal values;
- random seed;
- FPS gate;
- algorithm version.

If the search space changes, the script refuses to silently resume the old study unless reset is explicitly requested.

This protects reproducibility.

---

# 22. Full experiment count

The main declared validation workload is:

```text
Resolution sweep   = 15
Confidence sweep   = 10
NMS sweep          = 11
Temporal ablation  = 25
Joint Optuna       = 50
----------------------
Declared total     = 111 validation configurations/trials
```

Each full operating-point / temporal configuration runs across all 7 validation sequences.

For the first 36 operating-point runs alone:

```text
36 × 7 = 252 sequence evaluations
```

Important nuance:

> `111` is the headline count of planned configurations/trials. Actual detector executions can be larger because cue calibration, old-A3 reference evaluation, warmups, cached/reused runs, and metric evaluation steps also occur.

Do not misrepresent `111` as the exact count of every low-level model invocation.

---

# 23. Smoke test versus full experiment

The smoke test is only for verifying the pipeline before spending GPU time.

It is **not thesis evidence** and must not define final parameters.

## 23.1 Smoke operating values

Resolution:

```text
640, 736, 832
```

Confidence:

```text
0.15, 0.25, 0.35
```

NMS:

```text
0.45, 0.60, 0.75
```

Temporal:

```text
windows = [1, 7]
strides = [1, 10]
=> 4 combinations
```

Joint Optuna smoke:

```text
2 trials
```

Approximate smoke configuration count:

```text
3 + 3 + 3 + 4 + 2 = 15
```

## 23.2 What smoke proves

Smoke checks:

- Drive mount/path logic works;
- validation dataset is found;
- YOLO loads;
- ByteTrack runs;
- TrackEval runs;
- result files are written;
- the three scripts can execute in sequence;
- the Stage-1 smoke search-space file can be consumed by the Optuna smoke stage;
- test-dev is not accessed.

## 23.3 What smoke does NOT prove

Smoke does not prove:

- any smoke winner is scientifically optimal;
- the temporal smoke winner should be used later;
- 2 Optuna trials are sufficient;
- final test performance.

---

# 24. Exact current next action

As of 2026-09-11, the code is implemented and the **next action is the smoke test**.

Do not run the full validation workflow until smoke passes cleanly.

Do not run the final test yet.

## Colab Cell 1

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Colab Cell 2

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

## Colab Cell 3

```python
import os

os.environ['ACMOT_SMOKE_TEST'] = '1'
os.environ['ACMOT_OPTUNA_TRIALS'] = '2'

%run /content/AC-MOT/scripts/run_defensible_acmot_pipeline_colab.py
```

Expected stage order:

```text
Stage 1 — resolution/confidence/NMS smoke
Stage 2 — temporal smoke
Stage 3 — empirical joint Optuna smoke
```

---

# 25. Full run after smoke passes

Only after smoke is reviewed:

```python
import os

os.environ['ACMOT_SMOKE_TEST'] = '0'
os.environ['ACMOT_OPTUNA_TRIALS'] = '50'

%run /content/AC-MOT/scripts/run_defensible_acmot_pipeline_colab.py
```

Before any final test, inspect:

- resolution sweep table;
- selected resolution levels;
- confidence sweep table;
- supported confidence values;
- NMS sweep table;
- supported NMS values;
- temporal full table;
- selected smoothing window and stride;
- cue calibration summary;
- Optuna trial table;
- feasible-trial count;
- convergence behavior;
- parameter importance if available;
- selected SCI weights;
- selected confidence endpoints;
- selected NMS endpoints;
- selected resolution thresholds;
- validation MOTA/HOTA/IDF1/IDS/FPS;
- study signature and audit files.

---

# 26. Freeze rule

After validation selection:

```text
resolution levels
supported confidence/NMS choices
selected temporal values
SCI weights
confidence endpoints
NMS endpoints
SCI resolution thresholds
cue calibration
tracker profile
model weights
protocol
```

must be frozen.

After held-out test results are exposed, no parameter may be changed and then presented as if the later test were still unbiased.

---

# 27. One-shot held-out final test

Implementation:

```text
scripts/run_frozen_final_test_portable_colab.py
```

It requires:

```text
FROZEN_DEFENSIBLE_ACMOT_CONFIG.json
DETECTOR_DERIVED_CUE_CALIBRATION.json
```

It compares in the same locked run:

```text
Baseline_Default
vs
Full_ACMOT_Frozen
```

The baseline uses the repository default tracker profile and fixed detector settings from the baseline system.

The frozen AC-MOT system uses the frozen tuned tracker plus the frozen empirical controller.

No tuning occurs on test.

When complete, the script creates:

```text
FINAL_TEST_DONE.json
FINAL_TEST_RESULTS.json
```

If `FINAL_TEST_DONE.json` already exists, the script refuses to rerun.

Do not blindly delete that file. If a held-out result has already been exposed, the scientific status of the test set has changed.

---

# 28. Historical verified 17-sequence result — keep separate from the new pipeline

Before this new defensible re-calibration workflow, the project had a verified 17-sequence comparison used in the presentation.

Historical values:

### Baseline_Default

```text
MOTA = 19.718
HOTA = 28.418
IDF1 = 32.716
IDS  = 1238
FPS  = 44.181
```

### Historical Full AC-MOT / TRK_NEW24_MATCH88

```text
MOTA = 22.999
HOTA = 33.017
IDF1 = 40.021
IDS  = 994
FPS  = 37.686
```

### Historical finalist TRK_MATCH_090

```text
MOTA = 22.953
HOTA = 33.328
IDF1 = 40.656
IDS  = 946
FPS  = 38.028
```

Important:

> These are historical verified project results, not the outcome of the new 2026-09-11 defensible validation workflow.

Do not overwrite or reinterpret them as if the new pipeline has already produced a result.

---

# 29. What can be claimed about portability / new datasets?

There are two different claims.

## 29.1 If parameters are recalibrated on each new dataset

Correct claim:

> Retraining-free, validation-calibrated adaptive MOT.

This means YOLO and ByteTrack do not need retraining, but the controller is still dataset-calibrated.

## 29.2 If AC-MOT is frozen once and moved to another dataset with no retuning

Strongest protocol:

```text
VisDrone validation
→ ablations + Optuna
→ freeze AC-MOT
→ VisDrone test
→ different target dataset
→ no YOLO retraining
→ no new Optuna
→ no target-dataset parameter retuning
```

If that succeeds, a stronger cross-dataset generalization claim becomes defensible.

Potential future datasets include UAV-style datasets such as UAVDT or AU-AIR, subject to compatible classes and evaluation setup.

Do not claim zero-retuning generalization before actually testing it.

---

# 30. Open methodological issues / items a future agent must not silently ignore

1. **ByteTrack tuning evidence** — the tuned tracker is frozen now, but exact tuned values need separate documented tracker evidence if challenged.
2. **Custom class-agnostic protocol** — do not call it the official VisDrone leaderboard.
3. **GT category / detector-class mismatch** — the custom filter includes category IDs that are not a one-to-one COCO detector class set; this should be revisited before any class-specific official benchmark claim.
4. **Outer screening bounds** — `512–960`, confidence `0.05–0.50`, and NMS `0.30–0.80` are declared screening budgets, not discovered truths.
5. **One-factor-at-a-time limitation** — interactions are handled later by joint optimization, but Stage 1 itself is not globally optimal search.
6. **Validation overfitting risk** — only 7 validation sequences are used, so increasing the number of tuned dimensions/trials indefinitely can overfit validation.
7. **50-trial budget** — computational budget, not mathematical proof of convergence; inspect convergence evidence.
8. **Hardware comparability** — T4 FPS should not be directly compared with a different GPU without clearly saying so.
9. **Final-test lock** — never delete/reset casually after test exposure.
10. **No GT-dependent runtime controller** — runtime crowding must remain deployable; detector-output calibration is the current correction.
11. **Tiny threshold `32×32`** — retained as an external proxy, not learned; document this honestly.
12. **No silent methodology changes mid-run** — if candidate grids, filters, thresholds, or selection rules change, archive the old run and start a new clearly versioned experiment.

---

# 31. Output files to preserve as thesis evidence

From Stage 1:

```text
OPERATING_RESOLUTION_SWEEP.csv
OPERATING_CONFIDENCE_SWEEP.csv
OPERATING_NMS_SWEEP.csv
SCIENTIFIC_SEARCH_SPACE.json
OPERATING_ABLATION_REPORT.json
```

From Stage 2:

```text
TEMPORAL_ABLATION_FULL.csv
TEMPORAL_ABLATION_RANKED.csv
FROZEN_TEMPORAL_CONFIG.json
TEMPORAL_W*_S*.json
```

From Stage 3:

```text
DETECTOR_DERIVED_CUE_CALIBRATION.json
OLD_A3_VALIDATION_W*_S*.json
EMPIRICAL_OPTUNA_TRIALS.csv
EMPIRICAL_OPTUNA.db
EMPIRICAL_STUDY_SIGNATURE.json
FROZEN_DEFENSIBLE_ACMOT_CONFIG.json
EMPIRICAL_PARAMETER_IMPORTANCE_MOTA.json   # best effort
```

Final test:

```text
FINAL_TEST_DONE.json
FINAL_TEST_RESULTS.json
```

Preserve all of them. They are part of the reproducibility and defense evidence.

---

# 32. Rules for any future AI agent

Unless the researcher explicitly changes the protocol:

1. Do not retrain YOLOv8n during the SCI study.
2. Do not co-optimize ByteTrack with SCI.
3. Do not use test-dev to choose any parameter.
4. Do not treat smoke results as scientific winners.
5. Do not hard-code old `640/736/832` after the full resolution screen exists.
6. Do not hard-code old confidence/NMS values after empirical screening exists.
7. Do not use GT object counts as a runtime crowding dependency.
8. Do not call the custom evaluator an official VisDrone leaderboard protocol.
9. Do not change frozen parameters after viewing final-test results.
10. Keep the current 25-FPS requirement consistent unless explicitly redesigning the protocol.
11. Preserve experiment outputs and hashes/signatures.
12. Explain every important number using one of these categories:
    - standard/external definition,
    - declared screening budget,
    - validation ablation,
    - validation statistics,
    - optimization.
13. Distinguish historical presentation results from results produced by the new defensible workflow.
14. If a result or value is still heuristic, say so explicitly rather than inventing a scientific justification.

---

# 33. Supervisor-ready summary

> “The detector and tracker are held fixed to isolate the AC-MOT contribution. The original AC-MOT controller used heuristic SCI weights, detector-control mappings, resolution thresholds, and temporal constants. Those exact values are no longer treated as inherently optimal. A validation-only operating-point screen first establishes defensible resolution, confidence, and NMS choices while changing one factor at a time. A separate temporal ablation selects the SCI smoothing window and analysis stride. The supported detector-control space and selected temporal design are then passed to a joint TPE/Optuna search that learns normalized SCI cue weights, confidence and NMS adaptation endpoints, and SCI resolution-switching thresholds. Runtime crowding calibration uses fixed-detector outputs rather than ground-truth counts. Candidate systems must meet the 25-FPS real-time gate and must not exceed the old-A3 IDS reference. All selected parameters are frozen before a one-shot held-out test. The current evaluation remains a custom class-agnostic AC-MOT TrackEval protocol, not the official VisDrone leaderboard protocol.”

---

# 34. Exact current status

As of **2026-09-11**:

```text
[IMPLEMENTED] Scientific operating-point portable sweep
[IMPLEMENTED] Temporal portable ablation
[IMPLEMENTED] Empirical joint portable Optuna
[IMPLEMENTED] Detector-derived crowd calibration
[IMPLEMENTED] SQLite/Drive Optuna resume
[IMPLEMENTED] Study-signature protection
[IMPLEMENTED] One-command validation runner
[IMPLEMENTED] Separate one-shot frozen final-test runner
[IMPLEMENTED] Final-test lock

[NEXT] Run the smoke test
[WAITING FOR] Smoke-test output / errors
[DO NOT YET] Launch the full 111-run validation workflow until smoke is clean
[DO NOT YET] Run the held-out final test
```

This is the exact continuation point for any new agent.