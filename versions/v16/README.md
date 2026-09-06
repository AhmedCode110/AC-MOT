# AC-MOT v16 — Final Paper Evaluation and Ablation Protocol

## Status

Evaluation-only successor to the verified realtime v15 system. v15 remains the proposed method definition; v16 must not retune it from evaluation results.

## Starting point: verified v15 realtime result

The user-verified v15 run reported decoded/live-frame processing throughput:

| System | Processing FPS | Realtime status |
|---|---:|---|
| LIVE_ADAPTIVE_RT_V15 | 34.0618 | PASS_REALTIME_PROCESSING |
| FIXED_736_REFERENCE_V15 | 22.8399 | FAIL_REALTIME_PROCESSING |
| FIXED_832_REFERENCE_V15 | 21.4154 | FAIL_REALTIME_PROCESSING |

Target: 25 FPS.

The v15 speed result establishes realtime processing for the adaptive system, but it does not by itself establish tracking-quality superiority.

## Why v16 exists

A paper-ready claim requires quality and speed to be evaluated under one transparent protocol. v16 therefore generates complete tracking outputs and compares:

- HOTA
- DetA
- AssA
- MOTA
- IDF1
- IDS
- FN
- FP
- decoded/live processing FPS
- p95 processing latency
- mean runtime resolution
- resolution switches
- serialized JPEG dataset playback FPS (reported separately)

## Research question

Does AC-MOT provide a better quality-speed trade-off than fixed-resolution and tracker-configuration baselines while satisfying a 25 FPS realtime processing constraint on a Tesla T4?

The intended contribution is a Pareto/constraint result, not a claim that the adaptive system must maximize every offline quality metric.

## Frozen proposed system

`ACMOT_V15_FULL` preserves the v15 proposed system:

- YOLOv8n
- PyTorch FP16 on Tesla T4
- adaptive SCI/Controller
- recovery enabled
- detector feedback enabled
- adaptive birth enabled
- NMS IoU 0.45
- ByteTrack high=0.18, low=0.04, new=0.20
- buffer=45
- match=0.86
- fuse score enabled
- realtime resolution governor enabled
- min runtime size 320
- detector budget fraction 0.70
- exactly one fresh detector inference per frame
- no frame skipping
- no detection cache used as deployment FPS

v16 results must not be used to alter these settings. Any retuning after seeing v16 results requires v17 and a development-only selection protocol.

## Systems evaluated

### Main baselines

1. `BYTETRACK_DEFAULT_640`
   - fixed 640
   - high=0.25, low=0.10, new=0.25
   - buffer=30, match=0.80
   - no recovery, feedback, adaptive birth or governor

2. `BYTETRACK_TUNED_640`
   - fixed 640
   - high=0.18, low=0.04, new=0.20
   - buffer=45, match=0.86
   - no recovery, feedback, adaptive birth or governor

3. `FIXED_640_RECOVERY`
   - fixed 640 with the tuned tracker and recovery
   - also serves as the no-adaptive-resolution / no-SCI-resolution baseline

4. `FIXED_736_RECOVERY`

5. `FIXED_832_RECOVERY`

### Proposed method

6. `ACMOT_V15_FULL`

### Ablations

7. `ABL_NO_RECOVERY`

8. `ABL_NO_FEEDBACK`

9. `ABL_NO_ADAPTIVE_BIRTH`

## Dataset and metric protocol

Required split:

- VisDrone2019-MOT-test-dev
- 17 sequences
- 6635 frames

Detector classes:

- COCO person, car, bus, truck

Ground-truth research filter:

- categories [1,4,5,6,9]
- score = 1
- occlusion < 2
- truncation < 2
- class-agnostic evaluation

This is explicitly **not** the official VisDrone benchmark preprocessing protocol. The paper must state that limitation.

Quality evaluation uses the repository `evaluate.py` adapter and pinned TrackEval revision `12c8791b303e0a0b50f753af204249e622d0281a`.

## Realtime definition

The 25 FPS realtime metric is decoded/live-frame processing:

- SceneAnalyzer
- Controller
- exactly one fresh YOLOv8n inference
- ByteTrack

JPEG read/decode is measured separately and also reported as serialized dataset playback FPS.

Excluded from processing FPS:

- Drive-to-local copy
- JPEG decode
- model/backend initialization
- calibration and warmup
- output serialization
- TrackEval evaluation

## v16 outputs

Every invocation creates a unique Drive envelope containing:

- `result/configuration.json`
- `result/dataset_manifest.json`
- `result/timing.json`
- full per-system/per-sequence `.frames.jsonl.gz` recordings
- `result/trackeval/metrics.json`
- `result/trackeval/summary.csv`
- `result/trackeval/protocol.json`
- `result/paper_comparison.csv`
- `result/paper_comparison.md`
- `result/paper_protocol.json`

## Paper-ready main table

The generated comparison table contains:

| Method | Role | HOTA | DetA | AssA | MOTA | IDF1 | IDS | FPS | RT >=25 | Mean size |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|

No values are pre-filled. All quality values must come from the new v16 full-run TrackEval output.

## Interpretation rules

A strong result does not require AC-MOT to beat fixed 832 on every quality metric. A publishable quality-speed result can be supported if the proposed system:

- satisfies the 25 FPS processing constraint,
- maintains competitive HOTA/IDF1/MOTA,
- improves over lower-cost realtime-nearest baselines, and/or
- approaches fixed-832 quality while substantially exceeding its throughput.

Ablations must show whether recovery, detector feedback and adaptive birth materially contribute.

## BoT-SORT note

BoT-SORT is not silently added to v16 because the current repository does not contain a verified BoT-SORT implementation under the exact v15 detector/runtime protocol. Adding it correctly would be a new protocol/code change and therefore belongs in v17 unless implemented before v16 is run and frozen.

## Portability

v16 uses `dataset=AUTO` and `output_root=AUTO` and resolves account-specific paths through `portable_v16.py`.

Default results destination:

`MyDrive/AC-MOT-results/v16/paper_eval`

## Integrity rule

v16 is evaluation only. Do not inspect its metrics and then modify the proposed v15 configuration while still calling the result v16. Any such change must create v17 and must separate development selection from held-out evaluation.
