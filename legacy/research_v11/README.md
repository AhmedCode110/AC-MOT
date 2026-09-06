# AC-MOT v11 research candidate

Status: implemented candidate, not validated on real sequences and not final paper results.

## Files

- `AC_MOT_v11_Candidate_T4.ipynb`: new self-contained T4 experiment notebook, preserving legacy files.
- `controller.py`: experimental low-confidence recovery and three-analyzer-update resolution dwell.
- `evaluate.py`: pinned upstream TrackEval HOTA/CLEAR/Identity, validated recording adapter and combined sequence aggregation.
- `test_research.py`: synthetic metric and controller regression tests.

## What changed

Only the A3 candidate uses the new controller. It passes detections at confidence 0.04 to ByteTrack (high=0.18, low=0.04, new=0.20, buffer=45, match=0.86). This exposes the low-confidence recovery band. These are candidates, not optimized settings. This candidate replaces adaptive detector confidence with a fixed low input threshold; therefore it must not be described as an unchanged A3 adaptive-threshold method. Adaptive NMS remains. The blur-specific NMS nudge is omitted in this candidate. Resolution changes need three distinct scene-analyzer updates; repeated calls on the same state do not count. Tracker association threshold 0.86 is a permissive cost threshold, not a strict IoU requirement.

A0–A2 and A4 retain legacy behavior. A4 is a legacy reference, NOT a v11 ReID ablation; a matched v11 ablation suite is still required. Explicit FP32 avoids the earlier requested-versus-actual FP16 discrepancy. Dependency versions remain runtime-installed and recorded by the existing recorder; a fully pinned inference environment remains necessary before publication.

The recorder still emits legacy diagnostic summaries for compatibility. Use only the separate TrackEval output for the corrected metric comparison; do not rename legacy HOTA proxy to HOTA. No performance improvement is claimed until measured.

## Evaluate existing saved tracks without inference

Clone https://github.com/JonathonLuiten/TrackEval and checkout `12c8791b303e0a0b50f753af204249e622d0281a`. Install numpy, scipy, pycocotools, matplotlib, opencv-python-headless and tabulate in an isolated environment. The adapter restores removed NumPy float/int aliases for this upstream version without modifying equations.

```sh
python evaluate.py /path/to/recorded_run --dataset /path/to/VisDrone2019-MOT-test-dev --trackeval /path/to/TrackEval --output /path/to/new_evaluation
```

Requires the original configuration.json, dataset_manifest.json, annotations, and every per-frame gzip recording. Output: summary.csv, full per-sequence/combined metrics.json, protocol.json with hashes. Existing output directories are refused. Scores use 0–1 scale. HOTA uses upstream alpha thresholds and sequence combination; MOTA/IDF1 use IoU 0.5. Research GT filtering remains categories [1,4,5,6,9], score=1, truncation<2 and occlusion<2. Predictions are class-agnostic, and ignored-region processing is NOT implemented. Thus these are official metric implementations under a custom protocol, not official VisDrone benchmark scores.

## Publication evidence still needed

1. Verify dataset identity: existing notebook path identifies test-dev, despite VAL_SEQS variable name. Do not call this validation or tune on these 17 sequences.
2. Tune on separate train/validation sequences and freeze one configuration before test-dev. Since test-dev has already been inspected, disclose this and obtain additional untouched evaluation where possible.
3. Validate category mapping, ignored regions and class-wise preprocessing against the chosen benchmark protocol before making benchmark-comparable claims.
4. Run matched ablations: low-confidence recovery only, resolution dwell only, both, and any adaptive threshold variant. Keep detector/weights, hardware, precision and evaluator identical.
5. Compare against credible baselines under identical inputs; include per-sequence results, failures, IDSW/FN/FP and DetA/AssA. Use sequence-level paired uncertainty analysis, not independent-frame significance tests.
6. Report synchronized total FPS (total frames / total elapsed time), warm-up, repeated timings, memory and compute cost. Do not use mean per-sequence FPS as total throughput.
7. Release a fixed environment, source/weight hashes, frozen parameters, reproducible manifests, outputs, and protocol disclosure. Document limitations and contribution beyond parameter tuning.

Journal quartile or acceptance cannot be guaranteed by this code. Existing presentations and final CSVs have not been replaced.
