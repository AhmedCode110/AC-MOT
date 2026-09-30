# Re-scoring the archived U2MOT and SparseTrack outputs

`rescore_legacy_pipelines.py` evaluates tracker result files that already
exist on Drive. It does not run a detector or a tracker and changes no
parameter. For each system it:

- recomputes MOTA/IDF1/IDS/FP/FN with the exact motmetrics protocol that
  produced the archived numbers, and compares them with those numbers
  (`reproduction` block: `match` must be true);
- computes HOTA, DetA, AssA, LocA, MOTA and IDF1 with TrackEval
  (commit 12c8791) per sequence;
- for each pair of systems, reports the difference with a 95 % bootstrap
  interval over sequences (10 000 resamples, seed 0), win/tie/loss per
  sequence, and the leave-one-sequence-out range.

Tests: `TRACKEVAL_DIR=<TrackEval checkout> python -m pytest tests/test_legacy_rescore.py`.

## Colab (CPU runtime is enough)

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%%bash
pip -q install motmetrics==1.4.0 lap
git clone -q https://github.com/JonathonLuiten/TrackEval.git /content/TrackEval
git -C /content/TrackEval checkout -q 12c8791
git clone -q -b universal-adapters-v1-y0zkeh https://github.com/AhmedCode110/AC-MOT.git /content/AC-MOT  # private repo: upload tools/legacy_rescore/ instead
```

Locate the inputs (paths differ between accounts and shortcuts):

```bash
%%bash
find /content/drive -maxdepth 6 -type d -name 'FINAL_U2MOT_ACMOT_FREEZE_2026-09-14' 2>/dev/null
find /content/drive -maxdepth 6 -type d -path '*VisDrone2019-MOT-test-dev*' -name annotations 2>/dev/null
find /content/drive/MyDrive/AC-MOT-SparseTrack-NEW -type d -name track_results 2>/dev/null
```

### U2MOT, VisDrone2019-MOT-test-dev

```bash
%%bash
PKG="<path of FINAL_U2MOT_ACMOT_FREEZE_2026-09-14>"
ANN="<path of VisDrone2019-MOT-test-dev/annotations>"
cd /content/AC-MOT
python tools/legacy_rescore/rescore_legacy_pipelines.py --trackeval-dir /content/TrackEval \
  u2mot --gt-dir "$ANN" \
        --baseline-dir "$PKG/01_A0_BASELINE/track_res" \
        --controller-dir "$PKG/06_FINAL_TESTDEV_17SEQ/track_res" \
        --out-dir /content/drive/MyDrive/RESCORE_LEGACY/u2mot
```

Expected reproduction: baseline FP 41385, FN 63241, IDS 1239, MOTA 53.9 and
IDF1 69.8 at one decimal; controller FP 40155, FN 64801, IDS 1152,
MOTA 53.76678605352365, IDF1 69.84938968519636. If `reproduction_all_match`
is false, the archived controller numbers were not produced by the U2MOT
protocol; report that instead of the numbers.

### SparseTrack, MOT17 val_half

```bash
%%bash
P=/content/drive/MyDrive/AC-MOT-SparseTrack-NEW
cd /content/AC-MOT
python tools/legacy_rescore/rescore_legacy_pipelines.py --trackeval-dir /content/TrackEval \
  sparsetrack --gt-root "$P/data/MOT17/train" \
    --run static070="<track_results of the S1 static NMS 0.70 run>" \
    --run static075="$P/adaptive/STEP7K_H_STATIC075_20260917T093716Z/STATIC_075/NMS_075/track_results" \
    --run static080="$P/sensitivity/STEP7B_NMS_FULLVAL/NMS_080/track_results" \
    --run adaptive="$P/adaptive/STEP7K_E_FIX2_MATCHED_DETERMINISTIC_20260917T090646Z/ADAPTIVE_EDGE_V1/NMS_070/track_results" \
    --pair adaptive:static075 --pair static075:static070 --pair static080:static070 --pair adaptive:static070 \
    --out-dir /content/drive/MyDrive/RESCORE_LEGACY/sparsetrack
```

`adaptive:static075` is the comparison named in the frozen controller
manifest. Each run's `metrics.csv` (next to its `track_results`) is
recomputed and checked.

Outputs: `*_rescore.json` (reproduction checks, overall metrics, paired
statistics, SHA256 of every input file) and one per-sequence CSV per system.
