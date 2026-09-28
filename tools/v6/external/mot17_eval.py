"""
MOT17 val-half evaluation for the external transfer (official MOTChallenge
protocol through TrackEval, pinned 12c8791 = the project's evaluator):
HOTA / CLEAR (MOTA, IDS, FP, FN, precision, recall) / Identity (IDF1), per
sequence and pooled, plus the frozen project bootstrap (10,000 paired
sequence resamples, seed 42, percentile 95% CI) on pooled metrics.

GT = the authors' shipped val-half GT (BoostTrack results/gt/MOT17-val),
verified byte-identical to the val-half GT derived from the user's MOT17 copy
(reports/mot17_split_verification.json).

  python mot17_eval.py <trackers_root> <name> [<name> ...]          # table
  python mot17_eval.py --boot <trackers_root> <A> <B>                 # B - A
"""
from __future__ import annotations

import io
import json
import pickle
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

for _n, _v in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _n):
        setattr(np, _n, _v)
if not hasattr(np, "asfarray"):     # NumPy-2 alias used by motmetrics 1.4.0
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
TRACKEVAL = Path("/Users/ahmedgouda/Desktop/CUE_SELECTION/cue_ablation_tools/TrackEval")
sys.path.insert(0, str(TRACKEVAL))
import trackeval  # noqa: E402

GT = Path("/Users/ahmedgouda/Desktop/acmot_external/BoostTrack/results/gt")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]


def evaluate(trackers_root, name):
    """Per-sequence TrackEval results; cached next to the tracker folder.
    trackers_root/MOT17-val/<name>/data/<seq>.txt"""
    cache = Path(trackers_root) / "MOT17-val" / name / "trackeval_per_seq.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update(dict(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False,
                   OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False,
                   TIME_PROGRESS=False, DISPLAY_LESS_PROGRESS=True))
    dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dc.update(dict(GT_FOLDER=str(GT), TRACKERS_FOLDER=str(Path(trackers_root)), BENCHMARK="MOT17",
                   SPLIT_TO_EVAL="val", TRACKERS_TO_EVAL=[name], TRACKER_SUB_FOLDER="data",
                   PRINT_CONFIG=False, SEQMAP_FILE=str(GT / "seqmaps/MOT17-val.txt")))
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    with redirect_stdout(io.StringIO()):
        res, _ = trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)], metrics)
    r = res["MotChallenge2DBox"][name]
    per = {s: r[s]["pedestrian"] for s in SEQS}
    pickle.dump(per, open(cache, "wb"))
    return per


def combine(per_list):
    """Pooled metrics over a list of per-sequence result dicts."""
    H, C, I = trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()
    d = {str(i): p for i, p in enumerate(per_list)}
    with redirect_stdout(io.StringIO()):
        h = H.combine_sequences({k: v["HOTA"] for k, v in d.items()})
        c = C.combine_sequences({k: v["CLEAR"] for k, v in d.items()})
        i = I.combine_sequences({k: v["Identity"] for k, v in d.items()})
    return dict(HOTA=100 * float(np.mean(h["HOTA"])), DetA=100 * float(np.mean(h["DetA"])),
                AssA=100 * float(np.mean(h["AssA"])), MOTA=100 * float(c["MOTA"]),
                IDF1=100 * float(i["IDF1"]), IDS=int(c["IDSW"]), FP=int(c["CLR_FP"]),
                FN=int(c["CLR_FN"]), Precision=100 * float(c["CLR_Pr"]),
                Recall=100 * float(c["CLR_Re"]), Frag=int(c["Frag"]))


def table(trackers_root, names):
    rows = {}
    for n in names:
        per = evaluate(trackers_root, n)
        rows[n] = dict(pooled=combine([per[s] for s in SEQS]),
                       per_seq={s: combine([per[s]]) for s in SEQS})
    return rows


def bootstrap(trackers_root, a, b, n=10000, seed=42):
    """Paired sequence bootstrap of pooled (B - A); frozen project protocol."""
    pa, pb = evaluate(trackers_root, a), evaluate(trackers_root, b)
    A, B = [pa[s] for s in SEQS], [pb[s] for s in SEQS]
    ma, mb = combine(A), combine(B)
    rng = np.random.default_rng(seed)
    keys = ["HOTA", "MOTA", "IDF1", "IDS", "FP", "FN", "AssA", "DetA"]
    diffs = {k: np.empty(n) for k in keys}
    for i in range(n):
        idx = rng.integers(0, len(SEQS), len(SEQS))
        ra, rb = combine([A[j] for j in idx]), combine([B[j] for j in idx])
        for k in keys:
            diffs[k][i] = rb[k] - ra[k]
    out = {}
    for k in keys:
        lo, hi = np.percentile(diffs[k], [2.5, 97.5])
        out[k] = dict(diff=float(mb[k] - ma[k]), ci_lo=float(lo), ci_hi=float(hi),
                      p_le0=float((diffs[k] <= 0).mean()), p_ge0=float((diffs[k] >= 0).mean()))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--boot":
        root, a, b = sys.argv[2:5]
        r = bootstrap(root, a, b)
        print(json.dumps(r, indent=1))
    else:
        root, *names = sys.argv[1:]
        t = table(root, names)
        for n, v in t.items():
            p = v["pooled"]
            print(f"{n:<34} HOTA {p['HOTA']:6.3f} MOTA {p['MOTA']:6.3f} IDF1 {p['IDF1']:6.3f} "
                  f"IDS {p['IDS']:5d} FP {p['FP']:6d} FN {p['FN']:6d} P {p['Precision']:6.2f} "
                  f"R {p['Recall']:6.2f} AssA {p['AssA']:6.2f} DetA {p['DetA']:6.2f}")


def motmetrics_sparsetrack(results_dir):
    """SparseTrack's in-repo evaluation (track.py do_track): motmetrics,
    GT loaded with min_confidence=1, IoU 0.5, lap solver. GT files = the
    val-half GT (byte-identical to gt_val_half.txt of the official
    conversion)."""
    import glob
    import os
    from collections import OrderedDict
    import motmetrics as mm
    mm.lap.default_solver = "lap"
    gtfiles = sorted(glob.glob(str(GT / "MOT17-val" / "*" / "gt" / "gt.txt")))
    tsfiles = sorted(f for f in glob.glob(os.path.join(results_dir, "*.txt")))
    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1))
                      for f in tsfiles if Path(f).stem in gt])
    accs, names = [], []
    for k, tsacc in ts.items():
        accs.append(mm.utils.compare_to_groundtruth(gt[k], tsacc, "iou", distth=0.5))
        names.append(k)
    mh = mm.metrics.create()
    s = mh.compute_many(accs, names=names, metrics=["mota", "idf1", "num_switches", "num_false_positives",
                                                   "num_misses", "precision", "recall", "num_objects"],
                        generate_overall=True)
    o = s.loc["OVERALL"]
    return dict(MOTA=100 * float(o["mota"]), IDF1=100 * float(o["idf1"]), IDS=int(o["num_switches"]),
                FP=int(o["num_false_positives"]), FN=int(o["num_misses"]),
                Precision=100 * float(o["precision"]), Recall=100 * float(o["recall"]))
