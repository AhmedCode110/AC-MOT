"""
Local (Mac) evaluator for the custom class-agnostic AC-MOT protocol.

NOT official VisDrone MOT evaluation.

GT filter: category in {1,4,5,6,9}, score flag == 1, truncation < 2,
occlusion < 2. Class ignored during matching. IoU threshold 0.5.

All metrics come from the pinned TrackEval checkout (HOTA, CLEAR, Identity)
so every variant is scored by the same code. This is a stand-in for the
Colab motmetrics+TrackEval evaluator; its agreement with the Colab numbers
is checked against the saved V1 tracks before it is trusted.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

for _name, _val in (("float", float), ("int", int), ("bool", bool),
                     ("complex", complex)):
    if not hasattr(np, _name):
        setattr(np, _name, _val)
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

TRACKEVAL = Path(
    "/Users/ahmedgouda/Desktop/CUE_SELECTION/cue_ablation_tools/TrackEval"
)
sys.path.insert(0, str(TRACKEVAL))

from trackeval.metrics import CLEAR, HOTA, Identity  # noqa: E402

TARGET_CLASSES = {1, 4, 5, 6, 9}


def load_gt(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    keep = (
        np.isin(a[:, 7].astype(int), list(TARGET_CLASSES))
        & (a[:, 6] == 1)
        & (a[:, 8] < 2)
        & (a[:, 9] < 2)
    )
    return a[keep]


def load_tracks(path):
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return np.zeros((0, 10))
    return np.loadtxt(path, delimiter=",", ndmin=2)


def iou_xywh(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a.astype(float)
    b = b.astype(float)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) -
                 np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) -
                 np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def build_data(gt, tr, num_frames):
    gt_map = {v: i for i, v in enumerate(np.unique(gt[:, 1]).astype(int))}
    tr_map = {v: i for i, v in enumerate(np.unique(tr[:, 1]).astype(int))} \
        if len(tr) else {}
    data = dict(num_timesteps=num_frames, num_gt_ids=len(gt_map),
                num_tracker_ids=len(tr_map), num_gt_dets=len(gt),
                num_tracker_dets=len(tr), gt_ids=[], tracker_ids=[],
                similarity_scores=[])
    for t in range(1, num_frames + 1):
        g = gt[gt[:, 0] == t]
        p = tr[tr[:, 0] == t] if len(tr) else tr
        data["gt_ids"].append(
            np.array([gt_map[int(i)] for i in g[:, 1]], dtype=int))
        data["tracker_ids"].append(
            np.array([tr_map[int(i)] for i in p[:, 1]], dtype=int))
        data["similarity_scores"].append(iou_xywh(g[:, 2:6], p[:, 2:6]))
    return data


def summarize(h, c, i):
    tp, fp, fn = c["CLR_TP"], c["CLR_FP"], c["CLR_FN"]
    return dict(
        MOTA=100 * float(c["MOTA"]),
        HOTA=100 * float(np.mean(h["HOTA"])),
        IDF1=100 * float(i["IDF1"]),
        IDS=int(c["IDSW"]),
        FP=int(fp), FN=int(fn), GT=int(tp + fn),
        Precision=100 * tp / max(1, tp + fp),
        Recall=100 * tp / max(1, tp + fn),
    )


def motmetrics_rows(dataset, tracks_dir, seqs):
    """Reference protocol: MOTA/IDF1/IDS/FP/FN/P/R from motmetrics."""
    import motmetrics as mm

    accs = []
    for s in seqs:
        gt = load_gt(dataset / "annotations" / f"{s}.txt")
        tr = load_tracks(Path(tracks_dir) / f"{s}.txt")
        n = len(list((dataset / "sequences" / s).glob("*.jpg")))
        acc = mm.MOTAccumulator(auto_id=True)
        for t in range(1, n + 1):
            g = gt[gt[:, 0] == t]
            p = tr[tr[:, 0] == t] if len(tr) else tr
            dist = mm.distances.iou_matrix(g[:, 2:6], p[:, 2:6], max_iou=0.5)
            acc.update(g[:, 1].astype(int).tolist(),
                       p[:, 1].astype(int).tolist(), dist)
        accs.append(acc)
    mh = mm.metrics.create()
    names = ["mota", "idf1", "num_switches", "num_false_positives",
             "num_misses", "precision", "recall"]
    df = mh.compute_many(accs, metrics=names, names=seqs,
                         generate_overall=True)
    out = {}
    for name, r in df.iterrows():
        out["OVERALL" if name == "OVERALL" else name] = dict(
            MOTA=100 * r["mota"], IDF1=100 * r["idf1"],
            IDS=int(r["num_switches"]), FP=int(r["num_false_positives"]),
            FN=int(r["num_misses"]), Precision=100 * r["precision"],
            Recall=100 * r["recall"])
    return out


def evaluate(dataset, tracks_dir, sequences=None):
    """HOTA from TrackEval; the remaining metrics from motmetrics when it
    is installed (reference protocol), else TrackEval CLEAR/Identity with
    IDS flagged as provisional (IDS_source column)."""
    rows = _evaluate_trackeval(dataset, tracks_dir, sequences)
    try:
        ref = motmetrics_rows(Path(dataset), tracks_dir,
                              [r["sequence"] for r in rows[:-1]])
    except ImportError:
        for r in rows:
            r["IDS_source"] = "trackeval_provisional"
        return rows
    for r in rows:
        r.update(ref[r["sequence"]])
        r["IDS_source"] = "motmetrics"
    return rows


def _evaluate_trackeval(dataset, tracks_dir, sequences=None):
    dataset = Path(dataset)
    seqs = sorted(p.name for p in (dataset / "sequences").iterdir()
                  if p.is_dir())
    if sequences:
        seqs = [s for s in seqs if s in sequences]
    metrics = [HOTA(), CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
               Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False})]
    per = {m.get_name(): {} for m in metrics}
    rows = []
    for s in seqs:
        n = len(list((dataset / "sequences" / s).glob("*.jpg")))
        data = build_data(load_gt(dataset / "annotations" / f"{s}.txt"),
                          load_tracks(Path(tracks_dir) / f"{s}.txt"), n)
        for m in metrics:
            with redirect_stdout(io.StringIO()):
                per[m.get_name()][s] = m.eval_sequence(data)
        rows.append(dict(sequence=s, **summarize(
            per["HOTA"][s], per["CLEAR"][s], per["Identity"][s])))
    comb = {m.get_name(): m.combine_sequences(per[m.get_name()])
            for m in metrics}
    rows.append(dict(sequence="OVERALL", **summarize(
        comb["HOTA"], comb["CLEAR"], comb["Identity"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--output")
    ap.add_argument("--sequences", nargs="*")
    args = ap.parse_args()
    rows = evaluate(args.dataset, args.tracks, args.sequences)
    for r in rows:
        print(f"{r['sequence']:<22} MOTA {r['MOTA']:8.3f} HOTA {r['HOTA']:6.3f}"
              f" IDF1 {r['IDF1']:6.3f} IDS {r['IDS']:5d} FP {r['FP']:6d}"
              f" FN {r['FN']:6d} P {r['Precision']:6.2f} R {r['Recall']:6.2f}")
    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
