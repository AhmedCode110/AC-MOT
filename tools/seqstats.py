"""
Per-sequence sufficient statistics for the reference protocol, so that any
subset of sequences (cross-validation folds) can be scored exactly as
tools/eval_local.evaluate would score it:

  motmetrics counts: GT, FP, FN, IDS, IDTP, IDFP, IDFN  (summable)
  TrackEval HOTA per-sequence result dict (combined with HOTA's own
  combine_sequences)
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from tools.eval_local import HOTA, build_data, load_gt, load_tracks


def sequence_stats(dataset, seq, tracks_file):
    import motmetrics as mm

    dataset = Path(dataset)
    gt = load_gt(dataset / "annotations" / f"{seq}.txt")
    tr = load_tracks(tracks_file)
    n = len(list((dataset / "sequences" / seq).glob("*.jpg")))

    acc = mm.MOTAccumulator(auto_id=True)
    for t in range(1, n + 1):
        g = gt[gt[:, 0] == t]
        p = tr[tr[:, 0] == t] if len(tr) else tr
        acc.update(g[:, 1].astype(int).tolist(),
                   p[:, 1].astype(int).tolist(),
                   mm.distances.iou_matrix(g[:, 2:6], p[:, 2:6], max_iou=0.5))
    r = mm.metrics.create().compute(
        acc, metrics=["num_objects", "num_false_positives", "num_misses",
                      "num_switches", "idtp", "idfp", "idfn"], name=seq)
    counts = {k: int(r[k].iloc[0]) for k in r.columns}

    with redirect_stdout(io.StringIO()):
        hota = HOTA().eval_sequence(build_data(gt, tr, n))
    return dict(counts=counts, hota=hota,
                tracks_per_frame=len(tr) / max(n, 1))


def combine(stats_list):
    """Exact reference-protocol metrics over a set of sequences."""
    c = {k: sum(s["counts"][k] for s in stats_list)
         for k in stats_list[0]["counts"]}
    gt = c["num_objects"]
    tp = gt - c["num_misses"]
    with redirect_stdout(io.StringIO()):
        h = HOTA().combine_sequences({str(i): s["hota"]
                                      for i, s in enumerate(stats_list)})
    return dict(
        MOTA=100 * (1 - (c["num_misses"] + c["num_false_positives"]
                         + c["num_switches"]) / gt),
        HOTA=100 * float(np.mean(h["HOTA"])),
        IDF1=100 * 2 * c["idtp"] / max(1, 2 * c["idtp"] + c["idfp"]
                                       + c["idfn"]),
        IDS=c["num_switches"], FP=c["num_false_positives"],
        FN=c["num_misses"],
        Precision=100 * tp / max(1, tp + c["num_false_positives"]),
        Recall=100 * tp / max(1, gt),
    )
