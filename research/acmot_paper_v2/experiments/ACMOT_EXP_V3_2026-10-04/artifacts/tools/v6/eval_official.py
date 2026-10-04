"""
Official-compatible VisDrone2019-MOT (Task 4b) evaluation, ported from the
official MATLAB toolkit (github.com/VisDrone/VisDrone2018-MOT-toolkit:
evaluateTrackB.m, eval/dropObjects.m, eval/breakGts.m, eval/classEval.m,
utils/evaluateBenchmark.m):

* GT and results of category 0 (ignored region) / 11 (others) are dropped;
  any GT or result box whose area lies >= 50% inside an ignored/others
  region of the same frame is dropped (dropObjects).
* GT trajectories spanning several categories are split per category
  (breakGts). No occlusion / truncation filter (unlike the internal
  protocol).
* Class-aware evaluation on {pedestrian 1, car 4, van 5, truck 6, bus 9};
  CLEAR-MOT at IoU 0.5 (motmetrics) and ID measures per (sequence, class);
  counts summed over classes and sequences (evaluateBenchmark):
  MOTA = 1 - (FN + FP + IDS) / GT, IDF1 = 2 IDTP / (numGT + numPRED).
* HOTA is not part of the official toolkit; it is added class-aware with the
  pinned TrackEval HOTA (combine over all (sequence, class) pairs).

Detector classes are COCO; the integration mapping (adapter-level, not a
tuning choice) is person->pedestrian(1), car->car(4), bus->bus(9),
truck->truck(6). COCO has no "van": van GT can only be matched by a van
prediction, so every van object is a miss for COCO detectors (reported).
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from tools.eval_local import HOTA, build_data

EVAL_CLASSES = {1: "pedestrian", 4: "car", 5: "van", 6: "truck", 9: "bus"}
COCO_TO_VISDRONE = {0: 1, 2: 4, 5: 9, 7: 6}


def _inside_ignored(boxes, ign):
    """Fraction of each box (x,y,w,h) covered by the union of ignored
    regions, on the integer pixel grid as in dropObjects.m."""
    if not len(ign) or not len(boxes):
        return np.zeros(len(boxes))
    x0 = int(max(1, min(ign[:, 0].min(), boxes[:, 0].min())))
    y0 = int(max(1, min(ign[:, 1].min(), boxes[:, 1].min())))
    x1 = int(max((ign[:, 0] + ign[:, 2]).max(), (boxes[:, 0] + boxes[:, 2]).max())) + 2
    y1 = int(max((ign[:, 1] + ign[:, 3]).max(), (boxes[:, 1] + boxes[:, 3]).max())) + 2
    m = np.zeros((y1 - y0 + 2, x1 - x0 + 2), np.float64)
    for g in np.maximum(1, np.round(ign)).astype(int):
        m[g[1] - y0:g[1] + g[3] - y0 + 1, g[0] - x0:g[0] + g[2] - x0 + 1] = 1
    ii = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    out = np.zeros(len(boxes))
    for k, b in enumerate(np.maximum(1, np.round(boxes)).astype(int)):
        x, y, w, h = b
        xa, ya = x - x0, y - y0
        xb, yb = xa + w, ya + h
        xa, ya = np.clip([xa, ya], 0, [m.shape[1] - 1, m.shape[0] - 1])
        xb, yb = np.clip([xb, yb], 0, [m.shape[1] - 1, m.shape[0] - 1])
        s = ii[yb + 1, xb + 1] - ii[ya, xb + 1] - ii[yb + 1, xa] + ii[ya, xa]
        out[k] = s / max(w * h, 1)
    return out


def drop_objects(rows, gt_all):
    """rows: [frame, id, x, y, w, h, score, cls, ...]."""
    keep = (rows[:, 7] != 0) & (rows[:, 7] != 11)
    for f in np.unique(rows[:, 0]):
        idx = np.where((rows[:, 0] == f) & keep)[0]
        ign = gt_all[(gt_all[:, 0] == f) & np.isin(gt_all[:, 7], (0, 11)), 2:6]
        if len(ign) and len(idx):
            keep[idx[_inside_ignored(rows[idx, 2:6], ign) >= 0.5]] = False
    return rows[keep]


def break_gts(gt):
    gt = gt.copy()
    nid = gt[:, 1].max() + 1 if len(gt) else 0
    for tid in np.unique(gt[:, 1]):
        idx = gt[:, 1] == tid
        cls = np.unique(gt[idx, 7])
        for c in cls[1:]:
            gt[idx & (gt[:, 7] == c), 1] = nid
            nid += 1
    return gt


def official_sequence_stats(dataset, seq, tracks, class_map=COCO_TO_VISDRONE):
    import motmetrics as mm
    dataset = Path(dataset)
    gt_all = np.loadtxt(dataset / "annotations" / f"{seq}.txt", delimiter=",", ndmin=2)
    n = len(list((dataset / "sequences" / seq).glob("*.jpg")))
    gt = break_gts(drop_objects(gt_all, gt_all))
    tr = np.asarray(tracks, float).reshape(-1, 10) if len(tracks) else np.zeros((0, 10))
    if len(tr):
        tr = tr.copy()
        tr[:, 7] = [class_map.get(int(c), -1) for c in tr[:, 7]]
        tr = drop_objects(tr, gt_all)
        tr = tr[tr[:, 0] <= gt_all[:, 0].max()]
    out = {}
    for c in EVAL_CLASSES:
        g, p = gt[gt[:, 7] == c], tr[tr[:, 7] == c] if len(tr) else tr
        if not len(g):
            continue
        acc = mm.MOTAccumulator(auto_id=True)
        for t in range(1, n + 1):
            gg, pp = g[g[:, 0] == t], p[p[:, 0] == t] if len(p) else p
            acc.update(gg[:, 1].astype(int).tolist(), pp[:, 1].astype(int).tolist(),
                       mm.distances.iou_matrix(gg[:, 2:6], pp[:, 2:6], max_iou=0.5))
        r = mm.metrics.create().compute(
            acc, metrics=["num_objects", "num_false_positives", "num_misses",
                          "num_switches", "idtp", "idfp", "idfn"], name=seq)
        counts = {k: int(r[k].iloc[0]) for k in r.columns}
        counts["num_pred"] = int(len(p))
        with redirect_stdout(io.StringIO()):
            h = HOTA().eval_sequence(build_data(g, p if len(p) else np.zeros((0, 10)), n))
        out[c] = dict(counts=counts, hota=h)
    return out


def combine_official(stats_list, classes=tuple(EVAL_CLASSES)):
    """stats_list: list of official_sequence_stats outputs."""
    items = [s[c] for s in stats_list for c in classes if c in s]
    c = {k: sum(x["counts"][k] for x in items) for k in items[0]["counts"]}
    gt = c["num_objects"]
    tp = gt - c["num_misses"]
    with redirect_stdout(io.StringIO()):
        h = HOTA().combine_sequences({str(i): x["hota"] for i, x in enumerate(items)})
    return dict(MOTA=100 * (1 - (c["num_misses"] + c["num_false_positives"]
                                 + c["num_switches"]) / gt),
                HOTA=100 * float(np.mean(h["HOTA"])),
                IDF1=100 * 2 * c["idtp"] / max(1, gt + c["num_pred"]),
                IDS=c["num_switches"], FP=c["num_false_positives"], FN=c["num_misses"],
                Precision=100 * tp / max(1, tp + c["num_false_positives"]),
                Recall=100 * tp / max(1, gt), GT=gt)
