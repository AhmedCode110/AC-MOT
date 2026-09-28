"""Detection-level audit (diagnostic only; development data).

Every raw cached detection of a frame is matched one-to-one (Hungarian,
IoU >= 0.5) first to the evaluated GT (tools.eval_local filter), then the
unmatched rest to the NON-evaluated annotations (ignored regions cat 0,
excluded categories, heavy occlusion, score flag 0); an unmatched detection
lying > 50% inside an ignored region (cat 0) is also "unevaluated".
Labels: 0 = evaluated object, 1 = unevaluated object/region, 2 = clutter.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from tools.eval_local import TARGET_CLASSES, iou_xywh


def load_all(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    keep = (np.isin(a[:, 7].astype(int), list(TARGET_CLASSES)) & (a[:, 6] == 1)
            & (a[:, 8] < 2) & (a[:, 9] < 2))
    return a[keep], a[~keep]


def label_frame(det_xyxy, gt_eval, gt_other):
    lab = np.full(len(det_xyxy), 2, int)
    if not len(det_xyxy):
        return lab
    d = np.asarray(det_xyxy, float).copy()
    d[:, 2:] -= d[:, :2]                       # -> xywh
    free = np.arange(len(d))
    for gt, code in ((gt_eval, 0), (gt_other, 1)):
        if not len(gt) or not len(free):
            continue
        m = iou_xywh(d[free], gt[:, 2:6])
        r, c = linear_sum_assignment(-m)
        ok = m[r, c] >= 0.5
        lab[free[r[ok]]] = code
        free = np.setdiff1d(free, free[r[ok]])
    ign = gt_other[gt_other[:, 7] == 0] if len(gt_other) else gt_other
    if len(ign) and len(free):
        x1, y1 = d[free, 0], d[free, 1]
        x2, y2 = x1 + d[free, 2], y1 + d[free, 3]
        gx1, gy1 = ign[:, 2], ign[:, 3]
        gx2, gy2 = gx1 + ign[:, 4], gy1 + ign[:, 5]
        iw = np.clip(np.minimum(x2[:, None], gx2) - np.maximum(x1[:, None], gx1), 0, None)
        ih = np.clip(np.minimum(y2[:, None], gy2) - np.maximum(y1[:, None], gy1), 0, None)
        frac = (iw * ih).max(1) / np.maximum(d[free, 2] * d[free, 3], 1e-9)
        lab[free[frac > 0.5]] = 1
    return lab


def audit_sequence(npz, ann, res=736):
    zf = np.load(npz)
    z = zf[f"det_{res}"]
    ge, go = load_all(ann)
    out = []
    for f in range(1, int(zf["frames"]) + 1):
        dd = z[z[:, 0] == f]
        lab = label_frame(dd[:, 1:5], ge[ge[:, 0] == f], go[go[:, 0] == f])
        out.append(dict(score=dd[:, 5], cls=dd[:, 6], box=dd[:, 1:5], lab=lab,
                        n_gt=int((ge[:, 0] == f).sum())))
    return out
