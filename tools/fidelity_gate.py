"""Cross-hardware cache-fidelity gate (Amendment 5f): Mac-MPS vs T4-CUDA
caches of the same models/settings on the fixed gate subset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

GATE = ["uav0000020_00406_v", "uav0000315_00000_v", "uav0000316_01288_v",
        "uav0000342_04692_v"]
DETS = ["yolov8", "rtdetr"]


def iou_xyxy(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) -
                 np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) -
                 np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    ua = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None] + \
        ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None] - inter
    return inter / np.maximum(ua, 1e-9)


def compare_det(mps_npz, cuda_npz, res=736):
    A, B = np.load(mps_npz)[f"det_{res}"], np.load(cuda_npz)[f"det_{res}"]
    n_a = n_b = matched = cls_ok = 0
    ious, dscore, rank = [], [], []
    for f in np.union1d(np.unique(A[:, 0]), np.unique(B[:, 0])):
        a, b = A[A[:, 0] == f], B[B[:, 0] == f]
        n_a, n_b = n_a + len(a), n_b + len(b)
        if not len(a) or not len(b):
            continue
        m = iou_xyxy(a[:, 1:5], b[:, 1:5])
        r, c = linear_sum_assignment(-m)
        ok = m[r, c] >= 0.95
        matched += ok.sum()
        ious += list(m[r, c])
        dscore += list(np.abs(a[r, 5] - b[c, 5]))
        cls_ok += (a[r[ok], 6] == b[c[ok], 6]).sum()
        if ok.sum() > 3:
            ra = np.argsort(np.argsort(a[r[ok], 5]))
            rb = np.argsort(np.argsort(b[c[ok], 5]))
            rank.append(np.corrcoef(ra, rb)[0, 1])
    return dict(n_mps=int(n_a), n_cuda=int(n_b),
                match_rate=matched / max(max(n_a, n_b), 1),
                iou_median=float(np.median(ious)), iou_p5=float(np.percentile(ious, 5)),
                dscore_median=float(np.median(dscore)),
                class_agreement=cls_ok / max(matched, 1),
                score_rank_spearman=float(np.nanmean(rank)))


def main(mps_root, cuda_root):
    out = {}
    ok = True
    for d in DETS:
        for s in GATE:
            r = compare_det(f"{mps_root}/{d}/{s}.npz", f"{cuda_root}/{d}/{s}.npz")
            out[f"{d}/{s}"] = r
            ok &= (r["match_rate"] >= 0.99 and r["class_agreement"] >= 0.995
                   and r["dscore_median"] <= 0.01)
            print(d, s, {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in r.items()})
    out["detection_level_pass"] = bool(ok)
    json.dump(out, open("outputs/analysis/fidelity_gate_detection.json", "w"),
              indent=1)
    print("DETECTION-LEVEL", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
