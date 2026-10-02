"""
Summary of a resolution-sweep cache: per detector and resolution, detector
latency on the building device (mean / p50 / p95 ms over all val-7 frames),
candidate counts, and detection-level agreement with the V7-record native
cache (built on another device) where that resolution exists there.

  python tools/sci_v7/sweep_summary.py <sweep_root> <native_root> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ub = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ua[:, None] + ub[None, :] - inter + 1e-12)


def agreement(A, B, thr=0.25):
    """Of the boxes with score >= thr in A, fraction with an IoU >= 0.9 partner
    of the same class in B on the same frame; mean |score difference| of the pairs."""
    hit = tot = 0
    dif = []
    for f in np.unique(A[:, 0]):
        a = A[(A[:, 0] == f) & (A[:, 5] >= thr)]
        b = B[B[:, 0] == f]
        if not len(a):
            continue
        tot += len(a)
        M = iou(a[:, 1:5], b[:, 1:5])
        if M.size:
            same = a[:, 6][:, None] == b[:, 6][None, :]
            M = np.where(same, M, 0)
            j = M.argmax(1)
            ok = M[np.arange(len(a)), j] >= 0.9
            hit += int(ok.sum())
            dif += list(np.abs(a[ok, 5] - b[j[ok], 5]))
    return dict(boxes=tot, matched=hit / max(tot, 1), mean_abs_score_diff=float(np.mean(dif)) if dif else None)


def main(sweep, native, out):
    sweep, native = Path(sweep), Path(native)
    res = {}
    for dd in sorted(p for p in sweep.iterdir() if p.is_dir()):
        for rd in sorted((p for p in dd.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
            ms, n = [], 0
            fid = []
            for t in sorted(rd.glob("*.timing.json")):
                j = json.loads(t.read_text())
                ms += j["ms"]
                seq = t.name.split(".")[0]
                z = np.load(rd / f"{seq}.npz")
                n += len(z["det"])
                nf = native / dd.name / f"{seq}.npz"
                if nf.exists():
                    zn = np.load(nf)
                    if f"det_{rd.name}" in zn.files:
                        fid.append(agreement(z["det"], zn[f"det_{rd.name}"]))
            ms = np.asarray(ms)
            r = dict(frames=len(ms), mean_ms=float(ms.mean()), p50_ms=float(np.percentile(ms, 50)),
                     p95_ms=float(np.percentile(ms, 95)), candidates_per_frame=n / max(len(ms), 1))
            if fid:
                b = sum(x["boxes"] for x in fid)
                r["vs_native_cache"] = dict(
                    boxes_ge_0_25=b, matched_iou_0_9=sum(x["matched"] * x["boxes"] for x in fid) / max(b, 1),
                    mean_abs_score_diff=float(np.mean([x["mean_abs_score_diff"] for x in fid
                                                       if x["mean_abs_score_diff"] is not None])))
            res.setdefault(dd.name, {})[rd.name] = r
    Path(out).write_text(json.dumps(res, indent=1))
    for d, v in res.items():
        for r, x in v.items():
            print(d, r, f"{x['mean_ms']:.1f} ms (p95 {x['p95_ms']:.1f})", f"{x['candidates_per_frame']:.1f} cand/frame",
                  x.get("vs_native_cache", ""))


if __name__ == "__main__":
    main(*sys.argv[1:4])
