"""
Cue audit for the scene state (GT used OFFLINE only, to measure what the
online controller should predict).

Target: per-frame benefit of spending more pixels,
    benefit(f) = [#GT matched by top-K candidates at 832
                  - #GT matched by top-K candidates at 640] / max(#GT, 1)
with K = #GT of the frame (oracle count, diagnosis only), IoU 0.5.

Candidate causal cues for frame f (computed from frame f-1 detector output
at the resolution actually requested, here 736, and frame-f image stats):
  n_conf      number of candidates with ECDF-normalized score >= 0.9
  tiny        fraction of those with area < 32x32 px
  log_area    median log box area of those
  n_raw       raw candidate count
  score_gap   z-logit gap statistic median over top-10 (Platt-invariant)
  edges, brightness, blur   image statistics of frame f
Reports Spearman rho(cue, benefit) per detector x sequence and pooled.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from tools.eval_local import iou_xywh, load_gt

CUES = ["n_conf", "tiny", "log_area", "n_raw", "score_gap", "edges",
        "brightness", "blur"]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return np.nan
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def xywh(d):
    return np.c_[d[:, 1], d[:, 2], d[:, 3] - d[:, 1], d[:, 4] - d[:, 2]]


def frame_rows(npz, ann):
    z = np.load(npz)
    G = load_gt(ann)
    by = {r: z[f"det_{r}"] for r in (640, 736, 832)}
    hist = []
    rows = []
    prev = None
    for f in range(1, int(z["frames"]) + 1):
        g = G[G[:, 0] == f]
        k = len(g)
        cur = {}
        for r in (640, 832):
            d = by[r][by[r][:, 0] == f]
            d = d[np.argsort(-d[:, 5])][:max(k, 1)]
            cur[r] = int((iou_xywh(g[:, 2:6], xywh(d)).max(1) >= .5).sum()) \
                if k and len(d) else 0
        benefit = (cur[832] - cur[640]) / max(k, 1)
        d736 = by[736][by[736][:, 0] == f]
        if prev is not None:
            rows.append(dict(frame=f, benefit=benefit, n_gt=k, **prev))
        # cues from frame f (used for frame f+1)
        s = d736[:, 5]
        if len(hist):
            ref = np.sort(np.concatenate(hist))
            u = np.searchsorted(ref, s, side="right") / len(ref)
        else:
            u = np.argsort(np.argsort(s)) / max(len(s), 1)
        conf = d736[u >= 0.9]
        area = (conf[:, 3] - conf[:, 1]) * (conf[:, 4] - conf[:, 2])
        lg = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
        top = np.sort(lg)[::-1][:10]
        iqr = np.subtract(*np.percentile(lg, [75, 25])) if len(lg) > 3 else 1
        v = z["visual"][f - 1]
        prev = dict(n_conf=len(conf),
                    tiny=float((area < 1024).mean()) if len(conf) else 0.0,
                    log_area=float(np.median(np.log(area))) if len(conf) else np.nan,
                    n_raw=len(d736),
                    score_gap=float(np.median((top - top[0]) / max(iqr, 1e-6)))
                    if len(top) else np.nan,
                    edges=float(v[0]), brightness=float(v[1]),
                    blur=float(v[2]))
        if f % 10 == 0:
            hist.append(s)
            hist = hist[-20:]
    return rows


def main():
    dataset = Path(sys.argv[1])
    out = {}
    pooled = {c: [] for c in CUES}
    for det in ("yolov8", "rtdetr"):
        allrows = []
        for npz in sorted(Path("outputs/det_cache", det).glob("*.npz")):
            rows = frame_rows(npz, dataset / "annotations" /
                              f"{npz.stem}.txt")
            b = [r["benefit"] for r in rows]
            rec = {c: spearman([r[c] for r in rows], b) for c in CUES}
            rec["mean_benefit"] = float(np.mean(b))
            out[f"{det}/{npz.stem}"] = rec
            allrows += rows
            print(f"{det:<7}{npz.stem[:12]} benefit {np.mean(b):+.3f} " +
                  " ".join(f"{c}:{rec[c]:+.2f}" for c in CUES), flush=True)
        b = [r["benefit"] for r in allrows]
        rec = {c: spearman([r[c] for r in allrows], b) for c in CUES}
        out[f"{det}/POOLED"] = rec
        print(f"{det:<7}POOLED       " + " ".join(
            f"{c}:{rec[c]:+.2f}" for c in CUES), flush=True)
    json.dump(out, open("outputs/analysis/cue_benefit_audit.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
