"""
V7 diagnostic D2: where do V6's causal nested-Otsu thresholds fall relative
to the host's native operating point, and what is the precision / recall of
each band? (DEVELOPMENT ONLY; labels used offline.)

For each stream: frames processed in order, V6 dedup (IoU 0.5) optional,
t1/t2 = nested Otsu on the pooled logits of the previous 10 frames.
Reports the median raw-score equivalents of t1/t2, the fraction of true
positives and false positives in each band, and the same for a native
operating point (assoc a, low l) given on the command line.
"""
from __future__ import annotations

import sys
from collections import deque

import numpy as np

from online_calibration import dedup, nested_otsu
from tools.v7.streams import load

sig = lambda x: 1 / (1 + np.exp(-x))


def lg(s):
    s = np.clip(s, 1e-9, 1 - 1e-9)
    return np.log(s / (1 - s))


def run(name, use_dedup=True, floor=0.0, window=10):
    st = load(name)
    rec = []
    for seq, frames in st.items():
        W = deque(maxlen=window)
        for f in frames:
            s, lab = f["scores"], f["label"]
            b = f["boxes"]
            m = s >= floor
            s, lab, b = s[m], lab[m], b[m]
            if use_dedup and len(s) > 1:
                k = sorted(dedup(b, s, 0.5))
                s, lab = s[k], lab[k]
            L = lg(s)
            th = nested_otsu(np.concatenate(W)) if W else None
            if th is not None:
                t1, t2, _ = th
                rec.append((t1, t2, L, lab, len(f["gt"])))
            W.append(L)
    return rec


def report(name, native_a, native_l, **kw):
    rec = run(name, **kw)
    t1 = np.array([r[0] for r in rec])
    t2 = np.array([r[1] for r in rec])
    ngt = sum(r[4] for r in rec)
    def band(lo_fn, hi_fn):
        tp = fp = 0
        for (a, b, L, lab, _) in rec:
            m = (L >= lo_fn(a, b)) & (L < hi_fn(a, b))
            tp += int((lab[m] == 1).sum())
            fp += int((lab[m] == 0).sum())
        return tp, fp
    A, Lo = lg(native_a), lg(native_l)
    rows = {
        "V6 primary  [t2,inf)": band(lambda a, b: b, lambda a, b: np.inf),
        "V6 extension[t1,t2)": band(lambda a, b: a, lambda a, b: b),
        "V6 discard  (-,t1)": band(lambda a, b: -np.inf, lambda a, b: a),
        f"native prim [{native_a},inf)": band(lambda a, b: A, lambda a, b: np.inf),
        f"native low [{native_l},{native_a})": band(lambda a, b: Lo, lambda a, b: A),
        "projected prim [clip(a,t1,t2),inf)": band(lambda a, b: min(max(A, a), b), lambda a, b: np.inf),
    }
    print(f"\n=== {name} {kw}  frames {len(rec)}  GT {ngt}")
    print(f"  t1 raw median {sig(np.median(t1)):.3f} [p10 {sig(np.percentile(t1, 10)):.3f}, p90 {sig(np.percentile(t1, 90)):.3f}]"
          f"   t2 raw median {sig(np.median(t2)):.3f} [p10 {sig(np.percentile(t2, 10)):.3f}, p90 {sig(np.percentile(t2, 90)):.3f}]")
    print(f"  native a={native_a} inside [t1,t2): {np.mean((A >= t1) & (A < t2)):.2f}  below t1: {np.mean(A < t1):.2f}  above t2: {np.mean(A >= t2):.2f}")
    for k, (tp, fp) in rows.items():
        print(f"  {k:<36} TP {tp:7d} ({100 * tp / ngt:5.1f}% of GT)  FP {fp:7d}  prec {100 * tp / max(tp + fp, 1):5.1f}")


if __name__ == "__main__":
    cfg = {"mot17_yolox": (0.6, 0.1), "mot17_bt": (0.6, 0.1), "vd_yolov8": (0.25, 0.1),
           "vd_rtdetr": (0.25, 0.1), "vd_fasterrcnn": (0.25, 0.1)}
    names = sys.argv[1:] or list(cfg)
    for n in names:
        a, l = cfg[n]
        report(n, a, l)
        if n == "mot17_yolox":
            report(n, a, l, use_dedup=False)
            report(n, a, l, floor=0.1)
            report(n, a, l, floor=0.1, use_dedup=False)
