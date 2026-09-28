"""
V7 diagnostic D1: what does IoU-0.5 class-agnostic duplicate suppression
remove, per stream? (DEVELOPMENT ONLY; uses labels offline.)

For every overlapping pair (IoU > 0.3) inside a frame, with h = higher and
l = lower score:
  distinct : both true positives of DIFFERENT objects (suppressing l loses a TP)
  dup      : l is a false positive that overlaps h's object (IoU >= 0.5 with
             the GT that h matched)  -> a redundant detection
  other    : everything else (both FP, ignore regions, ...)
Features: IoU, containment (inter / min area), score gap in logits, same
class, area ratio, centre offset / sqrt(area_h), and track context.
"""
from __future__ import annotations

import sys

import numpy as np

from tools.v7.streams import STREAMS, iou, load


def lg(s):
    s = np.clip(s, 1e-9, 1 - 1e-9)
    return np.log(s / (1 - s))


def pair_table(stream, min_iou=0.3):
    rows = []
    for seq, frames in stream.items():
        for f in frames:
            b, s, lab, gi, gt = f["boxes"], f["scores"], f["label"], f["gt_idx"], f["gt"]
            n = len(b)
            if n < 2:
                continue
            M = iou(b, b)
            np.fill_diagonal(M, 0)
            I, J = np.where(np.triu(M, 1) > min_iou)
            if not len(I):
                continue
            G = iou(b, gt) if len(gt) else np.zeros((n, 0))
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            for i, j in zip(I, J):
                h, l = (i, j) if s[i] >= s[j] else (j, i)
                inter = M[h, l] * (area[h] + area[l]) / (1 + M[h, l])
                if lab[h] == 1 and lab[l] == 1 and gi[h] != gi[l]:
                    cat = 0      # distinct
                elif lab[l] == 0 and lab[h] == 1 and G.shape[1] and G[l, gi[h]] >= 0.5:
                    cat = 1      # duplicate of h's object
                elif lab[h] == 0 and lab[l] == 1 and G.shape[1] and G[h, gi[l]] >= 0.5:
                    cat = 2      # reversed duplicate (higher one is the redundant one)
                else:
                    cat = 3
                rows.append((M[h, l], inter / max(min(area[h], area[l]), 1e-9),
                             lg(s[h]) - lg(s[l]), s[h], s[l], int(f["cls"][h] == f["cls"][l]),
                             min(area[h], area[l]) / max(area[h], area[l]),
                             np.hypot(cx[h] - cx[l], cy[h] - cy[l]) / np.sqrt(max(area[h], 1e-9)),
                             cat))
    return np.array(rows) if rows else np.zeros((0, 9))


def summarize(name, T, floors=(0.1, 0.3, 0.5)):
    print(f"\n=== {name}: {len(T)} pairs with IoU > 0.3")
    cats = ["distinct", "dup", "revdup", "other"]
    for lo, hi in ((0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        m = (T[:, 0] > lo) & (T[:, 0] <= hi)
        for fl in floors[:1]:
            mm = m & (T[:, 4] >= fl)
            c = [int((T[mm, 8] == k).sum()) for k in range(4)]
            print(f"  IoU ({lo:.1f},{hi:.1f}]  lower>= {fl}: " +
                  " ".join(f"{cats[k]} {c[k]:6d}" for k in range(4)) +
                  f"   same-class {T[mm, 5].mean() if mm.any() else float('nan'):.2f}")
    for fl in floors:
        m = (T[:, 0] > 0.5) & (T[:, 4] >= fl)
        c = [int((T[m, 8] == k).sum()) for k in range(4)]
        print(f"  V6 rule (IoU>0.5) lower>= {fl}: " + " ".join(f"{cats[k]} {c[k]:6d}" for k in range(4)))
    m5 = T[:, 0] > 0.5
    for k, nm in ((0, "distinct"), (1, "dup")):
        x = T[m5 & (T[:, 8] == k) & (T[:, 4] >= 0.1)]
        if len(x):
            q = lambda c: np.percentile(x[:, c], [10, 50, 90]).round(2)
            print(f"  {nm:<8} IoU {q(0)} contain {q(1)} dlogit {q(2)} s_h {q(3)} s_l {q(4)} "
                  f"sameclass {x[:, 5].mean():.2f} arearatio {q(6)} offset {q(7)}")


if __name__ == "__main__":
    for n in (sys.argv[1:] or STREAMS):
        T = pair_table(load(n))
        np.save(f"outputs/v7/diag/pairs_{n}.npy", T)
        summarize(n, T)
