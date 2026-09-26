"""
Offline candidate-signal audit (uses GT: DIAGNOSTIC ONLY, never online).

For every cached raw candidate (top-40 per frame, one resolution) compute
causal / detector-agnostic features and label it:
  TP   : IoU>=0.5 with a protocol GT box
  EXCL : IoU>=0.5 with an annotated object the protocol excludes
  BG   : background false positive (neither)

Features (all computable online at frame t from frames <= t):
  rank          within-frame score rank (1 = best)
  pct_frame     rank / n_candidates
  leader_now    score / max score of frame t
  leader_ema    score / EMA(max score of frames < t)
  persist1      IoU>=0.3 match among top-40 of frame t-1
  chain3        persist1 at t and the matched box persisted at t-1
  score_persist mean score of the t-1 match (0 if none) / leader_ema ref

Reports the AUC(TP vs BG) of each feature per detector x sequence, with
feature direction fixed a priori (higher = more object-like, except rank).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tools.eval_local import iou_xywh

TARGET = {1, 4, 5, 6, 9}


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    # average ranks for ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def audit(cache_npz, ann_path, res=736, topn=40, decay=0.9):
    z = np.load(cache_npz)
    a = z[f"det_{res}"]
    A = np.loadtxt(ann_path, delimiter=",", ndmin=2)
    keep = (np.isin(A[:, 7], list(TARGET)) & (A[:, 6] == 1)
            & (A[:, 8] < 2) & (A[:, 9] < 2))
    rows = []
    prev = prev_persist = prev_score = None
    ref = None
    for f in range(1, int(z["frames"]) + 1):
        d = a[a[:, 0] == f]
        d = d[np.argsort(-d[:, 5])][:topn]
        if not len(d):
            prev = None
            continue
        box = np.c_[d[:, 1], d[:, 2], d[:, 3] - d[:, 1], d[:, 4] - d[:, 2]]
        s = d[:, 5]
        g = A[(A[:, 0] == f) & keep]
        o = A[(A[:, 0] == f) & ~keep & (A[:, 7] != 0)]
        tp = (iou_xywh(box, g[:, 2:6]).max(1) >= .5) if len(g) else \
            np.zeros(len(d), bool)
        ex = (iou_xywh(box, o[:, 2:6]).max(1) >= .5) if len(o) else \
            np.zeros(len(d), bool)
        label = np.where(tp, "TP", np.where(ex, "EXCL", "BG"))
        if prev is not None and len(prev):
            m = iou_xywh(box, prev)
            best = m.argmax(1)
            p1 = m.max(1) >= .3
            chain = p1 & prev_persist[best]
            sp = np.where(p1, prev_score[best], 0.0)
        else:
            p1 = chain = np.zeros(len(d), bool)
            sp = np.zeros(len(d))
        leader_ema = s / ref if ref else np.full(len(d), np.nan)
        for i in range(len(d)):
            rows.append(dict(frame=f, rank=i + 1, pct_frame=(i + 1) / len(d),
                             leader_now=s[i] / s[0], leader_ema=leader_ema[i],
                             persist1=float(p1[i]), chain3=float(chain[i]),
                             score_persist=(sp[i] / ref if ref else np.nan),
                             area=box[i, 2] * box[i, 3], label=label[i]))
        ref = s[0] if ref is None else decay * ref + (1 - decay) * s[0]
        prev, prev_persist, prev_score = box, p1, s
    return rows


FEATURES = [("rank", -1), ("leader_now", 1), ("leader_ema", 1),
            ("persist1", 1), ("chain3", 1), ("score_persist", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache-root", default="outputs/det_cache")
    ap.add_argument("--detectors", nargs="+", default=["yolov8", "rtdetr"])
    ap.add_argument("--output", default="outputs/analysis/signal_audit.csv")
    args = ap.parse_args()
    out = []
    ds = Path(args.dataset)
    for det in args.detectors:
        for npz in sorted(Path(args.cache_root, det).glob("*.npz")):
            seq = npz.stem
            rows = audit(npz, ds / "annotations" / f"{seq}.txt")
            lab = np.array([r["label"] for r in rows])
            rec = dict(detector=det, sequence=seq,
                       n_tp=int((lab == "TP").sum()),
                       n_bg=int((lab == "BG").sum()),
                       n_excl=int((lab == "EXCL").sum()))
            for name, sign in FEATURES:
                v = np.array([r[name] for r in rows], float) * sign
                ok = np.isfinite(v)
                rec[f"auc_{name}"] = auc(v[ok & (lab == "TP")],
                                         v[ok & (lab == "BG")])
                if name == "leader_ema":
                    for lb in ("TP", "BG"):
                        x = v[ok & (lab == lb)]
                        rec[f"{name}_{lb}_q10"] = np.quantile(x, .1) if len(x) else np.nan
                        rec[f"{name}_{lb}_q50"] = np.quantile(x, .5) if len(x) else np.nan
            out.append(rec)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else
                           f"{k}={v}" for k, v in rec.items()), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
