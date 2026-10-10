"""
Paired sequence bootstrap for V7 comparisons on VisDrone / UAVDT (same
method as the frozen project protocol, tools/v6/stats.py and
tools/v6/external/mot17_eval.py): 10,000 resamples of the sequence set
with replacement, numpy default_rng(42), the SAME indices for both systems,
pooled metrics recomputed on every resample, percentile 95% CI of B - A,
P(delta <= 0) and P(delta >= 0).

Effect sizes: dz = mean / sd of the per-sequence paired differences, and
the pooled delta over the bootstrap SD.

  python tools/v7/bootstrap.py <A> <B> [--protocol internal|official] [--n 10000]
Env: V7_SPLIT, V7_DETS (comma list; one bootstrap per detector, plus the
detector-pooled one when more than one: resampling unit = (detector,
sequence) cell). A system can be 'V6:<name>' (V6 record, reference only).
MOT17 (SparseTrack / BoostTrack): tools/v7/mot17_eval_v7.py --boot.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
KEYS = ["MOTA", "HOTA", "IDF1", "IDS", "FP", "FN"]


def cells(split, system, dets, protocol):
    from tools.v7 import dev
    seqs = dev.split_sequences(split)
    out = []
    for d in dets:
        for s in seqs:
            if protocol == "official":
                out.append(pickle.load(open(dev._path(split, system, d, s, ".official.pkl"), "rb")))
            else:
                out.append(dev.load(split, system, d, s))
    return out


def combiner(protocol):
    if protocol == "official":
        from tools.v6.eval_official import combine_official
        return combine_official
    from tools.seqstats import combine
    return combine


def paired(A, B, combine, n=10000, seed=42, keys=KEYS):
    assert len(A) == len(B)
    ma, mb = combine(A), combine(B)
    per = {k: np.array([combine([b])[k] - combine([a])[k] for a, b in zip(A, B)]) for k in keys}
    rng = np.random.default_rng(seed)
    diffs = {k: np.empty(n) for k in keys}
    m = len(A)
    for i in range(n):
        idx = rng.integers(0, m, m)
        ra, rb = combine([A[j] for j in idx]), combine([B[j] for j in idx])
        for k in keys:
            diffs[k][i] = rb[k] - ra[k]
    out = {}
    for k in keys:
        lo, hi = np.percentile(diffs[k], [2.5, 97.5])
        sd = float(np.std(diffs[k], ddof=1))
        psd = float(np.std(per[k], ddof=1)) if m > 1 else float("nan")
        delta = float(mb[k] - ma[k])
        out[k] = dict(A=float(ma[k]), B=float(mb[k]), diff=delta, ci_lo=float(lo), ci_hi=float(hi),
                      p_le0=float((diffs[k] <= 0).mean()), p_ge0=float((diffs[k] >= 0).mean()),
                      dz=float(per[k].mean() / psd) if psd and psd > 0 else float("nan"),
                      delta_over_bootsd=delta / sd if sd > 0 else float("nan"),
                      seq_wins=int((per[k] > 0).sum()), seq_losses=int((per[k] < 0).sum()))
    return out


def run(split, a, b, dets, protocol="internal", n=10000, seed=42):
    comb = combiner(protocol)
    res = dict(split=split, A=a, B=b, protocol=protocol, n=n, seed=seed, per_det={})
    allA, allB = [], []
    for d in dets:
        A, B = cells(split, a, [d], protocol), cells(split, b, [d], protocol)
        res["per_det"][d] = paired(A, B, comb, n, seed)
        allA += A
        allB += B
    if len(dets) > 1:
        res["pooled_cells"] = paired(allA, allB, comb, n, seed)
    return res


def fmt(r):
    lines = []
    for scope, v in list(r["per_det"].items()) + ([("pooled", r["pooled_cells"])]
                                                   if "pooled_cells" in r else []):
        lines.append(f"[{r['split']} {r['protocol']}] {scope}: {r['B']} - {r['A']}")
        for k, x in v.items():
            lines.append(f"  {k:<5} {x['A']:9.3f} -> {x['B']:9.3f}  d {x['diff']:+8.3f} "
                         f"[{x['ci_lo']:+8.3f}, {x['ci_hi']:+8.3f}]  P(d<=0) {x['p_le0']:.3f} "
                         f"P(d>=0) {x['p_ge0']:.3f}  dz {x['dz']:+.2f}  seq +{x['seq_wins']}/-{x['seq_losses']}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("A")
    ap.add_argument("B")
    ap.add_argument("--protocol", default="internal", choices=["internal", "official"])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default=None, help="append the result to this JSON list file")
    a = ap.parse_args()
    split = os.environ.get("V7_SPLIT", "val7")
    dets = os.environ.get("V7_DETS", "yolov8,rtdetr").split(",")
    from tools.v7.dev import guard
    guard(split)
    r = run(split, a.A, a.B, dets, a.protocol, a.n, a.seed)
    print(fmt(r))
    if a.json:
        p = Path(a.json)
        old = json.load(open(p)) if p.exists() else []
        old.append(r)
        json.dump(old, open(p, "w"), indent=1)
