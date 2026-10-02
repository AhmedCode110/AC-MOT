"""
P-GSCI-2: objective-weight scene index (SDC-style) and scene-response index,
development split val-7 only; GT is used for the targets, never as an input.

Weighting follows the SDC metric (Sensors 2026, 26(9), 2886): min-max
normalized indicators oriented so that larger = harder, entropy weights and
PCA weights (first k components reaching 85 % cumulative variance of the
correlation matrix), combined as w = a * wE + (1 - a) * wP with a = 0.6.

Indices per segment row of the wide-span cue audit:
  H  historical SCI weights (crowd .30, tiny .30, edges .20, dark .10, blur .05)
  O  SDC-style objective weights on the same scene indicators
  G  objective-weighted sum of [S, R, S*R, T]; S, R, T are the SDC-style
     indices of the scene, detector-response and tracker groups
  R, SxR  attribution only

Targets: B = q(960) - q(512) (benefit of compute, from the audit rows) and
D = -sum q(736) / sum n_GT (difficulty at the operating point), q = TP-FP-IDS.

  python tools/g2/gsci_audit.py <audit rows json> [--json out]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ALPHA = 0.6
CUM_VAR = 0.85
SCENE = ["trk_density", "trk_small", "img_edges", "img_dark", "img_blur"]
HIST_W = np.array([0.30, 0.30, 0.20, 0.10, 0.05])
RESPONSE = ["probe_up", "det_ambig"]
TRACKER = ["trk_churn"]
SEG = 30


def minmax(X):
    lo, hi = X.min(0), X.max(0)
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)


def entropy_weights(X):
    m = X.shape[0]
    P = X / np.where(X.sum(0) > 0, X.sum(0), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(P > 0, P * np.log(P), 0.0)
    e = np.where(X.sum(0) > 0, -L.sum(0) / np.log(m), 1.0)
    d = 1 - e
    return d / d.sum()


def pca_weights(X):
    if X.shape[1] == 1:
        return np.ones(1)
    C = np.corrcoef(X, rowvar=False)
    lam, U = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    theta = lam / lam.sum()
    k = int(np.searchsorted(np.cumsum(theta), CUM_VAR) + 1)
    c = (np.abs(U[:, :k]) * theta[:k]).sum(1)
    return c / c.sum()


def sdc(X):
    """Normalized matrix X (rows x indicators) -> (index, weights)."""
    w = ALPHA * entropy_weights(X) + (1 - ALPHA) * pca_weights(X)
    return X @ w, w


def matrix(rows, cues):
    X = np.array([[r[c] for c in cues] for r in rows], float)
    med = np.nanmedian(X, 0)
    X = np.where(np.isnan(X), med, X)
    return minmax(X)


def indices(rows):
    XS, XR, XT = matrix(rows, SCENE), matrix(rows, RESPONSE), matrix(rows, TRACKER)
    H = XS @ (HIST_W / HIST_W.sum())
    O, wS = sdc(XS)
    R, wR = sdc(XR)
    T, wT = sdc(XT)
    S, R, T = (minmax(v[:, None])[:, 0] for v in (O, R, T))
    G, wG = sdc(np.column_stack([S, R, S * R, T]))
    weights = dict(scene=dict(zip(SCENE, wS)), response=dict(zip(RESPONSE, wR)), tracker=dict(zip(TRACKER, wT)),
                   G=dict(zip(["S", "R", "SxR", "T"], wG)), historical=dict(zip(SCENE, HIST_W / HIST_W.sum())))
    return dict(H=H, O=O, G=G, R=R, SxR=S * R), weights


def difficulty(rows):
    os.environ.setdefault("SCI_CACHE", "sweep")
    from tools.sci_v7 import dev
    from tools.sci_v7.oracle import frame_quality
    from tools.eval_local import load_gt
    SPLITS, _ = dev._splits()
    data = Path(SPLITS[dev.SPLIT]["data"])
    cache = {}
    out = []
    for r in rows:
        key = (r["det"], r["seq"])
        if key not in cache:
            gt = load_gt(data / "annotations" / f"{r['seq']}.txt")
            q = frame_quality("V7f+R736", *key)
            n = np.bincount(gt[:, 0].astype(int), minlength=len(q) + 1)[1:len(q) + 1]
            cache[key] = (q, n)
        q, n = cache[key]
        a = r["f0"] - 1
        out.append(-q[a:a + SEG].sum() / max(n[a:a + SEG].sum(), 1))
    return np.array(out)


def spearman(x, y):
    from scipy.stats import spearmanr
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 4:
        return np.nan
    return float(spearmanr(x, y).statistic)


def cells(rows, idx, y):
    out = {}
    for d in sorted({r["det"] for r in rows}):
        for s in sorted({r["seq"] for r in rows}):
            m = np.array([r["det"] == d and r["seq"] == s for r in rows])
            out[(d, s)] = spearman(idx[m], y[m])
    return out


def rule(c):
    dets = sorted({d for d, _ in c})
    by = {d: float(np.nanmean([v for (dd, _), v in c.items() if dd == d])) for d in dets}
    vals = np.array([v for v in c.values() if np.isfinite(v)])
    sign = np.sign(np.mean(list(by.values())))
    same = len({np.sign(v) for v in by.values()}) == 1
    big = all(abs(v) >= 0.10 for v in by.values())
    agree = int((np.sign(vals) == sign).sum())
    return dict(by_detector=by, mean=float(vals.mean()), cells_with_sign=agree, cells=len(vals),
                retained=bool(same and big and agree >= 10))


def boot_diff(ca, cb, n=10000, seed=42):
    seqs = sorted({s for _, s in ca})
    rng = np.random.default_rng(seed)

    def mean(c, pick):
        v = [c[(d, s)] for s in pick for d in sorted({d for d, _ in c})]
        return float(np.nanmean(v))
    diff = mean(ca, seqs) - mean(cb, seqs)
    bs = []
    for _ in range(n):
        pick = rng.choice(seqs, len(seqs), replace=True)
        bs.append(mean(ca, pick) - mean(cb, pick))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(diff=diff, ci_lo=float(lo), ci_hi=float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = json.loads(Path(a.rows).read_text())["rows"]
    idx, weights = indices(rows)
    targets = dict(B=np.array([r["benefit"] for r in rows], float), D=difficulty(rows))
    res = dict(weights=weights, targets={})
    for tname, y in targets.items():
        C = {k: cells(rows, v, y) for k, v in idx.items()}
        res["targets"][tname] = dict(
            rule={k: rule(c) for k, c in C.items()},
            G_minus_O=boot_diff(C["G"], C["O"]), O_minus_H=boot_diff(C["O"], C["H"]),
            G_minus_H=boot_diff(C["G"], C["H"]),
            per_cell={k: {f"{d}/{s}": v for (d, s), v in c.items()} for k, c in C.items()})
    print("weights:", json.dumps(weights, default=lambda x: round(float(x), 3)))
    for tname, t in res["targets"].items():
        print(f"\ntarget {tname}")
        print("| index | mean rho YOLOv8n | mean rho RT-DETR-L | mean rho (14 cells) | cells with that sign | retained |")
        print("|---|---|---|---|---|---|")
        for k, r in t["rule"].items():
            print(f"| {k} | {r['by_detector']['yolov8']:+.3f} | {r['by_detector']['rtdetr']:+.3f} | {r['mean']:+.3f} | "
                  f"{r['cells_with_sign']}/{r['cells']} | {'yes' if r['retained'] else 'no'} |")
        for k in ("G_minus_O", "O_minus_H", "G_minus_H"):
            b = t[k]
            print(f"{k}: {b['diff']:+.3f} [{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}]")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
