"""
V5 stage S2 — cue utility per control target (VisDrone val; GT offline).

Unit: 30-frame window w of (detector, sequence). Features: the causal
scene/tracking state logged at the window's first frame by the reference
(V4) run of that detector. Cost: errors_w(v) = FP+FN+IDSW for each value v
of the target (from S1 fixed-value runs); regret_w(v) = errors_w(v) - min.

Model per cue: cost-sensitive stump  value = a if cue <= θ else b,
(θ, a, b) minimising total training regret. Evaluation: leave-one-sequence-
out, BOTH detectors in training and test. Reported per cue × target:
  cv_gain      held-out regret reduction vs the training-best single value
               (MOTA points of held-out GT), pooled and per detector
  null_p95     95th percentile of cv_gain when the cue is permuted across
               windows (200 permutations, seed 0) — chance level
  folds_better outer folds where the stump beats the global value on the
               TRAINING sequences (selection frequency)
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np

from scene_state import ALL_CUES
from tools.v5_s1_headroom import DETS, TARGETS, WINDOW, load, seqs

import os
rng = np.random.default_rng(0)
COST = os.environ.get("V5_COST", "mota")


def build(target):
    values = TARGETS[target][1]
    rows = []   # (det, seq, cue_vector, regret_vector, gt)
    for d, s in itertools.product(DETS, seqs()):
        runs = [load(target, v, d, s) for v in values]
        ev = [r["frame_events"] for r in runs]
        ref_states = load("gate_tau", 0.75, d, s)["states"]   # V4 run
        n = len(ev[0])
        for w in range(int(np.ceil(n / WINDOW))):
            sl = slice(w * WINDOW, (w + 1) * WINDOW)
            if COST == "hota_idf1":     # (FP+FN) + (IDFP+IDFN)
                err = np.array([e[sl, 2].sum() + e[sl, 3].sum() +
                                e[sl, 6].sum() + e[sl, 7].sum()
                                for e in ev], dtype=float)
            else:                       # MOTA-aligned FP+FN+IDSW (attempt 1)
                err = np.array([e[sl, 2:5].sum() for e in ev], dtype=float)
            st = ref_states[w * WINDOW]
            rows.append((d, s, np.array([st.get(c, np.nan) for c in ALL_CUES],
                                        dtype=float),
                         err - err.min(), ev[0][sl, 0].sum()))
    return values, rows


def fit_stump(x, R):
    """x: (n,) cue; R: (n, k) regret. Returns (theta, a, b, cost)."""
    ok = np.isfinite(x)
    base = R.sum(0)
    best = (np.inf, int(np.argmin(base)), int(np.argmin(base)), base.min())
    if ok.sum() < 10:
        return best
    xs = np.unique(np.quantile(x[ok], np.linspace(0.05, 0.95, 19)))
    for th in xs:
        le = (~ok) | (x <= th)
        cl, cg = R[le].sum(0), R[~le].sum(0)
        c = cl.min() + cg.min()
        if c < best[3]:
            best = (th, int(np.argmin(cl)), int(np.argmin(cg)), c)
    return best


def apply(theta, a, b, x):
    return np.where(np.isfinite(x) & (x > theta), b, a)


def cv_gain(rows, j, permute=False):
    seqlist = seqs()
    gain = {d: 0.0 for d in DETS}
    gts = {d: 0.0 for d in DETS}
    better = 0
    X = np.array([r[2][j] for r in rows])
    if permute:
        X = rng.permutation(X)
    R = np.array([r[3] for r in rows])
    S = np.array([r[1] for r in rows])
    D = np.array([r[0] for r in rows])
    G = np.array([r[4] for r in rows])
    for held in seqlist:
        tr, te = S != held, S == held
        th, a, b, cost = fit_stump(X[tr], R[tr])
        glob = int(np.argmin(R[tr].sum(0)))
        better += cost < R[tr][:, glob].sum() - 1e-9
        pred = apply(th, a, b, X[te])
        for d in DETS:
            m = te & (D == d)
            idx = np.where(m)[0]
            gain[d] += R[idx, glob].sum() - R[idx, pred[m[te]]].sum()
            gts[d] += G[m].sum()
    per_det = {d: 100 * gain[d] / gts[d] for d in DETS}
    pooled = 100 * sum(gain.values()) / sum(gts.values())
    return pooled, per_det, better


def main(targets):
    report = {}
    for target in targets:
        values, rows = build(target)
        print(f"\n== {target} values {values}")
        report[target] = {}
        for j, cue in enumerate(ALL_CUES):
            pooled, per_det, better = cv_gain(rows, j)
            null = [cv_gain(rows, j, permute=True)[0] for _ in range(200)]
            p95 = float(np.percentile(null, 95))
            report[target][cue] = dict(cv_gain=pooled, per_detector=per_det,
                                       null_p95=p95, folds_better=better)
            flag = "SIGNAL" if pooled > p95 and min(per_det.values()) > 0 \
                else ""
            print(f"  {cue:<16} cv_gain {pooled:+.3f}  (yolo {per_det['yolov8']:+.3f}"
                  f", rtdetr {per_det['rtdetr']:+.3f})  null95 {p95:+.3f}  "
                  f"folds {better}/7 {flag}", flush=True)
    json.dump(report, open("outputs/v5/s2_cue_utility.json", "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main(sys.argv[1:] or [t for t in TARGETS if t != "retention"])
