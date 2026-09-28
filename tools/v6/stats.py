"""Paired sequence bootstrap (frozen protocol: 10,000 resamples, seed 42,
percentile 95% CI, P(Δ ≤ 0)), identical in method to tools/heldout_run.py,
for the internal (tools.seqstats.combine) and the official-compatible
(tools.v6.eval_official.combine_official) protocols."""
from __future__ import annotations

import numpy as np


def bootstrap(stats_a, stats_b, metric, combine, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    k = len(stats_a)
    base = combine(stats_a)[metric] - combine(stats_b)[metric]
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, k)
        diffs[i] = (combine([stats_a[j] for j in idx])[metric]
                    - combine([stats_b[j] for j in idx])[metric])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(diff=float(base), ci_lo=float(lo), ci_hi=float(hi),
                p_le0=float((diffs <= 0).mean()))
