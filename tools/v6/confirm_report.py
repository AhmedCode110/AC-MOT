"""Post-freeze report for one split: pooled metrics, catastrophic cells,
per-sequence table, paired sequence bootstrap (10,000, seed 42) under the
internal and the official-compatible protocols. Writes JSON/CSV next to the
runs; computes nothing that could feed back into the frozen policy.

  V6_SPLIT=conf16 python tools/v6/confirm_report.py V6TF V4 shared_static ...
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.v6.dev import DETS, load, out_dir, split_sequences  # noqa: E402

PAIRS = [("V6TF", "V4"), ("V6TF", "shared_static")]
METRICS = ("HOTA", "IDF1", "MOTA")


def _stats(split, sy, d, proto):
    seqs = split_sequences(split)
    if proto == "internal":
        return [load(split, sy, d, s) for s in seqs]
    return [pickle.load(open(out_dir(split) / sy / d / f"{s}.official.pkl", "rb"))
            for s in seqs]


def _boot(job):
    split, a, b, d, proto, n = job
    os.chdir(ROOT)
    import numpy as np
    from tools.seqstats import combine
    from tools.v6.eval_official import combine_official
    comb = combine if proto == "internal" else combine_official
    A, B = _stats(split, a, d, proto), _stats(split, b, d, proto)
    rng = np.random.default_rng(42)
    k = len(A)
    ma, mb = comb(A), comb(B)
    diffs = {m: np.empty(n) for m in METRICS}
    for i in range(n):
        idx = rng.integers(0, k, k)
        ra, rb = comb([A[j] for j in idx]), comb([B[j] for j in idx])
        for m in METRICS:
            diffs[m][i] = ra[m] - rb[m]
    out = []
    for m in METRICS:
        lo, hi = np.percentile(diffs[m], [2.5, 97.5])
        out.append(dict(protocol=proto, system=a, baseline=b, detector=d, metric=m,
                        diff=float(ma[m] - mb[m]), ci_lo=float(lo), ci_hi=float(hi),
                        p_le0=float((diffs[m] <= 0).mean()), n_boot=n, seed=42))
    return out


def main():
    split = os.environ.get("V6_SPLIT", "conf16")
    systems = sys.argv[1:]
    n = int(os.environ.get("V6_NBOOT", "10000"))
    from tools.seqstats import combine
    from tools.v6.eval_official import combine_official
    seqs = split_sequences(split)
    pooled, rows = [], []
    for proto, comb in (("internal", combine), ("official", combine_official)):
        for sy in systems:
            for d in DETS:
                st = _stats(split, sy, d, proto)
                m = comb(st)
                per = [comb([x]) for x in st]
                ncat = sum(p["MOTA"] < 0 for p in per)
                pooled.append(dict(protocol=proto, system=sy, detector=d,
                                   n_catastrophic=ncat,
                                   **{k: v for k, v in m.items() if k != "GT"}))
                for s, p in zip(seqs, per):
                    rows.append(dict(protocol=proto, system=sy, detector=d, sequence=s,
                                     **{k: (round(v, 3) if isinstance(v, float) else v)
                                        for k, v in p.items()}))
    od = out_dir(split)
    json.dump(pooled, open(od / "pooled_metrics.json", "w"), indent=1)
    with open(od / "per_sequence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["GT"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    jobs = [(split, a, b, d, proto, n) for a, b in PAIRS if a in systems and b in systems
            for d in DETS for proto in ("internal", "official")]
    with ProcessPoolExecutor(6) as ex:
        boots = [r for res in ex.map(_boot, jobs) for r in res]
    json.dump(boots, open(od / "bootstrap.json", "w"), indent=1)
    for p in pooled:
        print(f"{p['protocol']:<9}{p['system']:<15}{p['detector']:<7} cat {p['n_catastrophic']:2d} "
              f"MOTA {p['MOTA']:6.2f} HOTA {p['HOTA']:6.2f} IDF1 {p['IDF1']:6.2f} "
              f"IDS {p['IDS']:5d} FP {p['FP']:6d} FN {p['FN']:6d} "
              f"P {p['Precision']:5.1f} R {p['Recall']:5.1f}")
    print(f"\nPaired sequence bootstrap ({n} resamples, seed 42, 95% CI)")
    for b in boots:
        print(f"  {b['protocol']:<9}{b['detector']:<7}{b['system']} vs {b['baseline']:<14}"
              f"{b['metric']:<5} Δ {b['diff']:6.2f} CI [{b['ci_lo']:6.2f}, {b['ci_hi']:6.2f}] "
              f"P(Δ≤0) {b['p_le0']:.3f}")


if __name__ == "__main__":
    main()
