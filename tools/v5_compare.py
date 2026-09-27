"""Pooled outer-CV comparison of V5 variants vs V4 (both at 736, identical
compute), with per-sequence paired differences and a sequence bootstrap."""
from __future__ import annotations

import os

import pickle
import sys
from pathlib import Path

import numpy as np

from tools.seqstats import combine

DATASET = Path(os.environ.get(
    "V5_DATASET", "/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val"))
SEQS = sorted(p.name for p in (DATASET / "sequences").iterdir() if p.is_dir())
DETS = ["yolov8", "rtdetr"]


def v4(det, s):
    return pickle.load(open(f"outputs/opt_v4sens/stats/s_ref/{det}/{s}.pkl",
                            "rb"))


def v5(variant, det, s, part="outer"):
    import os
    root = os.environ.get("V5_S3_OUT", "outputs/v5/s3")
    return pickle.load(open(f"{root}/{variant}/{part}/{det}/{s}.pkl", "rb"))


def q(m):
    return 0.5 * (m["HOTA"] + m["IDF1"])


def boot(a, b, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    k = len(a)
    d = [q(combine([a[i] for i in idx])) - q(combine([b[i] for i in idx]))
         for idx in (rng.integers(0, k, k) for _ in range(n))]
    return np.percentile(d, [2.5, 97.5]), float(np.mean(np.asarray(d) <= 0))


def main(variants, part="outer"):
    for det in DETS:
        base = [v4(det, s) for s in SEQS]
        mb = combine(base)
        ncb = sum(combine([x])["MOTA"] < 0 for x in base)
        print(f"{det}: V4_736          MOTA {mb['MOTA']:6.2f} HOTA {mb['HOTA']:6.2f}"
              f" IDF1 {mb['IDF1']:6.2f} IDS {mb['IDS']:4d} R {mb['Recall']:5.1f} ncat {ncb}")
        for v in variants:
            x = [v5(v, det, s, part) for s in SEQS]
            m = combine(x)
            nc = sum(combine([y])["MOTA"] < 0 for y in x)
            per = [q(combine([x[i]])) - q(combine([base[i]]))
                   for i in range(len(SEQS))]
            ci, p = boot(x, base)
            print(f"{det}: V5[{v:<14}] MOTA {m['MOTA']:6.2f} HOTA {m['HOTA']:6.2f}"
                  f" IDF1 {m['IDF1']:6.2f} IDS {m['IDS']:4d} R {m['Recall']:5.1f} "
                  f"ncat {nc} | Δq {q(m) - q(mb):+.2f} CI [{ci[0]:+.2f},"
                  f"{ci[1]:+.2f}] P(Δ≤0) {p:.2f} | seq Δq≥0 "
                  f"{sum(d >= 0 for d in per)}/7")


if __name__ == "__main__":
    args = sys.argv[1:]
    part = "outer"
    if args and args[0] in ("outer", "final"):
        part, args = args[0], args[1:]
    main(args or ["full"], part)
