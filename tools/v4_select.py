"""
V4 nested sequence-level selection + parameter stability report
(research/OPTIMIZATION_PROTOCOL.md, Amendment 4).
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

os.environ.setdefault("ACMOT_GRID", "v4")

from tools.optimize_protocol import (DETECTORS, OUT, grid, load_all,  # noqa
                                     metrics_on, n_catastrophic, sequences)
from tools.seqstats import combine  # noqa: E402


def key(S, cid, seqs, ref):
    per = {d: metrics_on(S, cid, d, seqs) for d in DETECTORS}
    if not all(m["MOTA"] > 0 for m in per.values()):
        return (-10 ** 6, -np.inf)
    q = {d: 0.5 * (per[d]["HOTA"] + per[d]["IDF1"]) for d in DETECTORS}
    j4 = min((q[d] - ref[d]) / ref[d] for d in DETECTORS)
    return (-n_catastrophic(S, cid, seqs), j4)


def main():
    S = load_all()
    seqs = sequences()
    cfgs = grid()
    ids = [c["id"] for c in cfgs]
    P = {c["id"]: c["params"] for c in cfgs}

    def ref_on(ss):
        out = {}
        for d in DETECTORS:
            m = metrics_on(S, "gate_r0.0", d, ss, "outputs/opt")
            out[d] = 0.5 * (m["HOTA"] + m["IDF1"])
        return out

    folds = []
    for held in seqs:
        tr = [s for s in seqs if s != held]
        ref = ref_on(tr)
        keys = {c: key(S, c, tr, ref) for c in ids}
        best = max(ids, key=lambda c: keys[c])
        folds.append(dict(held_out=held, chosen=best, params=P[best],
                          key=list(keys[best])))
        print(f"fold {held[:12]}: {best}  key={keys[best]}", flush=True)
    cv = {d: combine([S[(f["chosen"], d, f["held_out"])] for f in folds])
          for d in DETECTORS}
    ref7 = ref_on(seqs)
    keys7 = {c: key(S, c, seqs, ref7) for c in ids}
    final = max(ids, key=lambda c: keys7[c])
    final_m = {d: metrics_on(S, final, d, seqs) for d in DETECTORS}
    print("\nFINAL (all 7):", final, keys7[final])
    for d in DETECTORS:
        a, b = cv[d], final_m[d]
        print(f"  {d:<7} outer-CV MOTA {a['MOTA']:6.2f} HOTA {a['HOTA']:6.2f} "
              f"IDF1 {a['IDF1']:6.2f} IDS {a['IDS']:5d} | in-sample MOTA "
              f"{b['MOTA']:6.2f} HOTA {b['HOTA']:6.2f} IDF1 {b['IDF1']:6.2f} "
              f"IDS {b['IDS']:5d}")

    print("\nParameter stability across the 7 outer folds:")
    stab = {}
    for name in P[final]:
        vals = [f["params"][name] for f in folds]
        cnt = collections.Counter(vals)
        num = all(isinstance(v, (int, float)) for v in vals)
        stab[name] = dict(fold_values=vals, counts=dict(cnt),
                          final=P[final][name])
        extra = (f" median {np.median(vals)} range [{min(vals)}, "
                 f"{max(vals)}]") if num else ""
        print(f"  {name:<8} folds={vals}{extra}  freq={dict(cnt)}  "
              f"final={P[final][name]}")

    print("\nObjective sensitivity around the final config "
          "(one parameter changed, others fixed; all 7 sequences):")
    sens = {}
    for name in P[final]:
        rows = []
        for c in ids:
            if all(P[c][k] == P[final][k] for k in P[final] if k != name):
                k = keys7[c]
                rows.append((P[c][name], -k[0], k[1]))
        rows.sort(key=lambda r: str(r[0]))
        sens[name] = rows
        print(f"  {name:<8} " + "  ".join(
            f"{v}: ncat {n} J4 {j:+.3f}" for v, n, j in rows))
    json.dump(dict(folds=folds, final=final, final_params=P[final],
                   cv={d: cv[d] for d in DETECTORS},
                   in_sample=final_m, stability=stab,
                   sensitivity={k: [list(map(str, r)) for r in v]
                                for k, v in sens.items()}),
              open(OUT / "v4_selection.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
