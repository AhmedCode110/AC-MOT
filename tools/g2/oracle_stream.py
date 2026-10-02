"""
G2 oracle for the persistent-regime unit: ONE static profile per sequence
(stream-level compute calibration), GT-assisted, development split only.

Per detector: Lagrangian allocation of a profile to each sequence from the
sequence-level MOTA numerator of the static runs, mean per-frame cost over
all frames <= the anchor's cost; the oracle system is assembled from the
existing static runs (no new tracking run is needed because the profile is
constant within a sequence). Paired bootstrap against the anchor static
profile and against the concave static envelope.

  python tools/g2/oracle_stream.py <layer> <targets> <profiles...>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.g2 import dev  # noqa: E402
from tools.g2.gate_table import cost_for, hull, interp_h  # noqa: E402
from tools.g2.oracle import frame_quality, frontier  # noqa: E402


def main(layer, targets, profiles):
    from tools.v7.bootstrap import paired
    _, seqs_of = dev._splits()
    seqs = seqs_of(dev.SPLIT)
    dets = ["yolov8", "rtdetr"]
    res = {}
    cells_A, cells_B = {}, {}
    for det in dets:
        pts, eff = frontier(layer, det, profiles)
        env = hull([v for v in pts.values()])
        st = {(s, p): dev.load(f"{layer}+{p}", det, s) for s in seqs for p in profiles}
        Q = {(s, p): float(frame_quality(f"{layer}+{p}", det, s)[0].sum()) for s in seqs for p in profiles}
        C = {(s, p): float(sum(a["cost"] for a in st[(s, p)]["audit"])) for s in seqs for p in profiles}
        N = sum(len(st[(s, profiles[0])]["audit"]) for s in seqs)
        for tg in targets:
            anchor = min(eff, key=lambda p: abs(pts[p][0] - tg))
            budget = pts[anchor][0]

            def pick(lam):
                return {s: max(profiles, key=lambda p: Q[(s, p)] - lam * C[(s, p)]) for s in seqs}

            def mean_cost(sel):
                return sum(C[(s, sel[s])] for s in seqs) / N
            lo, hi = 0.0, 1e7
            sel = pick(0.0)
            if mean_cost(sel) > budget:
                for _ in range(100):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if mean_cost(pick(mid)) > budget else (lo, mid)
                sel = pick(hi)
            A = [st[(s, anchor)] for s in seqs]
            B = [st[(s, sel[s])] for s in seqs]
            c = mean_cost(sel)
            hb = dev.combine(B)["HOTA"]
            key = f"T{int(round(100 * tg))}"
            res.setdefault(key, {})[det] = dict(anchor=anchor, anchor_cost=budget, oracle_cost=c, oracle_HOTA=hb,
                                                anchor_HOTA=pts[anchor][1], envelope_HOTA=interp_h(env, c),
                                                envelope_cost_ratio=cost_for(env, hb) / c, selection=sel)
            cells_A.setdefault(key, []).extend(A)
            cells_B.setdefault(key, []).extend(B)
    print("| budget | detector | anchor (cost, HOTA) | oracle cost | oracle HOTA | Δ vs envelope | cost ratio |")
    print("|---|---|---|---|---|---|---|")
    for key in res:
        for det, r in res[key].items():
            print(f"| {key} | {det} | {r['anchor']} ({r['anchor_cost']:.3f}, {r['anchor_HOTA']:.2f}) | {r['oracle_cost']:.3f} | "
                  f"{r['oracle_HOTA']:.2f} | {r['oracle_HOTA'] - r['envelope_HOTA']:+.2f} | {r['envelope_cost_ratio']:.2f} |")
        b = paired(cells_A[key], cells_B[key], dev.combine, 10000, 42, ["HOTA", "IDF1", "MOTA"])
        res[key]["pooled_vs_anchor"] = b
        print(f"| {key} | pooled vs anchor | | | | HOTA {b['HOTA']['diff']:+.2f} [{b['HOTA']['ci_lo']:+.2f}, "
              f"{b['HOTA']['ci_hi']:+.2f}], IDF1 {b['IDF1']['diff']:+.2f}, MOTA {b['MOTA']['diff']:+.2f} | |")
    out = ROOT / "research/final/g2/oracle" / f"stream_{layer}_{len(profiles)}profiles.json"
    out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], [float(x) for x in sys.argv[2].split(",")], sys.argv[3:])
