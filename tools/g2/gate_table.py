"""
Oracle-gate table for G2 adaptation spaces (from the run outputs):
per detector, the oracle schedule's realized cost and HOTA, the static
frontier's HOTA interpolated at that cost, and the cost ratio at which the
static frontier reaches the oracle's HOTA; pooled paired bootstrap against
the anchor static profile.

  python tools/g2/gate_table.py <layer> <space> <bootstrap.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.g2 import dev  # noqa: E402


def hull(points):
    """Upper concave envelope of the static points, monotone in cost: every
    point on it is reachable by time-sharing two static profiles (a
    scene-blind mixture), so it is the matched-compute static baseline."""
    pts = sorted(set(points))
    up = []
    for p in pts:
        while len(up) >= 2 and ((up[-1][0] - up[-2][0]) * (p[1] - up[-2][1])
                                - (up[-1][1] - up[-2][1]) * (p[0] - up[-2][0])) >= 0:
            up.pop()
        up.append(p)
    out = []
    for c, h in up:
        if not out or h > out[-1][1]:
            out.append((c, h))
    return out


def interp_h(front, c):
    cs, hs = zip(*front)
    return float(np.interp(c, cs, hs))


def cost_for(front, h):
    cs, hs = zip(*front)
    if h > hs[-1]:
        return float("inf")
    return float(np.interp(h, hs, cs))


def main(layer, space, boot_json):
    rep = json.loads((dev.out_dir() / "schedules" / f"oracle_{layer}_{space}.json").read_text())
    boots = {r["B"].split(":")[-1]: r for r in json.loads(Path(boot_json).read_text())}
    print(f"| budget tag | detector | oracle cost | oracle HOTA | static envelope HOTA at that cost | Δ vs envelope | "
          f"envelope cost for the oracle HOTA / oracle cost |")
    print("|---|---|---|---|---|---|---|")
    for tag in sorted(next(iter(rep.values()))["anchors"], key=lambda t: int(t.split("_")[-1])):
        for det, r in rep.items():
            front = hull([(v["cost"], v["HOTA"]) for v in r["points"].values()])
            m = dev.summary(f"{layer}+SCHED:ORB{tag}", [det])[det]
            c, h = m["ops"]["mean_cost"], m["HOTA"]
            fh = interp_h(front, c)
            fc = cost_for(front, h)
            print(f"| {tag} | {det} | {c:.3f} | {h:.2f} | {fh:.2f} | {h - fh:+.2f} | "
                  f"{'∞' if fc == float('inf') else f'{fc / c:.2f}'} |")
        b = boots.get(f"ORB{tag}")
        if b:
            p = b["pooled_cells"]
            print(f"| {tag} | pooled vs anchor | | | | HOTA {p['HOTA']['diff']:+.2f} [{p['HOTA']['ci_lo']:+.2f}, "
                  f"{p['HOTA']['ci_hi']:+.2f}]; IDF1 {p['IDF1']['diff']:+.2f}; MOTA {p['MOTA']['diff']:+.2f} | |")


if __name__ == "__main__":
    main(*sys.argv[1:4])
