"""
G2 oracle gate (GT-assisted, development split only; never a controller).

For a set of static profiles P ('R<px>K<k>'), per detector:
  1. static frontier: pooled HOTA and mean normalized cost of every static
     run '<layer>+<p>'; Pareto-efficient profiles;
  2. anchors: for each target cost, the frontier profile closest to it;
  3. oracle: one profile per 30-frame segment, Lagrangian allocation of the
     per-frame quality q = TP - FP - IDS of the static runs, mean segment cost
     <= the anchor's realized cost;
  4. schedules written for the runner: SCHED:STAT<tag> (the anchor every frame)
     and SCHED:ORB<tag> (the oracle), so that both detectors share one system
     name in the paired bootstrap.

  python tools/g2/oracle.py <layer> <space-name> <targets, e.g. 0.35,0.5,0.7,1.0,1.4> <profiles...>
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.g2 import dev  # noqa: E402  (val-7 guard)

SEG = 30


def frame_quality(system, det, seq):
    import motmetrics as mm
    from tools.eval_local import load_gt
    from tools.seqstats import apply_ignore_regions
    SPLITS, _ = dev._splits()
    data = Path(SPLITS[dev.SPLIT]["data"])
    st = dev.load(system, det, seq)
    n = len(st["audit"])
    gt = load_gt(data / "annotations" / f"{seq}.txt")
    txt = st["tracks_txt"]
    tr = np.loadtxt(io.StringIO(txt), delimiter=",", ndmin=2) if txt else np.zeros((0, 10))
    tr = apply_ignore_regions(data, seq, tr)
    acc = mm.MOTAccumulator(auto_id=True)
    for t in range(1, n + 1):
        g = gt[gt[:, 0] == t]
        p = tr[tr[:, 0] == t] if len(tr) else tr
        acc.update(g[:, 1].astype(int).tolist(), p[:, 1].astype(int).tolist(),
                   mm.distances.iou_matrix(g[:, 2:6], p[:, 2:6], max_iou=0.5))
    ev = acc.mot_events.reset_index()
    q = np.zeros(n)
    for typ, w in (("MATCH", 1), ("FP", -1)):
        np.add.at(q, ev.loc[ev["Type"] == typ, "FrameId"].to_numpy().astype(int), w)
    cost = np.array([a["cost"] for a in st["audit"]])
    return q, cost


def _item(p):
    m = dev.STATIC.match(p)
    return [f"R{m.group(1)}" + (f"T{m.group(2)}" if m.group(2) else ""), int(m.group(3))]


def frontier(layer, det, profiles):
    pts = {}
    for p in profiles:
        m = dev.summary(f"{layer}+{p}", [det])[det]
        pts[p] = (m["ops"]["mean_cost"], m["HOTA"])
    eff = [p for p in profiles if not any(pts[o][0] <= pts[p][0] and pts[o][1] > pts[p][1] for o in profiles)]
    return pts, sorted(eff, key=lambda p: pts[p][0])


def allocate(Q, C, budget):
    """Q[s][p], C[s][p] (summed over the segment); mean cost per frame <= budget."""
    n = np.array([C[s]["_n"] for s in range(len(Q))], float)
    prof = [p for p in Q[0] if p != "_n"]

    def pick(lam):
        return [max(prof, key=lambda p: Q[s][p] - lam * C[s][p]) for s in range(len(Q))]

    def mean_cost(sel):
        return sum(C[s][sel[s]] for s in range(len(Q))) / n.sum()
    if mean_cost(pick(0.0)) <= budget:
        return pick(0.0)
    lo, hi = 0.0, 1e7
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_cost(pick(mid)) > budget else (lo, mid)
    return pick(hi)


def build(layer, space, targets, profiles):
    _, seqs_of = dev._splits()
    report = {}
    for det in os.environ.get("V7_DETS", "yolov8,rtdetr").split(","):
        pts, eff = frontier(layer, det, profiles)
        anchors = {}
        for tg in targets:
            a = min(eff, key=lambda p: abs(pts[p][0] - tg))
            anchors[f"{space}{int(round(100 * tg))}"] = a
        report[det] = dict(points={p: dict(cost=pts[p][0], HOTA=pts[p][1]) for p in profiles}, efficient=eff,
                           anchors=anchors)
        for seq in seqs_of(dev.SPLIT):
            qs = {p: frame_quality(f"{layer}+{p}", det, seq) for p in profiles}
            N = len(next(iter(qs.values()))[0])
            segs = [(i, min(i + SEG, N)) for i in range(0, N, SEG)]
            Q = [{p: float(qs[p][0][a:b].sum()) for p in profiles} for a, b in segs]
            C = [dict({p: float(qs[p][1][a:b].sum()) for p in profiles}, _n=b - a) for a, b in segs]
            for s in range(len(Q)):
                Q[s]["_n"] = 0.0
            f = dev.out_dir() / "schedules" / layer / det / f"{seq}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            old = json.loads(f.read_text()) if f.exists() else {}
            for tag, a in anchors.items():
                stat_cost = float(np.mean(qs[a][1]))
                sel = allocate(Q, C, stat_cost)
                items = []
                for k, (lo, hi) in enumerate(segs):
                    items += [_item(sel[k])] * (hi - lo)
                old[f"STAT{tag}"] = [_item(a)] * N
                old[f"ORB{tag}"] = items
            f.write_text(json.dumps(old))
    out = dev.out_dir() / "schedules" / f"oracle_{layer}_{space}.json"
    out.write_text(json.dumps(report, indent=1))
    for det, r in report.items():
        print(det, "efficient:", ", ".join(f"{p}({r['points'][p]['cost']:.3f}, {r['points'][p]['HOTA']:.2f})" for p in r["efficient"]))
        print(det, "anchors:", r["anchors"])


if __name__ == "__main__":
    layer, space, tg = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(",")]
    build(layer, space, tg, sys.argv[4:])
