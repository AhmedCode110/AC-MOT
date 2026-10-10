"""
Headroom diagnostic for scene-dependent compute allocation (uses GT; never
a candidate controller). For each (layer, detector, sequence) and each
fixed level, per-frame CLEAR events of the static run '<layer>+<LEVEL>' give
a frame quality q = TP - FP - IDS (the MOTA numerator). Summed over 30-frame
segments, they rank the segments by what a higher level buys there.

Schedules (per sequence, written to outputs/sci_v7/val7/oracle/):
  ORACLE   the level counts of '<layer>+SCI' (same compute as SCI and as
           PERMk), levels assigned by segment benefit instead of by SCI
  ORACLEM  compute equal to fixed MEDIUM: HIGH segments paired with enough
           LOW frames to pay for them, taken while the paired benefit is
           positive

  python tools/sci_v7/oracle.py <layer> [<layer> ...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sci_v7 import dev  # noqa: E402  (guard: val-7 only)

SEG = dev.SEGMENT


def frame_quality(system, det, seq):
    import io
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
    # auto_id frame ids are 0..n-1. TP = MATCH + SWITCH and IDS = SWITCH, so
    # TP - FP - IDS = MATCH - FP.
    ev = acc.mot_events.reset_index()
    q = np.zeros(n)
    for typ, w in (("MATCH", 1), ("FP", -1)):
        np.add.at(q, ev.loc[ev["Type"] == typ, "FrameId"].to_numpy().astype(int), w)
    return q


def segments(n):
    return [(i, min(i + SEG, n)) for i in range(0, n, SEG)]


def oracle_same_counts(qL, qM, qH, levels):
    n = len(levels)
    segs = segments(n)
    nH, nL = levels.count("HIGH"), levels.count("LOW")
    gH = [qH[a:b].sum() - qM[a:b].sum() for a, b in segs]
    lossL = [qM[a:b].sum() - qL[a:b].sum() for a, b in segs]
    out = ["MEDIUM"] * n
    used = set()
    left = nH
    for k in np.argsort(gH)[::-1]:
        if left <= 0:
            break
        a, b = segs[k]
        take = min(left, b - a)
        out[a:a + take] = ["HIGH"] * take
        left -= take
        used.add(k)
    left = nL
    for k in np.argsort(lossL):
        if left <= 0:
            break
        if k in used:
            continue
        a, b = segs[k]
        take = min(left, b - a)
        out[a:a + take] = ["LOW"] * take
        left -= take
    if left > 0:   # not enough MEDIUM segments: fill LOW into HIGH segments' remainder
        for i in range(n):
            if left and out[i] == "MEDIUM":
                out[i] = "LOW"
                left -= 1
    assert out.count("HIGH") == nH and out.count("LOW") == nL
    return out


def oracle_medium_budget(qL, qM, qH, n, ratio):
    """HIGH segments paid by LOW frames (ratio LOW frames per HIGH frame)."""
    segs = segments(n)
    gH = sorted(((qH[a:b] - qM[a:b]).mean(), k) for k, (a, b) in enumerate(segs))[::-1]
    lossL = sorted(((qM[a:b] - qL[a:b]).mean(), k) for k, (a, b) in enumerate(segs))
    out = ["MEDIUM"] * n
    used = set()
    li = 0
    for g, k in gH:
        if k in used:
            continue
        a, b = segs[k]
        need = int(np.ceil(ratio * (b - a)))
        pay, cost, picks = 0, 0.0, []
        j = li
        while pay < need and j < len(lossL):
            l, m = lossL[j]
            j += 1
            if m in used or m == k:
                continue
            ma, mb = segs[m]
            picks.append(m)
            pay += mb - ma
            cost += l * (mb - ma)
        if pay < need or g * (b - a) - cost * need / max(pay, 1) <= 0:
            break
        out[a:b] = ["HIGH"] * (b - a)
        used.add(k)
        rem = need
        for m in picks:
            ma, mb = segs[m]
            take = min(rem, mb - ma)
            out[ma:ma + take] = ["LOW"] * take
            rem -= take
            used.add(m)
        li = j
    return out


def build(layer):
    prof = json.loads((ROOT / "configs/sci_v7_profiles.json").read_text())
    _, seqs_of = dev._splits()
    for det in os.environ.get("V7_DETS", "yolov8,rtdetr").split(","):
        r = prof["detectors"][det][dev.PROFILE]
        ratio = (r["HIGH"] ** 2 - r["MEDIUM"] ** 2) / (r["MEDIUM"] ** 2 - r["LOW"] ** 2)
        for seq in seqs_of(dev.SPLIT):
            q = {lv: frame_quality(f"{layer}+{lv}", det, seq) for lv in ("LOW", "MEDIUM", "HIGH")}
            sci = [a["level"] for a in dev.load(f"{layer}+SCI", det, seq)["audit"]]
            sched = dict(ORACLE=oracle_same_counts(q["LOW"], q["MEDIUM"], q["HIGH"], sci),
                         ORACLEM=oracle_medium_budget(q["LOW"], q["MEDIUM"], q["HIGH"], len(sci), ratio))
            d = dev.out_dir() / "oracle" / layer / det
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{seq}.json").write_text(json.dumps(sched))
            c = {k: {lv: v.count(lv) for lv in ("LOW", "MEDIUM", "HIGH")} for k, v in sched.items()}
            print(layer, det, seq, c)




# ---- budget frontier over every cached resolution --------------------------
def budget_schedule(Q, n, costs, budget):
    """Q[s][r]: segment quality at resolution r; n[s]: frames of segment s;
    costs[r]: per-frame cost. Lagrangian allocation (exact on the convex hull)
    of one resolution per segment with mean per-frame cost <= budget."""
    res = sorted(costs)

    def pick(lam):
        return [max(res, key=lambda r: Q[s][r] - lam * n[s] * costs[r]) for s in range(len(n))]

    def mean_cost(p):
        return sum(n[s] * costs[p[s]] for s in range(len(n))) / sum(n)
    lo, hi = 0.0, 1e6
    if mean_cost(pick(0.0)) <= budget:
        return pick(0.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        if mean_cost(pick(mid)) > budget:
            lo = mid
        else:
            hi = mid
    return pick(hi)


def build_budget(layer, budgets, resolutions, ref=736):
    _, seqs_of = dev._splits()
    costs = {r: (r / ref) ** 2 for r in resolutions}
    for det in os.environ.get("V7_DETS", "yolov8,rtdetr").split(","):
        for seq in seqs_of(dev.SPLIT):
            q = {r: frame_quality(f"{layer}+R{r}", det, seq) for r in resolutions}
            N = len(q[resolutions[0]])
            segs = segments(N)
            Q = [{r: float(q[r][a:b].sum()) for r in resolutions} for a, b in segs]
            n = [b - a for a, b in segs]
            out = {}
            for B in budgets:
                p = budget_schedule(Q, n, costs, B)
                out[f"ORACLEB{int(round(100 * B))}"] = [p[k] for k, (a, b) in enumerate(segs) for _ in range(a, b)]
            d = dev.out_dir() / "oracle" / layer / det
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"{seq}.json"
            old = json.loads(f.read_text()) if f.exists() else {}
            old.update(out)
            f.write_text(json.dumps(old))
            print(layer, det, seq, {k: round(float(np.mean([costs[r] for r in v])), 3) for k, v in out.items()})


if __name__ == "__main__":
    if sys.argv[1] == "budget":       # oracle.py budget <layer> <B1,B2,..> <r1,r2,..>
        build_budget(sys.argv[2], [float(x) for x in sys.argv[3].split(",")],
                     [int(x) for x in sys.argv[4].split(",")])
    else:
        for layer in sys.argv[1:]:
            build(layer)
