"""
E12 -- ID-switch / membership-churn diagnostics for the VisDrone ByteTrack
host (reads outputs of `tools/v7/dev.py run` (.pkl) or `track` (.trk.pkl)).

Label-free part (per frame, from the host output + the layer audit):
  births      track ids appearing for the first time at t
  drops       ids output at t-1 and not at t
  returns     ids output at t that were absent at t-1 but seen before
  churn       (births + drops) / max(1, |ids at t-1|)
  short       tracks whose whole life is < 5 frames (fragments)
  oscillation |assoc_t - assoc_{t-1}| > 0.05 or a regime change
  band move   |t2_t - t2_{t-1}| (logit scale)
Each quantity is split by the regime of the frame (clean / noisy / cold)
and by proximity (<= 2 frames) to a regime change or a threshold jump.

Label-based part (only when the dataset annotations exist; offline
diagnosis, never used by the layer): the motmetrics accumulator of the
reference protocol is replayed and every SWITCH event is attributed to its
frame, so the per-frame switches sum EXACTLY to the reported IDS.

  python tools/v7/diag_churn.py <split> <det> <system> [<system> ...]
"""
from __future__ import annotations

import io
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _tracks(txt):
    if not txt:
        return np.zeros((0, 10))
    return np.loadtxt(io.StringIO(txt), delimiter=",", ndmin=2)


def load_run(split, system, det, seq):
    base = ROOT / "outputs/v7" / split / system / det
    for suf in (".pkl", ".trk.pkl"):
        p = base / f"{seq}{suf}"
        if p.exists():
            return pickle.load(open(p, "rb"))
    raise FileNotFoundError(base / f"{seq}.pkl")


def frame_features(run):
    """Per-frame label-free features (list of dicts, frame 1..n)."""
    audit = run["audit"]
    n = len(audit)
    tr = _tracks(run["tracks_txt"])
    ids = [set() for _ in range(n + 1)]
    for f, i in tr[:, :2].astype(int) if len(tr) else []:
        ids[f].add(i)
    life = defaultdict(int)
    for f in range(1, n + 1):
        for i in ids[f]:
            life[i] += 1
    short = {i for i, c in life.items() if c < 5}
    seen, feats = set(), []
    prev_a = prev_t2 = prev_reg = None
    for f in range(1, n + 1):
        a = audit[f - 1]
        cur, prv = ids[f], ids[f - 1]
        births = cur - seen
        drops = prv - cur
        returns = (cur - prv) & seen
        seen |= cur
        reg = a.get("regime", "na")
        assoc = float(a.get("assoc", np.nan))
        t2 = float(a.get("t2", np.nan))
        feats.append(dict(
            frame=f, regime=reg, n=len(cur), births=len(births), drops=len(drops),
            returns=len(returns), short_births=len(births & short),
            churn=(len(births) + len(drops)) / max(1, len(prv)),
            regime_change=prev_reg is not None and reg != prev_reg,
            assoc_jump=prev_a is not None and abs(assoc - prev_a) > 0.05,
            band_move=abs(t2 - prev_t2) if prev_t2 is not None and np.isfinite(t2 + prev_t2) else np.nan,
            n_dup=int(a.get("n_dup", 0)), n_pass=int(a.get("n_pass", 0)),
            match=float(a.get("match", np.nan)), motion_ratio=float(a.get("motion_ratio", np.nan)),
            assoc=assoc))
        prev_a, prev_t2, prev_reg = assoc, t2, reg
    return feats


def switch_frames(dataset, seq, txt):
    """Frame of every ID switch of the reference protocol (motmetrics)."""
    import motmetrics as mm
    from tools.eval_local import load_gt
    from tools.seqstats import apply_ignore_regions
    dataset = Path(dataset)
    gt = load_gt(dataset / "annotations" / f"{seq}.txt")
    tr = apply_ignore_regions(dataset, seq, _tracks(txt))
    n = len(list((dataset / "sequences" / seq).glob("*.jpg")))
    acc = mm.MOTAccumulator(auto_id=True)
    for t in range(1, n + 1):
        g = gt[gt[:, 0] == t]
        p = tr[tr[:, 0] == t] if len(tr) else tr
        acc.update(g[:, 1].astype(int).tolist(), p[:, 1].astype(int).tolist(),
                   mm.distances.iou_matrix(g[:, 2:6], p[:, 2:6], max_iou=0.5))
    ev = acc.mot_events
    fr = ev[ev["Type"] == "SWITCH"].index.get_level_values(0).to_numpy() + 1   # auto_id is 0-based
    return np.bincount(fr, minlength=n + 1)[1:n + 1]


def near(mask, k=2):
    m = np.asarray(mask, bool)
    out = m.copy()
    for d in range(1, k + 1):
        out[d:] |= m[:-d]
    return out


def summarise(split, det, system, seqs, dataset=None):
    rows = []
    for seq in seqs:
        run = load_run(split, system, det, seq)
        F = frame_features(run)
        sw = None
        if dataset is not None and (Path(dataset) / "annotations" / f"{seq}.txt").exists():
            sw = switch_frames(dataset, seq, run["tracks_txt"])
        for i, x in enumerate(F):
            x["seq"] = seq
            x["ids"] = int(sw[i]) if sw is not None else None
        rows += F
    reg = np.array([r["regime"] for r in rows])
    unstable = near([r["regime_change"] or r["assoc_jump"] for r in rows])
    out = dict(system=system, det=det, frames=len(rows))
    for key, m in (("all", np.ones(len(rows), bool)), ("clean", reg == "clean"),
                   ("noisy", reg == "noisy"), ("cold", reg == "cold"),
                   ("near_change", unstable), ("stable", ~unstable)):
        sub = [r for r, k in zip(rows, m) if k]
        if not sub:
            continue
        d = dict(frames=len(sub),
                 births=int(sum(r["births"] for r in sub)),
                 drops=int(sum(r["drops"] for r in sub)),
                 returns=int(sum(r["returns"] for r in sub)),
                 short_births=int(sum(r["short_births"] for r in sub)),
                 churn=float(np.mean([r["churn"] for r in sub])),
                 tracks=float(np.mean([r["n"] for r in sub])),
                 n_dup=float(np.mean([r["n_dup"] for r in sub])))
        if rows[0]["ids"] is not None:
            d["ids"] = int(sum(r["ids"] for r in sub))
        out[key] = d
    out["regime_changes"] = int(sum(r["regime_change"] for r in rows))
    out["assoc_jumps"] = int(sum(r["assoc_jump"] for r in rows))
    bm = [r["band_move"] for r in rows if np.isfinite(r["band_move"])]
    out["band_move_mean"] = float(np.mean(bm)) if bm else float("nan")
    return out, rows


if __name__ == "__main__":
    import os
    from tools.v7.dev import SPLITS, split_sequences
    split, det, *systems = sys.argv[1:]
    seqs = split_sequences(split)
    ds = SPLITS[split]["data"]
    ds = ds if (Path(ds) / "annotations").exists() else None
    res = {}
    for sy in systems:
        s, _ = summarise(split, det, sy, seqs, ds)
        res[sy] = s
        print(json.dumps(s))
    dest = os.environ.get("V7_DIAG_OUT")
    if dest:
        json.dump(res, open(dest, "w"), indent=1)
