"""
Generic TrackEval (pinned 12c8791) evaluation for the recent external systems:
per-sequence and pooled HOTA / DetA / AssA / MOTA / IDF1 / IDS / FP / FN /
Precision / Recall, the frozen project bootstrap (10,000 paired sequence
resamples, seed 42, percentile 95% CI of pooled B - A) and sequence
wins / ties / losses on HOTA (tie: |dHOTA| < 0.01).

A benchmark is described by a small JSON file (kind, gt_folder, seqmap, ...):
  kind "mot"   : MotChallenge2DBox, GT <gt_folder>/<seq>/gt/gt.txt + seqinfo.ini,
                 trackers <trackers_root>/<name>/data/<seq>.txt, class pedestrian
  kind "kitti" : Kitti2DBox, GT <gt_folder>/label_02 + <gt_folder>/<seqmap>,
                 trackers <trackers_root>/<name>/data/<seq>.txt, classes car, pedestrian

  python te_eval.py table <bench.json> <trackers_root> <name> [...]
  python te_eval.py boot  <bench.json> <trackers_root> <A> <B> [--out f.json]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

for _n, _v in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _n):
        setattr(np, _n, _v)
TE = os.environ.get("ACMOT_TRACKEVAL", str(Path.home() / "acmot_work/CUE_SELECTION/cue_ablation_tools/TrackEval"))
sys.path.insert(0, TE)
import trackeval  # noqa: E402

N_BOOT, SEED, TIE = 10000, 42, 0.01
with redirect_stdout(io.StringIO()):
    _H, _C, _I = trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()


def _seqs(bench):
    if bench["kind"] == "kitti":
        return [l.split()[0] for l in open(Path(bench["gt_folder"]) / bench["seqmap"]) if l.strip()]
    lines = [l.strip() for l in open(bench["seqmap"]) if l.strip()]
    return [l for l in lines if l != "name"]


def classes(bench):
    return ["car", "pedestrian"] if bench["kind"] == "kitti" else ["pedestrian"]


def evaluate(bench, trackers_root, name):
    """{seq: {class: {HOTA, CLEAR, Identity}}} for one tracker."""
    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update(dict(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False,
                   OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False,
                   TIME_PROGRESS=False, DISPLAY_LESS_PROGRESS=True))
    if bench["kind"] == "kitti":
        dc = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
        dc.update(dict(GT_FOLDER=bench["gt_folder"], TRACKERS_FOLDER=str(trackers_root),
                       TRACKERS_TO_EVAL=[name], SPLIT_TO_EVAL="training", TRACKER_SUB_FOLDER="data",
                       PRINT_CONFIG=False, CLASSES_TO_EVAL=["car", "pedestrian"]))
        ds = trackeval.datasets.Kitti2DBox(dc)
        ds.gt_fol = bench["gt_folder"]
        ds.seq_list, ds.seq_lengths = _kitti_seqmap(bench)
    else:
        dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dc.update(dict(GT_FOLDER=bench["gt_folder"], TRACKERS_FOLDER=str(trackers_root),
                       BENCHMARK=bench.get("benchmark", "MOT17"), SPLIT_TO_EVAL=bench.get("split", "val"),
                       TRACKERS_TO_EVAL=[name], TRACKER_SUB_FOLDER="data", SKIP_SPLIT_FOL=True,
                       PRINT_CONFIG=False, SEQMAP_FILE=bench["seqmap"]))
        ds = trackeval.datasets.MotChallenge2DBox(dc)
    with redirect_stdout(io.StringIO()):
        res, _ = trackeval.Evaluator(ec).evaluate([ds], [_H, _C, _I])
    r = res[type(ds).__name__][name]
    return {s: {c: r[s][c] for c in classes(bench)} for s in _seqs(bench)}


def _kitti_seqmap(bench):
    seqs, lengths = [], {}
    for l in open(Path(bench["gt_folder"]) / bench["seqmap"]):
        p = l.split()
        if p:
            seqs.append(p[0])
            lengths[p[0]] = int(p[3])
    return seqs, lengths


def combine(per_list):
    H, C, I = _H, _C, _I
    d = {str(i): p for i, p in enumerate(per_list)}
    with redirect_stdout(io.StringIO()):
        h = H.combine_sequences({k: v["HOTA"] for k, v in d.items()})
        c = C.combine_sequences({k: v["CLEAR"] for k, v in d.items()})
        i = I.combine_sequences({k: v["Identity"] for k, v in d.items()})
    return dict(HOTA=100 * float(np.mean(h["HOTA"])), DetA=100 * float(np.mean(h["DetA"])),
                AssA=100 * float(np.mean(h["AssA"])), MOTA=100 * float(c["MOTA"]),
                IDF1=100 * float(i["IDF1"]), IDS=int(c["IDSW"]), FP=int(c["CLR_FP"]),
                FN=int(c["CLR_FN"]), Precision=100 * float(c["CLR_Pr"]),
                Recall=100 * float(c["CLR_Re"]), Frag=int(c["Frag"]))


def table(bench, trackers_root, names):
    out = {}
    for n in names:
        per = evaluate(bench, trackers_root, n)
        out[n] = {c: dict(pooled=combine([per[s][c] for s in per]),
                          per_seq={s: combine([per[s][c]]) for s in per}) for c in classes(bench)}
    return out


def bootstrap(bench, trackers_root, a, b):
    pa, pb = evaluate(bench, trackers_root, a), evaluate(bench, trackers_root, b)
    seqs = list(pa)
    out = {}
    for c in classes(bench):
        A, B = [pa[s][c] for s in seqs], [pb[s][c] for s in seqs]
        ma, mb = combine(A), combine(B)
        rng = np.random.default_rng(SEED)
        keys = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDS", "FP", "FN"]
        diffs = {k: np.empty(N_BOOT) for k in keys}
        for i in range(N_BOOT):
            idx = rng.integers(0, len(seqs), len(seqs))
            ra, rb = combine([A[j] for j in idx]), combine([B[j] for j in idx])
            for k in keys:
                diffs[k][i] = rb[k] - ra[k]
        r = {}
        for k in keys:
            lo, hi = np.percentile(diffs[k], [2.5, 97.5])
            r[k] = dict(diff=float(mb[k] - ma[k]), ci_lo=float(lo), ci_hi=float(hi))
        dh = [combine([pb[s][c]])["HOTA"] - combine([pa[s][c]])["HOTA"] for s in seqs]
        r["seq_wins_ties_losses"] = [sum(d >= TIE for d in dh), sum(abs(d) < TIE for d in dh),
                                     sum(d <= -TIE for d in dh)]
        r["per_seq_dHOTA"] = dict(zip(seqs, map(float, dh)))
        r["baseline"], r["v7"] = ma, mb
        out[c] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "boot"])
    ap.add_argument("bench")
    ap.add_argument("root")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    bench = json.load(open(a.bench))
    res = table(bench, a.root, a.names) if a.cmd == "table" else bootstrap(bench, a.root, *a.names[:2])
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    if a.cmd == "table":
        for n, v in res.items():
            for c, t in v.items():
                p = t["pooled"]
                print(f"{n:28s} {c:10s} HOTA {p['HOTA']:.3f} DetA {p['DetA']:.3f} AssA {p['AssA']:.3f} "
                      f"MOTA {p['MOTA']:.3f} IDF1 {p['IDF1']:.3f} IDS {p['IDS']} FP {p['FP']} FN {p['FN']}")
    else:
        for c, r in res.items():
            print(c, "W/T/L", r["seq_wins_ties_losses"])
            for k in ("HOTA", "MOTA", "IDF1"):
                print(f"  d{k} {r[k]['diff']:+.3f} [{r[k]['ci_lo']:+.3f}, {r[k]['ci_hi']:+.3f}]")


if __name__ == "__main__":
    main()
