"""
V7 diagnostic D6: is the rho regime decision right? For every development
sequence, 'intervention benefit' = metric(V6-style intervention) -
metric(host native), and the V7c regime statistics (median rho_bar, share
of clean frames). A good regime statistic is 'clean' exactly where the
benefit is negative. (DEVELOPMENT ONLY.)
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from tools.seqstats import combine
from tools.v6.dev import split_sequences

ROOT = "outputs"
rows = []
for split, dets, v6 in (("val7", ["yolov8", "rtdetr"], "X5"), ("val7", ["fasterrcnn"], "V6TF"),
                        ("dev40", ["yolov8", "rtdetr"], "X5")):
    for det in dets:
        for s in split_sequences(split):
            nat = pickle.load(open(f"{ROOT}/v7/{split}/NATIVE/{det}/{s}.pkl", "rb"))
            six = pickle.load(open(f"{ROOT}/v6/{split}/{v6}/{det}/{s}.pkl", "rb"))
            v7 = pickle.load(open(f"{ROOT}/v7/{split}/V7c/{det}/{s}.pkl", "rb"))
            mn, m6, m7 = combine([nat]), combine([six]), combine([v7])
            aud = v7["audit"]
            rb = np.nanmedian([a.get("rho_bar", np.nan) for a in aud])
            cl = np.mean([a.get("regime") == "clean" for a in aud])
            rows.append((split, det, s, m6["HOTA"] - mn["HOTA"], m6["MOTA"] - mn["MOTA"], m6["IDF1"] - mn["IDF1"],
                         m7["HOTA"] - mn["HOTA"], m7["HOTA"] - m6["HOTA"], rb, cl, mn["MOTA"]))
# MOT17: SparseTrack per-sequence V6 vs baseline (external results file)
ext = json.load(open(str(Path(__import__("os").environ.get("ACMOT_EXT", "/Users/ahmedgouda/Desktop/acmot_external")) / "reports/external_results.json")))
pf = json.load(open(str(Path(__import__("os").environ.get("ACMOT_EXT", "/Users/ahmedgouda/Desktop/acmot_external")) / "runs/sparsetrack/MOT17-val/ST7_V7c/per_frame.json")))
for s, b in ext["ST_A_official"]["per_seq"].items():
    v = ext["ST_plus_V6TF"]["per_seq"][s]
    q = [x for x in pf if x["seq"] == s]
    rows.append(("mot17", "yolox", s, v["HOTA"] - b["HOTA"], v["MOTA"] - b["MOTA"], v["IDF1"] - b["IDF1"],
                 np.nan, np.nan, np.nanmedian([x.get("rho_bar", np.nan) for x in q]),
                 np.mean([x.get("regime") == "clean" for x in q]), b["MOTA"]))
R = np.array([r[3:] for r in rows], float)
print(f"{'split':<6}{'det':<11}{'seq':<22}{'dHOTA6':>7}{'dMOTA6':>8}{'dIDF6':>7}{'dHOTA7':>8}{'7-6':>6}{'rho':>6}{'clean':>6}")
for r in rows:
    print(f"{r[0]:<6}{r[1]:<11}{r[2][:21]:<22}{r[3]:7.2f}{r[4]:8.2f}{r[5]:7.2f}{r[6]:8.2f}{r[7]:6.2f}{r[8]:6.2f}{r[9]:6.2f}")
ben = R[:, 0] > 0          # V6 intervention helps (HOTA)
cln = R[:, 6] >= 0.5       # V7c calls it clean (median rho_bar)
print("\nsequences:", len(R), " intervention helps (HOTA):", int(ben.sum()))
print("clean & helps (regime wrong):", int((cln & ben).sum()), " noisy & hurts (regime wrong):", int((~cln & ~ben).sum()))
print("clean & hurts (right):", int((cln & ~ben).sum()), " noisy & helps (right):", int((~cln & ben).sum()))
for thr in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
    c = R[:, 6] >= thr
    print(f"  rho>={thr}: errors {int((c & ben).sum() + (~c & ~ben).sum())}, clean {int(c.sum())}")
