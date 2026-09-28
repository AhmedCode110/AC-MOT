"""
V7 diagnostic D5: does a label-free proxy predict the precision of the
ambiguous band [t1, t2)? (DEVELOPMENT ONLY; labels only for the check.)

Per sequence:
  prec_amb   true precision of candidates in [t1, t2)   (label-based)
  prec_conf  true precision of candidates >= t2         (label-based)
  rho        confident share of the foreground          (label-free)
  sup_amb    share of [t1,t2) candidates that correspond (IoU >= 0.5) to a
             track of frame t-1                          (label-free)
  sup_conf   same for confident candidates               (label-free)
Track context: MOT17 = SparseTrack baseline output, VisDrone = V6 (X5) run.
"""
from __future__ import annotations

import sys
from collections import deque

import numpy as np

from online_calibration import dedup, nested_otsu
from tools.v7.diag_dup_rules import mot17_tracks, vd_tracks
from tools.v7.streams import iou, load


def lg(s):
    s = np.clip(s, 1e-9, 1 - 1e-9)
    return np.log(s / (1 - s))


def analyse(name, floor=0.0, vd_system="X5"):
    st = load(name)
    print(f"\n=== {name} floor {floor}")
    rows = []
    for seq, frames in st.items():
        trk = mot17_tracks(seq) if name.startswith("mot17") else vd_tracks(name[3:], seq, vd_system)
        W = deque(maxlen=10)
        acc = dict(tp_a=0, n_a=0, tp_c=0, n_c=0, sup_a=0, sup_c=0, rho=[])
        for f in frames:
            s, b, lab = f["scores"], f["boxes"], f["label"]
            m = s >= floor
            s, b, lab = s[m], b[m], lab[m]
            if len(s) > 1:
                k = sorted(dedup(b, s, 0.5))
                s, b, lab = s[k], b[k], lab[k]
            L = lg(s)
            if W:
                H = np.concatenate(W)
                th = nested_otsu(H)
                if th is not None and th[1] > th[0]:
                    t1, t2, _ = th
                    acc["rho"].append((H >= t2).sum() / max((H >= t1).sum(), 1))
                    prev = trk.get(f["frame"] - 1, (np.zeros((0, 4)), None))[0]
                    sup = (iou(b, prev).max(1) >= 0.5) if len(prev) and len(b) else np.zeros(len(b), bool)
                    a = (L >= t1) & (L < t2) & (lab >= 0)
                    c = (L >= t2) & (lab >= 0)
                    acc["tp_a"] += int((lab[a] == 1).sum()); acc["n_a"] += int(a.sum())
                    acc["tp_c"] += int((lab[c] == 1).sum()); acc["n_c"] += int(c.sum())
                    acc["sup_a"] += int(sup[a].sum()); acc["sup_c"] += int(sup[c].sum())
            W.append(L)
        r = (seq, acc["tp_a"] / max(acc["n_a"], 1), acc["tp_c"] / max(acc["n_c"], 1), float(np.median(acc["rho"])),
             acc["sup_a"] / max(acc["n_a"], 1), acc["sup_c"] / max(acc["n_c"], 1), acc["n_a"] / max(acc["n_c"], 1))
        rows.append(r)
        print(f"  {seq[:22]:<22} prec_amb {r[1]:.2f} prec_conf {r[2]:.2f} | rho {r[3]:.2f} sup_amb {r[4]:.2f} "
              f"sup_conf {r[5]:.2f} n_amb/n_conf {r[6]:.2f}")
    return rows


if __name__ == "__main__":
    allrows = []
    for n in (sys.argv[1:] or ["mot17_yolox", "vd_yolov8", "vd_rtdetr", "vd_fasterrcnn"]):
        allrows += analyse(n)
        if n == "mot17_yolox":
            allrows += analyse(n, floor=0.1)
    R = np.array([r[1:] for r in allrows])
    for j, nm in ((2, "rho"), (3, "sup_amb"), (5, "n_amb/n_conf")):
        print(f"Spearman(prec_amb, {nm}) = {np.corrcoef(np.argsort(np.argsort(R[:, 0])), np.argsort(np.argsort(R[:, j])))[0, 1]:.2f}")
