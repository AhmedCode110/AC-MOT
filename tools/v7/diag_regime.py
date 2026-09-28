"""
V7 diagnostic D4: label-free regime statistics per stream and sequence
(DEVELOPMENT ONLY; no labels are read here).

For each frame (causal, pooled logits of the previous 10 frames after the
V6 duplicate rule), nested Otsu gives t1 (background|foreground) and t2
(ambiguous|confident). Statistics:
  rho   = |{l >= t2}| / |{l >= t1}|   confident share of the foreground
  sep   = (median confident - t2) / (t2 - t1)  (how far the confident mode
          sits above its own boundary, in units of the ambiguous band)
  pos   = (logit(native) - t1) / (t2 - t1)     host's point inside the band
Printed: medians over frames, per sequence, for emission floors 0.01 / 0.1.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

from online_calibration import dedup, nested_otsu
from tools.v7.streams import ROOT, load


def lg(s):
    s = np.clip(s, 1e-9, 1 - 1e-9)
    return np.log(s / (1 - s))


def seq_stats(frames, native, floor=0.0, window=10):
    W = deque(maxlen=window)
    out = []
    for f in frames:
        s, b = f["scores"], f["boxes"]
        m = s >= floor
        s, b = s[m], b[m]
        if len(s) > 1:
            k = sorted(dedup(b, s, 0.5))
            s = s[k]
        if W:
            H = np.concatenate(W)
            th = nested_otsu(H)
            if th is not None and th[1] > th[0]:
                t1, t2, _ = th
                conf = H[H >= t2]
                fg = H[H >= t1]
                out.append((len(conf) / max(len(fg), 1),
                            (np.median(conf) - t2) / (t2 - t1),
                            (lg(native) - t1) / (t2 - t1)))
        W.append(lg(s))
    return np.array(out) if out else np.zeros((0, 3))


def raw_visdrone(det, split_dir, seqs, res=736):
    out = {}
    for seq in seqs:
        z = np.load(ROOT / split_dir / det / f"{seq}.npz")
        a = z[f"det_{res}"]
        out[seq] = [dict(scores=a[a[:, 0] == t][:, 5].astype(float), boxes=a[a[:, 0] == t][:, 1:5].astype(float))
                    for t in range(1, int(z["frames"]) + 1)]
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    if which == "val":
        streams = {n: (load(n), 0.6 if n.startswith("mot17") else 0.25)
                   for n in ["mot17_yolox", "vd_yolov8", "vd_rtdetr", "vd_fasterrcnn"]}
    else:
        dev = json.load(open(ROOT / "research/TRAIN_SPLIT_V5.json"))["development"]
        streams = {f"dev40_{d}": (raw_visdrone(d, "outputs/det_cache_train_native", dev), 0.25)
                   for d in ["yolov8", "rtdetr"]}
    for n, (st, native) in streams.items():
        print(f"\n=== {n} (native assoc {native})")
        allv = {0.0: [], 0.1: []}
        for seq, frames in st.items():
            row = []
            for fl in (0.0, 0.1):
                x = seq_stats(frames, native, fl)
                allv[fl].append(x)
                row.append(f"floor {fl:.2f}: rho {np.median(x[:, 0]):.2f} [{np.percentile(x[:, 0], 10):.2f},{np.percentile(x[:, 0], 90):.2f}]"
                           f" sep {np.median(x[:, 1]):5.2f} pos {np.median(x[:, 2]):5.2f}")
            print(f"  {seq[:22]:<22} " + " | ".join(row))
        for fl in (0.0, 0.1):
            x = np.concatenate(allv[fl])
            print(f"  ALL floor {fl:.2f}: rho median {np.median(x[:, 0]):.2f}  frac(rho>=0.5) {np.mean(x[:, 0] >= 0.5):.2f}"
                  f"  sep {np.median(x[:, 1]):.2f}  pos {np.median(x[:, 2]):.2f}")
