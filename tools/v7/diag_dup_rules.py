"""
V7 diagnostic D3: offline comparison of duplicate rules on labelled streams
(DEVELOPMENT ONLY). Each rule runs as a greedy pass in descending score on
the candidates of a frame (above a relevance floor) and removes candidates;
we count removed true positives (lost real objects) and removed false
positives (duplicates / clutter) among candidates with score >= floor.

Rules:
  none      nothing removed
  iou       V6: remove l if IoU(l, kept h) > 0.5
  xclass    remove l only if a kept h of a DIFFERENT class has IoU > 0.5
  track     remove l if IoU(l, kept h) > 0.5 unless l corresponds (IoU >= 0.5)
            to a previous-frame track different from h's best track
  xtrack    xclass OR track (cross-class pairs always, same-class via track)

Track context = the host's output tracks of frame t-1 (MOT17: SparseTrack
baseline output; VisDrone: a V6 run) -- a proxy for the layer's own tracks.
"""
from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path

import numpy as np

from tools.v7.streams import EXT, ROOT, iou, load


def mot17_tracks(seq, name="ST_replay_baseline"):
    a = np.loadtxt(EXT / f"runs/sparsetrack/MOT17-val/{name}/data/{seq}.txt", delimiter=",", ndmin=2)
    return _by_frame(a)


def vd_tracks(det, seq, system="X5"):
    st = pickle.load(open(ROOT / f"outputs/v6/val7/{system}/{det}/{seq}.pkl", "rb"))
    txt = st["tracks_txt"]
    a = np.loadtxt(io.StringIO(txt), delimiter=",", ndmin=2) if txt else np.zeros((0, 10))
    return _by_frame(a)


def _by_frame(a):
    d = {}
    for fr in np.unique(a[:, 0]).astype(int):
        r = a[a[:, 0] == fr]
        b = r[:, 2:6].copy()
        b[:, 2:] += b[:, :2]
        d[fr] = (b, r[:, 1].astype(int))
    return d


def apply_rule(rule, b, s, c, prev):
    order = np.argsort(-s, kind="stable")
    M = iou(b, b)
    P = iou(b, prev) if prev is not None and len(prev) else np.zeros((len(b), 0))
    best = P.argmax(1) if P.shape[1] else -np.ones(len(b), int)
    bval = P.max(1) if P.shape[1] else np.zeros(len(b))
    kept, removed = [], []
    for l in order:
        hs = [h for h in kept if M[h, l] > 0.5]
        rm = False
        for h in hs:
            if rule == "iou":
                rm = True
            elif rule == "xclass":
                rm = c[h] != c[l]
            elif rule in ("track", "xtrack"):
                if rule == "xtrack" and c[h] != c[l]:
                    rm = True
                else:
                    indep = bval[l] >= 0.5 and best[l] != best[h]
                    rm = not indep
            if rm:
                break
        (removed if rm else kept).append(l)
    return np.array(removed, int)


def evaluate(name, floors=(0.1, 0.3, 0.5), vd_system="X5"):
    st = load(name)
    rules = ["iou", "xclass", "track", "xtrack"]
    res = {r: {f: [0, 0] for f in floors} for r in rules}
    tot = {f: [0, 0] for f in floors}
    for seq, frames in st.items():
        if name.startswith("mot17"):
            trk = mot17_tracks(seq)
        else:
            trk = vd_tracks(name[3:], seq, vd_system)
        for f in frames:
            b, s, c, lab = f["boxes"], f["scores"], f["cls"], f["label"]
            prev = trk.get(f["frame"] - 1, (np.zeros((0, 4)), None))[0]
            for fl in floors:
                m = s >= fl
                tot[fl][0] += int((lab[m] == 1).sum())
                tot[fl][1] += int((lab[m] == 0).sum())
            m0 = s >= min(floors)
            idx = np.where(m0)[0]
            if len(idx) < 2:
                continue
            for r in rules:
                rem = idx[apply_rule(r, b[idx], s[idx], c[idx], prev)] if len(idx) else []
                for fl in floors:
                    rr = [i for i in rem if s[i] >= fl]
                    res[r][fl][0] += int((lab[rr] == 1).sum())
                    res[r][fl][1] += int((lab[rr] == 0).sum())
    print(f"\n=== {name}  (track context: {'SparseTrack baseline' if name.startswith('mot17') else vd_system})")
    for fl in floors:
        print(f"  candidates >= {fl}: TP {tot[fl][0]}  FP {tot[fl][1]}")
        for r in rules:
            tp, fp = res[r][fl]
            print(f"    {r:<7} removes TP {tp:6d} ({100 * tp / max(tot[fl][0], 1):5.2f}%)  FP {fp:7d} ({100 * fp / max(tot[fl][1], 1):5.2f}%)")


if __name__ == "__main__":
    args = sys.argv[1:] or ["mot17_yolox", "vd_yolov8", "vd_rtdetr", "vd_fasterrcnn"]
    for n in args:
        if n.startswith("vd_"):
            evaluate(n, vd_system="X5")
            evaluate(n, vd_system="X5@dedup_iou=0")
        else:
            evaluate(n)
