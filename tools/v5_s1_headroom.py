"""
V5 stage S1 — adaptation-value (headroom) analysis (VisDrone val only;
GT used offline only).

For each control target, run whole sequences with each fixed value (all
other parameters at V4 values, resolution 736), for YOLOv8n and RT-DETR-L.
Store per-frame TP / FP / FN / IDSW and the logged scene/tracking state.

Headroom of target k (per detector, pooled over sequences):
  oracle  = sum over 30-frame windows of min_v errors_w(v)
  global  = errors of the single best value for BOTH detectors together
  headroom = (global - oracle) / #GT      [MOTA points, an upper bound]
Also reported: headroom of the best value per detector (to separate
"adaptation over time/scenes" from "detector-specific value").
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATASET = "/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val"
OUT = Path(os.environ.get("V5_S1_OUT", "outputs/v5/s1"))
DETS = ["yolov8", "rtdetr"]
TARGETS = {
    "resolution": ("fixed_resolution", [640, 736, 832]),
    "sensitivity": ("fixed_sensitivity", [0.3, 0.4, 0.5, 0.6]),
    "gate_tau": ("gate_tau", [0.5, 0.75, 1.0, 1.25, 1.5]),
    "assoc_offset": ("assoc_offset", [0.0, 0.05, 0.10, 0.18, 0.26]),
    "retention": ("tracker_buffer", [15, 30, 45, 60, 90]),
}
WINDOW = 30


def base_overrides():
    o = json.load(open("configs/universal_acmot_policy_v4.json"))["overrides"]
    o["scene_state"] = True
    return o


def per_frame_events(seq, tracks_file, n_frames):
    import motmetrics as mm

    from tools.eval_local import load_gt, load_tracks
    from tools.seqstats import apply_ignore_regions
    gt = load_gt(Path(DATASET) / "annotations" / f"{seq}.txt")
    tr = apply_ignore_regions(DATASET, seq, load_tracks(tracks_file))
    acc = mm.MOTAccumulator(auto_id=False)
    for t in range(1, n_frames + 1):
        g, p = gt[gt[:, 0] == t], (tr[tr[:, 0] == t] if len(tr) else tr)
        acc.update(g[:, 1].astype(int).tolist(), p[:, 1].astype(int).tolist(),
                   mm.distances.iou_matrix(g[:, 2:6], p[:, 2:6], max_iou=0.5),
                   frameid=t)
    ev = acc.mot_events.reset_index()
    ev = ev[ev.Type != "RAW"]
    out = np.zeros((n_frames + 1, 5), dtype=np.int64)  # gt,tp,fp,fn,idsw
    for f, grp in ev.groupby("FrameId"):
        ty = grp.Type.values
        tp = np.isin(ty, ["MATCH", "SWITCH", "TRANSFER", "ASCEND",
                          "MIGRATE"]).sum()
        out[f] = [tp + (ty == "MISS").sum(), tp, (ty == "FP").sum(),
                  (ty == "MISS").sum(), (ty == "SWITCH").sum()]
    out = out[1:]
    idc = per_frame_identity(gt, tr, n_frames)
    return np.c_[out, idc]


def per_frame_identity(gt, tr, n_frames):
    """Per-frame IDTP / IDFP / IDFN under the global optimal one-to-one
    trajectory mapping maximising IDTP (IoU >= 0.5), i.e. the IDF1
    definition used by motmetrics."""
    from scipy.optimize import linear_sum_assignment

    from tools.eval_local import iou_xywh
    gids = np.unique(gt[:, 1]).astype(int)
    hids = np.unique(tr[:, 1]).astype(int) if len(tr) else np.array([], int)
    gi = {g: i for i, g in enumerate(gids)}
    hi = {h: i for i, h in enumerate(hids)}
    co = np.zeros((len(gids), len(hids)))
    pairs = []
    for t in range(1, n_frames + 1):
        g, p = gt[gt[:, 0] == t], (tr[tr[:, 0] == t] if len(tr) else tr)
        if len(g) and len(p):
            ok = iou_xywh(g[:, 2:6], p[:, 2:6]) >= 0.5
            a, b = np.nonzero(ok)
            for x, y in zip(a, b):
                co[gi[int(g[x, 1])], hi[int(p[y, 1])]] += 1
                pairs.append((t, gi[int(g[x, 1])], hi[int(p[y, 1])]))
    mapped = set()
    if co.size:
        r, c = linear_sum_assignment(-co)
        mapped = {(i, j) for i, j in zip(r, c) if co[i, j] > 0}
    idtp = np.zeros(n_frames + 1, dtype=np.int64)
    for t, i, j in pairs:
        if (i, j) in mapped:
            idtp[t] += 1
    ngt = np.bincount(gt[:, 0].astype(int), minlength=n_frames + 1)[:n_frames + 1]
    nh = (np.bincount(tr[:, 0].astype(int), minlength=n_frames + 1)[:n_frames + 1]
          if len(tr) else np.zeros(n_frames + 1, dtype=np.int64))
    return np.c_[idtp, nh - idtp, ngt - idtp][1:]


def run_one(job):
    target, value, det, seq = job
    dest = OUT / target / str(value) / det / f"{seq}.pkl"
    if dest.exists():
        return
    from run_universal_acmot import build_config
    from tools.run_policy_validation import CachedDetector, make_tracker
    from tools.seqstats import sequence_stats
    from universal_policy_pipeline import (POLICIES, UniversalPolicyPipeline,
                                           replace)
    field, _ = TARGETS[target]
    pol = replace(POLICIES["V1"], **dict(base_overrides(), **{field: value}))
    cd = CachedDetector(f"outputs/det_cache/{det}/{seq}.npz")
    cfg = build_config()
    pipe = UniversalPolicyPipeline(cfg, cd, make_tracker(cfg, pol), pol)
    img = np.empty(cd.shape + (0,), np.uint8)
    lines, states = [], []
    for i in range(1, cd.frames + 1):
        cd.frame = i
        r = pipe.process(i, img, cd.visual_dict(i))
        states.append({k[2:]: v for k, v in r["audit"].items()
                       if k.startswith("s_")})
        for t in r["tracks"]:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},"
                         f"{t.x2 - t.x1:.3f},{t.y2 - t.y1:.3f},"
                         f"{t.confidence:.6f},{t.class_id},-1,-1\n")
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.writelines(lines)
    f.close()
    st = sequence_stats(DATASET, seq, f.name)
    st["frame_events"] = per_frame_events(seq, f.name, cd.frames)
    st["states"] = states
    os.unlink(f.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(dest, "wb"))


def seqs():
    return sorted(p.name for p in Path(DATASET, "sequences").iterdir()
                  if p.is_dir())


def load(target, value, det, seq):
    return pickle.load(open(OUT / target / str(value) / det / f"{seq}.pkl",
                            "rb"))


def headroom():
    rows = []
    for target, (_, values) in TARGETS.items():
        E = {}   # (det, seq) -> array [n_values, n_windows]
        G = {}
        for d, s in itertools.product(DETS, seqs()):
            ev = [load(target, v, d, s)["frame_events"] for v in values]
            n = len(ev[0])
            nw = int(np.ceil(n / WINDOW))
            err = np.array([[e[w * WINDOW:(w + 1) * WINDOW, 2:5].sum()
                             for w in range(nw)] for e in ev])
            E[(d, s)] = err
            G[(d, s)] = ev[0][:, 0].sum()
        tot = {d: np.sum([E[(d, s)].sum(1) for s in seqs()], 0) for d in DETS}
        gt = {d: sum(G[(d, s)] for s in seqs()) for d in DETS}
        shared = int(np.argmin(sum(tot[d] / gt[d] for d in DETS)))
        rec = dict(target=target, values=values,
                   shared_best=values[shared])
        for d in DETS:
            own = int(np.argmin(tot[d]))
            oracle = sum(E[(d, s)].min(0).sum() for s in seqs())
            seq_oracle = sum(E[(d, s)].sum(1).min() for s in seqs())
            rec[d] = dict(
                own_best=values[own],
                gain_detector_specific=100 * (tot[d][shared] - tot[d][own])
                / gt[d],
                gain_per_sequence=100 * (tot[d][shared] - seq_oracle) / gt[d],
                gain_per_window=100 * (tot[d][shared] - oracle) / gt[d])
        rows.append(rec)
        print(f"{target:<13} shared best {rec['shared_best']}")
        for d in DETS:
            r = rec[d]
            print(f"   {d:<7} own best {r['own_best']:<5} | headroom (MOTA pts):"
                  f" detector-specific {r['gain_detector_specific']:5.2f}  "
                  f"per-sequence {r['gain_per_sequence']:5.2f}  "
                  f"per-window {r['gain_per_window']:5.2f}")
    json.dump(rows, open(OUT / "headroom.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        headroom()
    else:
        jobs = [(t, v, d, s) for t, (_, vals) in TARGETS.items()
                for v in vals for d in DETS for s in seqs()]
        with ProcessPoolExecutor(5) as ex:
            list(ex.map(run_one, jobs, chunksize=1))
        headroom()
