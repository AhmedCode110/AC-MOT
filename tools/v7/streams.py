"""
V7 development diagnostics: labelled candidate streams (DEVELOPMENT ONLY).

Loads the raw candidate stream of every development system together with
its ground truth so that mechanisms can be analysed offline before any
tracker run. Labels are NEVER available to the layer; they are used here
only to diagnose (e.g. which overlapping pairs are duplicates of the same
object and which are two real objects).

Streams (all at the detector adapter's emission floor):
  mot17_yolox   : SparseTrack's published YOLOX-X detections, MOT17 val-half
                  (conf >= 0.01, NMS 0.7), official run cache
  mot17_bt      : BoostTrack's published YOLOX-X detections (conf >= 0.1)
  vd_<det>      : VisDrone val-7 native caches at 736 (yolov8, rtdetr,
                  fasterrcnn), internal class-agnostic GT filter

Per frame: boxes (N,4) xyxy, scores (N,), cls (N,), label (N,) with
1 = true positive (greedy score-ordered matching at IoU >= 0.5 to a target),
0 = false positive, -1 = matches only an ignored/distractor region;
gt_idx (N,) index of the matched GT row (or -1).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXT = Path("/Users/ahmedgouda/Desktop/acmot_external")
MOT_GT = EXT / "BoostTrack/results/gt/MOT17-val"
MOT_DATA = EXT / "data_mirror/MOT17"
VD_VAL = Path("/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val")
MOT_SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
            "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]


def iou(a, b):
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None] - inter, 1e-12)


def label_frame(boxes, scores, gt, ign):
    """Greedy score-ordered matching (detection-AP convention)."""
    n = len(boxes)
    lab = np.zeros(n, np.int8)
    gi = -np.ones(n, np.int64)
    if n == 0:
        return lab, gi
    order = np.argsort(-scores, kind="stable")
    used = np.zeros(len(gt), bool)
    M = iou(boxes, gt) if len(gt) else np.zeros((n, 0))
    MI = iou(boxes, ign) if len(ign) else np.zeros((n, 0))
    for i in order:
        if M.shape[1]:
            c = np.where(used, -1.0, M[i])
            j = int(np.argmax(c))
            if c[j] >= 0.5:
                used[j] = True
                lab[i], gi[i] = 1, j
                continue
        if MI.shape[1] and MI[i].max() >= 0.5:
            lab[i] = -1
    return lab, gi


def _mot17_gt(seq):
    a = np.loadtxt(MOT_GT / seq / "gt/gt.txt", delimiter=",", ndmin=2)
    tgt = a[(a[:, 6] == 1) & (a[:, 7] == 1)]
    ign = a[np.isin(a[:, 7], [2, 7, 8, 12])]
    return tgt, ign


def mot17_stream(kind="yolox"):
    ann = json.load(open(MOT_DATA / "annotations/val_half.json"))
    if kind == "yolox":
        cache = pickle.load(open(EXT / "runs/sparsetrack_A_official/published_detections.pkl", "rb"))
    else:
        cache = pickle.load(open(EXT / "BoostTrack/cache/det_bytetrack_ablation.pkl", "rb"))
    out = {}
    gts = {s: _mot17_gt(s) for s in MOT_SEQS}
    for im in ann["images"]:
        seq, fid = im["file_name"].split("/")[0], int(im["frame_id"])
        if kind == "yolox":
            d = np.asarray(cache[im["file_name"]], np.float64).reshape(-1, 5)
        else:
            p = cache[f"{seq}:{fid}"]
            p = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
            p = np.asarray(p, np.float64).reshape(-1, p.shape[-1] if p.ndim == 2 else 5)
            H, W = im["height"], im["width"]
            scale = min(800 / H, 1440 / W)
            d = p[:, :5].copy()
            d[:, :4] /= scale
        tgt, ign = gts[seq]
        g = tgt[tgt[:, 0] == fid][:, 2:6].copy()
        g[:, 2:] += g[:, :2]
        ig = ign[ign[:, 0] == fid][:, 2:6].copy()
        ig[:, 2:] += ig[:, :2]
        lab, gi = label_frame(d[:, :4], d[:, 4], g, ig)
        out.setdefault(seq, []).append(dict(frame=fid, boxes=d[:, :4], scores=d[:, 4],
                                            cls=np.zeros(len(d), int), label=lab, gt_idx=gi,
                                            gt=g, shape=(im["height"], im["width"])))
    return out


def _vd_gt(seq, data=VD_VAL):
    a = np.loadtxt(data / "annotations" / f"{seq}.txt", delimiter=",", ndmin=2)
    keep = (np.isin(a[:, 7].astype(int), [1, 4, 5, 6, 9]) & (a[:, 6] == 1)
            & (a[:, 8] < 2) & (a[:, 9] < 2))
    return a[keep]


def visdrone_stream(det, split_dir="outputs/det_cache_val_native", data=VD_VAL, res=736):
    out = {}
    for f in sorted((ROOT / split_dir / det).glob("*.npz")):
        seq = f.stem
        z = np.load(f)
        a = z[f"det_{res}"]
        n = int(z["frames"])
        gt = _vd_gt(seq, data)
        frames = []
        for t in range(1, n + 1):
            r = a[a[:, 0] == t]
            g = gt[gt[:, 0] == t][:, 2:6].copy()
            g[:, 2:] += g[:, :2]
            lab, gi = label_frame(r[:, 1:5], r[:, 5], g, np.zeros((0, 4)))
            frames.append(dict(frame=t, boxes=r[:, 1:5].astype(np.float64), scores=r[:, 5].astype(np.float64),
                               cls=r[:, 6].astype(int), label=lab, gt_idx=gi, gt=g,
                               shape=tuple(int(v) for v in z["shape"])))
        out[seq] = frames
    return out


def load(name, cache_dir=ROOT / "outputs/v7/diag"):
    p = Path(cache_dir) / f"stream_{name}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    if name == "mot17_yolox":
        s = mot17_stream("yolox")
    elif name == "mot17_bt":
        s = mot17_stream("bt")
    elif name.startswith("vd_"):
        s = visdrone_stream(name[3:])
    else:
        raise KeyError(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(s, open(p, "wb"))
    return s


STREAMS = ["mot17_yolox", "mot17_bt", "vd_yolov8", "vd_rtdetr", "vd_fasterrcnn"]

if __name__ == "__main__":
    import sys
    for n in (sys.argv[1:] or STREAMS):
        s = load(n)
        nf = sum(len(v) for v in s.values())
        nd = sum(len(f["scores"]) for v in s.values() for f in v)
        tp = sum(int((f["label"] == 1).sum()) for v in s.values() for f in v)
        ng = sum(len(f["gt"]) for v in s.values() for f in v)
        print(f"{n:<14} seqs {len(s):3d} frames {nf:6d} dets {nd:8d} ({nd / nf:6.1f}/fr) TP {tp:7d} GT {ng:7d}")
