"""
POST-FREEZE recent external system: TrackTrack (Shim, Ko, Yang, Kim, "Focusing
on Tracks for Online Multi-Object Tracking", CVPR 2025;
github.com/kamkyu94/TrackTrack @ ee7f1c5).

Reproduces `3. Tracker/run.py` (track() and the post-processing of run():
AFLink for DanceTrack, Gaussian interpolation for MOT17) on the detection +
feature pickles (NMS 0.80 and NMS 0.95 views) with the official per-sequence
parameters of utils/etc.py::set_parameters and seed 10000.

V7f arm (V7_RECENT_EXTERNAL_PROTOCOL.md, amendments 1-2): per frame the NMS-0.80
candidates go through the frozen layer (no image cue: the tracker reads no
image); kept candidates carry the layer's scores; in the NMS-0.95 view a row
matching a candidate at IoU >= 0.97 (the tracker's own rule) follows that
candidate (removed or rescored), other rows are unchanged; det_thr <- assoc,
init_thr <- birth. Contract: assoc det_thr, birth init_thr, low 0.1 (detector
conf), match match_thr (not mapped).

Compatibility: CPU (`.cuda()` no-op, torch.load to CPU), NumPy-2 alias np.float_.

  python tracktrack_v7.py --dataset MOT17|DanceTrack --system BASELINE|V7f --name N \
      --pickles <dir with {mot17|dance}_val_0.80.pickle and _0.95.pickle (with features)> --data <dataset split dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

if not hasattr(np, "float_"):
    np.float_ = np.float64
ACMOT = Path(os.environ.get("ACMOT_ROOT", Path(__file__).resolve().parents[3]))
EXT = Path(os.environ.get("ACMOT_EXT", Path.home() / "acmot_work/acmot_external"))
TT = Path(os.environ.get("TRACKTRACK_ROOT", EXT / "TrackTrack")) / "3. Tracker"


def cpu_shims():
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    load = torch.load

    def cpu_load(*a, **k):
        k.setdefault("map_location", "cpu")
        k.setdefault("weights_only", False)
        return load(*a, **k)
    torch.load = cpu_load


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["MOT17", "DanceTrack"])
    ap.add_argument("--system", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pickles", required=True)
    ap.add_argument("--data", required=True, help="split folder with <seq>/seqinfo.ini")
    ap.add_argument("--root", default=str(EXT / "runs/tracktrack"))
    a = ap.parse_args()
    baseline = a.system == "BASELINE"
    pick, data = Path(a.pickles).resolve(), str(Path(a.data).resolve()) + "/"
    out = Path(a.root).resolve() / f"{a.dataset}-val" / a.name
    (out / "data").mkdir(parents=True, exist_ok=True)
    (Path(str(out) + "_post") / "data").mkdir(parents=True, exist_ok=True)
    cpu_shims()
    os.chdir(TT)
    sys.path.insert(0, str(TT))
    import pickle
    import run as ttrun
    from utils.etc import set_parameters, write_results
    from trackers.tracker import Tracker
    from AFLink.AppFreeLink import AFLink
    from AFLink.model import PostLinker
    from AFLink.dataset import LinkData
    from utils.gbi import gb_interpolation
    args = ttrun.make_parser().parse_args(["--dataset", a.dataset, "--mode", "val"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    stem = "mot17_val" if a.dataset == "MOT17" else "dance_val"
    detections = pickle.load(open(pick / f"{stem}_0.80.pickle", "rb"))
    detections_95 = pickle.load(open(pick / f"{stem}_0.95.pickle", "rb"))
    if not baseline:
        sys.path.insert(0, str(ACMOT))
        from acmot_v7 import HostContract, V7Layer, spec_from_dict
        cfg = json.loads((ACMOT / "configs/universal_acmot_policy_v7.json").read_text())
        assert a.system == cfg["system"], "post-freeze: only the frozen policy may run"
        spec = spec_from_dict(cfg["spec"])
    audit = []
    for vid_name in detections.keys():
        # --- official track() ---
        set_parameters(args, vid_name, "val")
        for s_i in open(data + vid_name + "/seqinfo.ini").readlines():
            if "frameRate" in s_i:
                args.max_time_lost = int(s_i.split("=")[-1]) * 2
            if "imWidth" in s_i:
                args.img_w = int(s_i.split("=")[-1])
            if "imHeight" in s_i:
                args.img_h = int(s_i.split("=")[-1])
        tracker = Tracker(args, vid_name)
        layer = None
        if not baseline:
            host = HostContract(assoc=args.det_thr, birth=args.init_thr, low=0.1, match=args.match_thr)
            host_det, host_init = args.det_thr, args.init_thr
            layer = V7Layer(spec, host)
        results = []
        for frame_id in detections[vid_name].keys():
            d, d95 = detections[vid_name][frame_id], detections_95[vid_name][frame_id]
            if layer is not None:
                args.det_thr, args.init_thr = host_det, host_init
                if d is not None:
                    dec = layer.step(d[:, :4].astype(np.float64), d[:, 4].astype(np.float64), None,
                                     classes=np.zeros(len(d), int))
                    keep = np.asarray(dec.keep, int)
                    args.det_thr, args.init_thr = float(dec.assoc), float(dec.birth)
                    audit.append(dict(seq=vid_name, frame=int(frame_id),
                                      **{k: v for k, v in dec.log.items() if isinstance(v, (int, float, str))}))
                    new_score = np.full(len(d), np.nan)
                    new_score[keep] = np.asarray(dec.scores, np.float64)
                    if d95 is not None:
                        m = iou(d95[:, :4].astype(np.float64), d[:, :4].astype(np.float64))
                        j = m.argmax(1) if m.size else np.zeros(len(d95), int)
                        hit = (m.max(1) >= 0.97) if m.size else np.zeros(len(d95), bool)
                        rows = []
                        for r in range(len(d95)):
                            if not hit[r]:
                                rows.append(d95[r])
                            elif not np.isnan(new_score[j[r]]):
                                x = d95[r].copy(); x[4] = new_score[j[r]]; rows.append(x)
                        d95 = np.asarray(rows, d95.dtype).reshape(-1, d95.shape[1]) if rows else None
                    if len(keep):
                        d = d[keep].copy()
                        d[:, 4] = np.asarray(dec.scores, d.dtype)
                    else:
                        d = None
            if d is not None and d95 is not None:
                track_results = tracker.update(d, d95)
            elif d is not None:
                track_results = tracker.update(d, d[:0])
            else:
                track_results = tracker.update_without_detections()
            if layer is not None:
                ids = [t.track_id for t in track_results if t.track_id > 0]
                boxes = [t.x1y1x2y2 for t in track_results if t.track_id > 0]
                layer.observe(np.asarray(boxes, np.float64).reshape(-1, 4), ids)
            x1y1whs, track_ids, scores = [], [], []
            for t in track_results:
                if a.dataset == "MOT17" and t.x1y1wh[2] / t.x1y1wh[3] > 1.6:   # official: "MOT" in data_path
                    continue
                if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                    x1y1whs.append(t.x1y1wh)
                    track_ids.append(t.track_id)
                    scores.append(t.score)
            results.append([frame_id, track_ids, x1y1whs, scores])
        write_results(str(out / "data" / f"{vid_name}.txt"), results)
        print("done", vid_name, flush=True)
    # --- official post-processing of run() ---
    model = PostLinker()
    model.load_state_dict(torch.load("./AFLink/AFLink_epoch20.pth"))
    aflink_dataset = LinkData("", "")
    for f in sorted(os.listdir(out / "data")):
        path_in, path_out = str(out / "data" / f), str(Path(str(out) + "_post") / "data" / f)
        if "Dance" in a.dataset:
            AFLink(path_in=path_in, path_out=path_out, model=model, dataset=aflink_dataset,
                   thrT=(0, 20), thrS=100, thrP=0.05).link()
        if "MOT" in a.dataset:
            gb_interpolation(path_in, path_out, interval=30, tau=12)
    if audit:
        json.dump(audit, open(out / "audit.json", "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
