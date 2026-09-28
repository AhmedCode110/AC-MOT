"""
V7 development host: SparseTrack (Liu et al., IEEE TCSVT 2025; official code
@ 499844f), MOT17 val-half. SparseTrack is a DEVELOPMENT / STRESS system for
V7 (its V6 failure drove the redesign); it is never external evidence for V7.

Both arms replay the SAME published YOLOX-X detections of the official run:
  replay_baseline : published detections -> official SparseTracker
  v7              : published detections -> V7 layer -> official SparseTracker
Host contract (the tracker's own operating point, official config):
  assoc = track_thresh 0.6, birth = det_thresh 0.7 (track_thresh + 0.1 for
  the half split), low = 0.1 (hard-coded second stage), match = 0.85.

  python sparsetrack_v7.py --system V7a[@mods] --name NAME [--transform T] [--floor F]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as _np
for _n, _v in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(_np, _n):
        setattr(_np, _n, _v)
import numpy as np
import torch

ACMOT = Path("/Users/ahmedgouda/Desktop/Universal-ACMOT")
EXT = Path("/Users/ahmedgouda/Desktop/acmot_external")
ST = EXT / "SparseTrack"
sys.path.insert(0, str(ACMOT))
sys.path.insert(0, str(ST))

TRANSFORMS = {
    None: lambda s: s,
    "temp2": lambda s: 1 / (1 + np.exp(-np.log(s / (1 - s)) / 2.0)),
    "temp05": lambda s: 1 / (1 + np.exp(-np.log(s / (1 - s)) / 0.5)),
    "scale05": lambda s: 0.5 * s,
    "pow3": lambda s: s ** 3,
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="V7 system or 'BASELINE'")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache", default=str(EXT / "runs/sparsetrack_A_official/published_detections.pkl"))
    ap.add_argument("--data", default=str(EXT / "data_mirror/MOT17"))
    ap.add_argument("--root", default=str(EXT / "runs/sparsetrack"))
    a = ap.parse_args()
    v6drv = _load("st_v6_driver", ACMOT / "tools/v6/external/sparsetrack_v6.py")
    systems = _load("v7_systems", ACMOT / "tools/v7/systems.py")
    from acmot_v7 import HostContract, V7Layer, spec_from_dict
    from tracker.sparse_tracker import SparseTracker
    from detectron2.structures import Boxes, Instances

    baseline = a.system == "BASELINE"
    tf = floor = None
    if not baseline:
        base, ov, tf, floor, _ = systems.parse(a.system)
        spec = spec_from_dict(dict(ov, name=base))
    targs = v6drv.track_args()
    host = HostContract(assoc=float(targs.track_thresh), birth=float(targs.track_thresh) + 0.1,
                        low=0.1, match=float(targs.match_thresh))
    cache = pickle.load(open(a.cache, "rb"))
    ann = json.load(open(Path(a.data) / "annotations/val_half.json"))
    out = Path(a.root) / "MOT17-val" / a.name / "data"
    out.mkdir(parents=True, exist_ok=True)
    seq_results, per_frame, results, video = {}, [], [], None
    tracker = layer = None
    for im in ann["images"]:
        seq, fid = im["file_name"].split("/")[0], int(im["frame_id"])
        if fid == 1:
            if video is not None:
                seq_results[video] = results
            video, results = seq, []
            tracker = SparseTracker(copy.deepcopy(targs))
            if not baseline:
                layer = V7Layer(spec, host)
        img = cv2.imread(str(Path(a.data) / "train" / im["file_name"]))
        d = np.asarray(cache[im["file_name"]], np.float32).reshape(-1, 5).copy()
        if tf is not None:
            d[:, 4] = TRANSFORMS[tf](np.clip(d[:, 4].astype(np.float64), 1e-6, 1 - 1e-6))
        if floor is not None:
            d = d[d[:, 4] >= floor]
        t0 = time.perf_counter()
        log = {}
        if baseline:
            boxes, scores = d[:, :4], d[:, 4]
        else:
            m = layer.image_motion(img)
            t_l0 = time.perf_counter()
            dec = layer.step(d[:, :4], d[:, 4], m, classes=np.zeros(len(d), int))
            t_layer = time.perf_counter() - t_l0
            boxes = d[dec.keep, :4]
            scores = np.asarray(dec.scores, np.float32)
            tracker.args.track_thresh = float(dec.assoc)
            tracker.det_thresh = float(dec.birth)
            tracker.args.match_thresh = float(dec.match)
            log = dict(dec.log, t_layer=t_layer)
        inst = Instances((img.shape[0], img.shape[1]))
        inst.pred_boxes = Boxes(torch.from_numpy(np.ascontiguousarray(boxes, np.float32)))
        inst.scores = torch.from_numpy(np.ascontiguousarray(scores, np.float32))
        t1 = time.perf_counter()
        stracks = tracker.update(inst, img)
        t_trk = time.perf_counter() - t1
        if layer is not None:
            layer.observe([t.tlbr for t in stracks], [t.track_id for t in stracks])
        dt = time.perf_counter() - t0
        tl, ids, sc = v6drv.official_filter(stracks, targs.min_box_area)
        results.append((fid, tl, ids, sc))
        rec = dict(seq=seq, frame=fid, t_total=dt, t_trk=t_trk, n_det=len(d), n_out=len(ids))
        rec.update({k: v for k, v in log.items() if isinstance(v, (int, float, str))})
        per_frame.append(rec)
    seq_results[video] = results
    for s, r in seq_results.items():
        v6drv.write_results(out / f"{s}.txt", r)
    json.dump(per_frame, open(out.parent / "per_frame.json", "w"))
    print("wrote", len(seq_results), "sequences to", out)


if __name__ == "__main__":
    main()
