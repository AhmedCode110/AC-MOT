"""
V7 development host: BoostTrack (Stanojevic & Todorovic, MVA 2024; official
code @ fb5bfc3, README setting --s_sim_corr), MOT17 val-half. BoostTrack is a
DEVELOPMENT / STRESS system for V7 (its V6 failure drove the redesign).

Both arms replay BoostTrack's own published YOLOX-X detections (conf 0.1,
NMS 0.7) and the same cached ECC transforms; the authors' post-processing
(linear interpolation, then GBI) is applied to both.
Host contract: assoc = birth = det_thresh 0.6 (single stage), low = 0.1 (the
detector emission floor: every emitted detection can be confidence-boosted),
match = 1 - iou_threshold = 0.7.

  python boosttrack_v7.py --system V7b[@mods] --name NAME
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as _np
for _n, _v in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(_np, _n):
        setattr(_np, _n, _v)
import numpy as np

ACMOT = Path("/Users/ahmedgouda/Desktop/Universal-ACMOT")
EXT = Path("/Users/ahmedgouda/Desktop/acmot_external")
BT = EXT / "BoostTrack"
DATA = EXT / "data_mirror/MOT17"
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
    ap.add_argument("--root", default=str(EXT / "runs/boosttrack"))
    a = ap.parse_args()
    sys.path.insert(0, str(ACMOT))
    v6drv = _load("bt_v6_driver", ACMOT / "tools/v6/external/boosttrack_v6.py")
    systems = _load("v7_systems", ACMOT / "tools/v7/systems.py")
    from acmot_v7 import HostContract, V7Layer, spec_from_dict
    GS = v6drv.setup_boosttrack()              # chdir(BT), official settings
    import utils
    from tracker.GBI import GBInterpolation
    from tracker.boost_track import BoostTrack

    baseline = a.system == "BASELINE"
    tf = floor = None
    if not baseline:
        base, ov, tf, floor, _ = systems.parse(a.system)
        spec = spec_from_dict(dict(ov, name=base))
    dets_cache = pickle.load(open(BT / "cache/det_bytetrack_ablation.pkl", "rb"))
    ann = json.load(open(DATA / "annotations/val_half.json"))
    results, per_frame = {}, []
    tracker = layer = host = None
    for im in ann["images"]:
        video, fid = im["file_name"].split("/")[0], int(im["frame_id"])
        tag = f"{video}:{fid}"
        results.setdefault(video, [])
        if fid == 1:
            if tracker is not None:
                tracker.dump_cache()
            tracker = BoostTrack(video_name=video)
            if not baseline:
                host = HostContract(assoc=float(tracker.det_thresh), birth=float(tracker.det_thresh),
                                    low=0.1, match=1.0 - float(tracker.iou_threshold))
                layer = V7Layer(spec, host)
        np_img = cv2.imread(str(DATA / "train" / im["file_name"]))
        pred = dets_cache[tag]
        pred = pred.cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)
        pred = np.asarray(pred, np.float32).reshape(-1, pred.shape[-1] if np.ndim(pred) == 2 else 5).copy()
        H, W = np_img.shape[:2]
        scale = min(800 / H, 1440 / W)
        if tf is not None:
            pred[:, 4] = TRANSFORMS[tf](np.clip(pred[:, 4].astype(np.float64), 1e-6, 1 - 1e-6))
        if floor is not None:
            pred = pred[pred[:, 4] >= floor]
        t0 = time.perf_counter()
        log = {}
        if baseline:
            d_in = pred
        else:
            m = layer.image_motion(np_img)
            t_l0 = time.perf_counter()
            dec = layer.step(pred[:, :4] / scale, pred[:, 4], m, classes=np.zeros(len(pred), int))
            t_layer = time.perf_counter() - t_l0
            d_in = pred[dec.keep].copy()
            d_in[:, 4] = np.asarray(dec.scores, np.float32)
            tracker.det_thresh = float(dec.assoc)
            tracker.iou_threshold = 1.0 - float(dec.match)
            log = dict(dec.log, t_layer=t_layer)
        t1 = time.perf_counter()
        targets = tracker.update(d_in, v6drv.ShapeOnly((1, 3, 800, 1440)), np_img, tag)
        t_trk = time.perf_counter() - t1
        if layer is not None:
            layer.observe(targets[:, :4] if len(targets) else np.zeros((0, 4)),
                          targets[:, 4].astype(int) if len(targets) else None)
        dt = time.perf_counter() - t0
        tlwhs, ids, confs = utils.filter_targets(targets, GS["aspect_ratio_thresh"], GS["min_box_area"])
        results[video].append((fid, tlwhs, ids, confs))
        rec = dict(seq=video, frame=fid, t_total=dt, t_trk=t_trk, n_det=len(pred), n_out=len(ids))
        rec.update({k: v for k, v in log.items() if isinstance(v, (int, float, str))})
        per_frame.append(rec)
    tracker.dump_cache()
    base_dir = Path(a.root) / "MOT17-val"
    folder = base_dir / a.name / "data"
    folder.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        utils.write_results_no_score(str(folder / f"{name}.txt"), res)
    json.dump(per_frame, open(base_dir / a.name / "per_frame.json", "w"))
    post = base_dir / (a.name + "_post")
    if post.exists():
        shutil.rmtree(post)
    shutil.copytree(base_dir / a.name, post)
    utils.dti(str(post / "data"), str(post / "data"), n_dti=1000, n_min=25)
    gbi = base_dir / (a.name + "_post_gbi") / "data"
    gbi.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(folder):
        GBInterpolation(path_in=str(post / "data" / f), path_out=str(gbi / f), interval=1000)
    print("done", a.name)


if __name__ == "__main__":
    main()
