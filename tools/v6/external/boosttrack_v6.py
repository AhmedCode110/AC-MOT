"""
Supporting external transfer: BoostTrack (Stanojevic & Todorovic, Machine
Vision and Applications 2024; official code github.com/vukasin-stanojevic/
BoostTrack @ fb5bfc3; README "Run BoostTrack" setting, --s_sim_corr) + the
FROZEN V6-TF layer.

Both arms replay the SAME published YOLOX-X detections (BoostTrack's own
detector cache, conf 0.1 / NMS 0.7) and the same cached ECC transforms:
  replay_baseline : published detections -> official BoostTrack
  v6              : published detections -> frozen V6-TF -> official BoostTrack
then the authors' post-processing (linear interpolation, GBI) on both.

Generic controls of the frozen layer -> BoostTrack API (integration only):
  association = birth threshold -> det_thresh (single stage; BoostTrack filters
                                   and initialises with the same threshold)
  IoU-match tolerance           -> 1 - iou_threshold (native 0.3 -> 0.7)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as _np  # NUMPY2-COMPAT
for _n, _v in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(_np, _n):
        setattr(_np, _n, _v)
import numpy as np

ACMOT = Path("/Users/ahmedgouda/Desktop/Universal-ACMOT")
BT = Path("/Users/ahmedgouda/Desktop/acmot_external/BoostTrack")
DATA = Path("/Users/ahmedgouda/Desktop/acmot_external/data_mirror/MOT17")


def setup_boosttrack():
    os.chdir(BT)
    sys.path.insert(0, str(BT))
    sys.path.insert(0, str(BT / "external/YOLOX"))
    sys.modules.setdefault("torchreid", types.ModuleType("torchreid"))
    fr = types.ModuleType("external.adaptors.fastreid_adaptor")
    fr.FastReID = type("FastReID", (), {})
    sys.modules["external.adaptors.fastreid_adaptor"] = fr
    from default_settings import (BoostTrackPlusPlusSettings, BoostTrackSettings,
                                  GeneralSettings)
    GeneralSettings.values.update(dataset="mot17", use_embedding=False, use_ecc=True,
                                  test_dataset=False)
    BoostTrackSettings.values["s_sim_corr"] = True
    BoostTrackPlusPlusSettings.values.update(use_rich_s=False, use_sb=False, use_vt=False)
    return GeneralSettings


class ShapeOnly:
    """Stands in for the network-input tensor; BoostTrack only reads .shape."""
    def __init__(self, shape):
        self.shape = shape


class BoostTrackAdapter:
    needs_image = True

    def __init__(self, video_name):
        self.video_name = video_name
        self.tag = None
        self.reset()

    def reset(self):
        from tracker.boost_track import BoostTrack
        self.bt = BoostTrack(video_name=self.video_name)
        self._native_match = 1.0 - float(self.bt.iou_threshold)

    @property
    def native_match(self):
        return self._native_match

    def set_association_tolerance(self, value):
        self.bt.iou_threshold = 1.0 - float(value)

    def set_retention(self, frames):
        raise RuntimeError("V6-TF does not control retention")

    def update(self, detections, frame_shape, association_threshold=None,
               birth_threshold=None, image=None):
        from adapters.types import Track
        if association_threshold is not None:
            self.bt.det_thresh = float(association_threshold)
        d = np.array([[x.x1, x.y1, x.x2, x.y2, x.confidence] for x in detections],
                     np.float32).reshape(-1, 5)
        h, w = image.shape[:2]
        tg = self.bt.update(d, ShapeOnly((1, 3, h, w)), image, self.tag)   # scale = 1
        self.last_targets = tg
        return [Track(x1=float(t[0]), y1=float(t[1]), x2=float(t[2]), y2=float(t[3]),
                      track_id=int(t[4]), confidence=float(t[5]), class_id=0) for t in tg]


class PublishedDetections:
    capabilities = None

    def __init__(self):
        self.current = np.zeros((0, 5), np.float32)

    def detect(self, image, confidence, suppression, resolution):
        from adapters.types import Detection
        if suppression is not None:
            raise RuntimeError("published detector suppression is fixed")
        return [Detection(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                          confidence=float(b[4]), class_id=0) for b in self.current if b[4] >= confidence]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay_baseline", "v6"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default="/Users/ahmedgouda/Desktop/acmot_external/runs/boosttrack")
    a = ap.parse_args()
    sys.path.insert(0, str(ACMOT))
    if a.mode == "v6":
        import importlib.util as _ilu   # load the repo's lock verifier by path
        _sp = _ilu.spec_from_file_location("acmot_v6_dev", ACMOT / "tools/v6/dev.py")
        _dev = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_dev)
        _dev.verify_lock()              # frozen files byte-identical to the lock
        from universal_acmot import UniversalACMOT, load_policy
        policy, dk = load_policy(ACMOT / "configs/universal_acmot_policy_v6tf.json")
    GS = setup_boosttrack()
    import torch
    import utils
    from tracker.GBI import GBInterpolation
    from tracker.boost_track import BoostTrack
    dets_cache = pickle.load(open(BT / "cache/det_bytetrack_ablation.pkl", "rb"))
    ann = json.load(open(DATA / "annotations/val_half.json"))
    results, per_frame, prev = {}, [], None
    tracker = layer = det = adapter = None
    for im in ann["images"]:
        video, fid = im["file_name"].split("/")[0], int(im["frame_id"])
        tag = f"{video}:{fid}"
        results.setdefault(video, [])
        if fid == 1:
            if tracker is not None:
                tracker.dump_cache()
            if a.mode == "v6":
                det = PublishedDetections()
                adapter = BoostTrackAdapter(video)
                layer = UniversalACMOT(det, adapter, policy=policy, density_kwargs=dk)
                tracker = adapter.bt
            else:
                tracker = BoostTrack(video_name=video)
        np_img = cv2.imread(str(DATA / "train" / im["file_name"]))
        pred = dets_cache[tag]
        pred = pred.cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)
        H, W = np_img.shape[:2]
        scale = min(800 / H, 1440 / W)
        t0 = time.perf_counter()
        if a.mode == "v6":
            d = pred.copy()
            d[:, :4] /= scale
            det.current = d
            adapter.tag = tag
            layer(np_img)
            if layer.pipeline.tracker is not adapter:
                raise RuntimeError("adapter mismatch")
            tracker = adapter.bt          # reset() inside the layer rebuilds it
            targets = adapter.last_targets
            aud = layer.last["audit"]
            n_in, n_pass = int(aud["raw_count"]), int(aud["accepted_after_topk"])
        else:
            targets = tracker.update(pred.copy(), ShapeOnly((1, 3, 800, 1440)), np_img, tag)
            n_in = n_pass = len(pred)
        dt = time.perf_counter() - t0
        tlwhs, ids, confs = utils.filter_targets(targets, GS["aspect_ratio_thresh"], GS["min_box_area"])
        results[video].append((fid, tlwhs, ids, confs))
        per_frame.append(dict(seq=video, frame=fid, t=dt, n_in=n_in, n_pass=n_pass, n_out=len(ids)))
    tracker.dump_cache()
    base = Path(a.root) / "MOT17-val"
    folder = base / a.name / "data"
    folder.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        utils.write_results_no_score(str(folder / f"{name}.txt"), res)
    json.dump(per_frame, open(base / a.name / "per_frame.json", "w"))
    import shutil
    post = base / (a.name + "_post")
    if post.exists():
        shutil.rmtree(post)
    shutil.copytree(base / a.name, post)
    utils.dti(str(post / "data"), str(post / "data"), n_dti=1000, n_min=25)
    gbi = base / (a.name + "_post_gbi") / "data"
    gbi.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(folder):
        GBInterpolation(path_in=str(post / "data" / f), path_out=str(gbi / f), interval=1000)
    print("done", a.name)


if __name__ == "__main__":
    main()
