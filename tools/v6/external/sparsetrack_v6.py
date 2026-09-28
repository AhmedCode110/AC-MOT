"""
External transfer: SparseTrack (Liu et al., IEEE TCSVT 2025, official code
github.com/hustvl/SparseTrack @ 499844f) + the FROZEN V6-TF layer.

Integration only. Nothing in this file tunes the layer:
  published YOLOX-X detections (cached from the official run, identical input
  for both arms)  ->  frozen V6-TF (UniversalACMOT live wrapper, policy file
  configs/universal_acmot_policy_v6tf.json, lock verified)  ->  official
  SparseTracker (pseudo-depth DCM + GMC)  ->  official output filter.

Generic tracker controls of the frozen layer mapped onto SparseTracker's API:
  association threshold -> args.track_thresh (first-stage DCM, "scores > thr")
  birth threshold       -> det_thresh (track initialisation)
  low threshold         -> SparseTracker's native 0.1 (unchanged)
  IoU-match tolerance   -> args.match_thresh (native 0.85, same semantics as
                           ByteTrack's match_thresh: max 1 - IoU cost)

  python sparsetrack_v6.py --mode {replay_baseline,v6} --cache DETS.pkl --out DIR
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as _np  # NUMPY2-COMPAT: removed aliases (behaviour-identical builtins)
for _n, _v in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(_np, _n):
        setattr(_np, _n, _v)
import numpy as np
import torch

ACMOT = Path("/Users/ahmedgouda/Desktop/Universal-ACMOT")
ST = Path("/Users/ahmedgouda/Desktop/acmot_external/SparseTrack")
sys.path.insert(0, str(ACMOT))
sys.path.insert(0, str(ST))


class PublishedDetections:
    """Detector adapter: returns the published detector's output for the
    current frame (x1, y1, x2, y2, score) as generic Detections."""
    capabilities = None

    def __init__(self):
        self.current = np.zeros((0, 5), np.float32)

    def detect(self, image, confidence, suppression, resolution):
        from adapters.types import Detection
        if suppression is not None:
            raise RuntimeError("published detector suppression is fixed (NMS 0.7)")
        return [Detection(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                          confidence=float(b[4]), class_id=0)
                for b in self.current if b[4] >= confidence]


class SparseTrackAdapter:
    """Tracker adapter around the OFFICIAL SparseTracker (unmodified)."""
    needs_image = True

    def __init__(self, track_args):
        self.base_args = track_args
        self.reset()

    def reset(self):
        from tracker.sparse_tracker import SparseTracker
        from tracker.basetrack import BaseTrack
        self.args = copy.deepcopy(self.base_args)
        self.tracker = SparseTracker(self.args)
        self._native_match = float(self.args.match_thresh)

    @property
    def native_match(self):
        return self._native_match

    def set_association_tolerance(self, value):
        self.tracker.args.match_thresh = float(value)

    def set_retention(self, frames):
        raise RuntimeError("V6-TF does not control retention")

    def update(self, detections, frame_shape, association_threshold=None,
               birth_threshold=None, image=None):
        from detectron2.structures import Boxes, Instances
        from adapters.types import Track
        if association_threshold is not None:
            self.tracker.args.track_thresh = float(association_threshold)
        if birth_threshold is not None:
            self.tracker.det_thresh = float(birth_threshold)
        b = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], np.float32).reshape(-1, 4)
        s = np.array([d.confidence for d in detections], np.float32)
        inst = Instances(tuple(frame_shape))
        inst.pred_boxes = Boxes(torch.from_numpy(b))
        inst.scores = torch.from_numpy(s)
        _t0 = time.perf_counter()
        out = self.tracker.update(inst, image)
        self.last_trk_time = time.perf_counter() - _t0
        self.last_stracks = out
        tr = []
        for t in out:
            x1, y1, w, h = t.tlwh
            tr.append(Track(x1=float(x1), y1=float(y1), x2=float(x1 + w), y2=float(y1 + h),
                            track_id=int(t.track_id), confidence=float(t.score), class_id=0))
        return tr


def official_filter(stracks, min_box_area):
    """The official evaluator's output filter (evaluators.py), verbatim logic."""
    tlwhs, ids, scores = [], [], []
    for t in stracks:
        tlwh = t.tlwh
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > min_box_area and not vertical:
            tlwhs.append(tlwh)
            ids.append(t.track_id)
            scores.append(t.score)
    return tlwhs, ids, scores


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                f.write(save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1),
                                           w=round(w, 1), h=round(h, 1), s=round(score, 2)))


def track_args():
    """Official mot17_ab_track_cfg.py `track` dict (values copied verbatim)."""
    from omegaconf import OmegaConf
    return OmegaConf.create(dict(
        experiment_name="yolox_mix17_ablation_det", track_thresh=0.6, track_buffer=30,
        match_thresh=0.85, min_box_area=100, down_scale=4, depth_levels=1, depth_levels_low=8,
        confirm_thresh=0.7, mot20=False, byte=False, deep=True, bot=False, sort=False,
        ocsort=False, fp16=True, fuse=True, val_ann="val_half.json", is_public=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay_baseline", "v6"], required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/Users/ahmedgouda/Desktop/acmot_external/data_mirror/MOT17",
                    help="byte-identical local mirror of the Drive MOT17 (sha256 manifest)")
    ap.add_argument("--diag", nargs="*", default=[],
                    help="DIAGNOSTIC ONLY (failure analysis, never the method): k=v overrides "
                         "applied to an in-memory copy of the frozen policy")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = pickle.load(open(a.cache, "rb"))       # {file_name: np.array Nx5 (x1,y1,x2,y2,score)}
    ann = json.load(open(Path(a.data) / "annotations/val_half.json"))
    images = ann["images"]
    targs = track_args()

    if a.mode == "v6":
        import importlib.util as _ilu   # load the repo's lock verifier by path
        _sp = _ilu.spec_from_file_location("acmot_v6_dev", ACMOT / "tools/v6/dev.py")
        _dev = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_dev)
        _dev.verify_lock()              # frozen files byte-identical to the lock                               # frozen files byte-identical
        from universal_acmot import UniversalACMOT, load_policy
        policy, dk = load_policy(ACMOT / "configs/universal_acmot_policy_v6tf.json")
        if a.diag:
            from dataclasses import fields, replace as _rep
            ty = {f.name: f.type for f in fields(policy)}
            conv = lambda k, v: (v in ("1", "True")) if "bool" in str(ty[k]) else float(v) if "float" in str(ty[k]) else int(v) if "int" in str(ty[k]) else v
            policy = _rep(policy, **{k: conv(k, v) for k, v in (x.split("=") for x in a.diag)})
            print("DIAGNOSTIC policy override (not the frozen method):", a.diag)

    from tracker.sparse_tracker import SparseTracker
    results, per_frame, video = [], [], None
    tracker = layer = det = None
    seq_results = {}
    for im in images:
        seq = im["file_name"].split("/")[0]
        fid = int(im["frame_id"])
        if fid == 1:
            if video is not None:
                seq_results[video] = results
            video, results = seq, []
            if a.mode == "v6":
                det = PublishedDetections()
                layer = UniversalACMOT(det, SparseTrackAdapter(targs), policy=policy, density_kwargs=dk)
            else:
                tracker = SparseTracker(copy.deepcopy(targs))
        img = cv2.imread(str(Path(a.data) / "train" / im["file_name"]))
        d = cache[im["file_name"]]
        t0 = time.perf_counter()
        if a.mode == "v6":
            det.current = d
            layer(img)
            stracks = layer.pipeline.tracker.last_stracks
            t_trk = layer.pipeline.tracker.last_trk_time
            aud = layer.last["audit"]
            n_in, n_pass = int(aud["raw_count"]), int(aud["accepted_after_topk"])
            extra = dict(t1=float(aud.get("tf_otsu_t1", np.nan)), t2=float(aud.get("tf_otsu_t2", np.nan)),
                         n_primary=int(aud.get("tf_n_primary", 0)), n_extension=int(aud.get("tf_n_secondary", 0)),
                         motion_ratio=float(aud.get("tf_motion_ratio", np.nan)),
                         match_thresh=float(layer.pipeline.tracker.tracker.args.match_thresh))
        else:
            extra = {}
            from detectron2.structures import Boxes, Instances
            inst = Instances((img.shape[0], img.shape[1]))
            inst.pred_boxes = Boxes(torch.from_numpy(d[:, :4].copy()))
            inst.scores = torch.from_numpy(d[:, 4].copy())
            _t1 = time.perf_counter()
            stracks = tracker.update(inst, img)
            t_trk = time.perf_counter() - _t1
            n_in = n_pass = len(d)
        dt = time.perf_counter() - t0
        tl, ids, sc = official_filter(stracks, targs.min_box_area)
        results.append((fid, tl, ids, sc))
        per_frame.append(dict(seq=seq, frame=fid, t_total=dt, t_trk=t_trk, n_in=n_in, n_pass=n_pass,
                              n_out=len(ids), **extra))
    seq_results[video] = results
    for s, r in seq_results.items():
        write_results(out / f"{s}.txt", r)
    json.dump(per_frame, open(out / "per_frame.json", "w"))
    print("wrote", len(seq_results), "sequences to", out)


if __name__ == "__main__":
    main()
