"""
V7 DEVELOPMENT host (cloud fallback, 2026-09-28): ByteTrack on MOT17
val-half from the PUBLISHED YOLOX-X detections, without MOT17 frames.

The MOT17 frames are unreachable in the cloud environment (motchallenge.net
denied). ByteTrack needs no pixels (IoU + Kalman only; no CMC, no ReID), so
this configuration is an EXACT evaluation of itself: detections + GT are
the original files. It is NOT a reproduction of a published MOT17 result
and it is a development system for V7 (never external evidence).

Streams (the same YOLOX-X detector of ByteTrack's ablation, two emission floors):
  st  SparseTrack's published detections (conf floor 0.01, original coords)
  bt  BoostTrack's published detections  (conf floor 0.1, NMS 0.7, /scale)
Host contracts (documented operating points, not tuned):
  ultra     ultralytics bytetrack.yaml (the VisDrone development host):
            assoc 0.25, birth 0.25, low 0.1, match 0.8, fuse on, no output filter
  official  ByteTrack's MOT17 setting (ifzhang/ByteTrack, mot17 args):
            track_thresh 0.6 -> assoc 0.6, birth 0.7 (track_thresh + 0.1),
            low 0.1, match 0.8, fuse on; output filter w/h > 1.6 or area
            <= 100 removed (official demo/eval writer).
Motion cue: unavailable without frames -> motion=None (the V7 motion rule is
inactive; stated in every result).

  python mot17_bytetrack_v7.py --system <S|BASELINE> --stream st|bt --host ultra|official --name N
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ACMOT = Path(os.environ.get("ACMOT_ROOT", Path(__file__).resolve().parents[3]))
EXT = Path(os.environ.get("ACMOT_EXT", "/Users/ahmedgouda/Desktop/acmot_external"))
sys.path.insert(0, str(ACMOT))

TRANSFORMS = {
    None: lambda s: s,
    "temp2": lambda s: 1 / (1 + np.exp(-np.log(s / (1 - s)) / 2.0)),
    "temp05": lambda s: 1 / (1 + np.exp(-np.log(s / (1 - s)) / 0.5)),
    "scale05": lambda s: 0.5 * s,
    "pow3": lambda s: s ** 3,
}
HOSTS = {"ultra": dict(assoc=0.25, birth=0.25, low=0.1, match=0.8, filt=False),
         "official": dict(assoc=0.6, birth=0.7, low=0.1, match=0.8, filt=True),
         # OC-SORT (Cao et al., CVPR 2023; noahcao/OC_SORT @ 8462e7e), official MOT17
         # args: track_thresh 0.6 (single threshold: association = birth), iou_thresh
         # 0.3 -> match 0.7, delta_t 3, inertia 0.2, asso iou, no BYTE stage: the
         # lowest score it can use is its det_thresh, so low = 0.6 (declared host
         # property; runs before 2026-09-28 16:00 declared 0.1, which no V7 option
         # read); writer drops w/h > 1.6 or area <= 10.
         "ocsort": dict(assoc=0.6, birth=0.6, low=0.6, match=0.7, filt=True, min_area=10)}


class OCSortHost:
    """Official OC-SORT tracker driven through the generic controls."""

    def __init__(self, h):
        root = EXT / "OC_SORT"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from trackers.ocsort_tracker.ocsort import OCSort
        self.t = OCSort(det_thresh=h["assoc"], iou_threshold=1.0 - h["match"], asso_func="iou",
                        delta_t=3, inertia=0.2)

    def set_association_tolerance(self, m):
        self.t.iou_threshold = 1.0 - float(m)

    def update(self, dets, shape, assoc):
        from types import SimpleNamespace
        self.t.det_thresh = float(assoc)
        a = np.asarray(dets, np.float64).reshape(-1, 5)
        out = self.t.update(a.copy(), shape, shape)
        return [SimpleNamespace(x1=r[0], y1=r[1], x2=r[2], y2=r[3], track_id=int(r[4]), confidence=1.0)
                for r in out]


def load_stream(stream, ann):
    if stream == "st":
        c = pickle.load(open(EXT / "runs/sparsetrack_A_official/published_detections.pkl", "rb"))
        return {im["file_name"]: np.asarray(c[im["file_name"]], np.float64).reshape(-1, 5)
                for im in ann["images"]}
    c = pickle.load(open(EXT / "BoostTrack/cache/det_bytetrack_ablation.pkl", "rb"))
    out = {}
    for im in ann["images"]:
        seq, fid = im["file_name"].split("/")[0], int(im["frame_id"])
        p = c[f"{seq}:{fid}"]
        p = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
        p = np.asarray(p, np.float64).reshape(-1, p.shape[-1] if np.ndim(p) == 2 else 5)[:, :5].copy()
        scale = min(800 / im["height"], 1440 / im["width"])
        p[:, :4] /= scale
        out[im["file_name"]] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--stream", required=True, choices=["st", "bt"])
    ap.add_argument("--host", required=True, choices=list(HOSTS))
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default=str(EXT / "runs/bytetrack_mot17"))
    a = ap.parse_args()
    from acmot_v7 import HostContract, V7Layer, spec_from_dict
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from adapters.types import Detection
    from tools.v7.systems import parse
    from ultralytics.trackers.basetrack import BaseTrack

    ann = json.load(open(EXT / "data_mirror/MOT17/annotations/val_half.json"))
    dets = load_stream(a.stream, ann)
    h = HOSTS[a.host]
    host = HostContract(assoc=h["assoc"], birth=h["birth"], low=h["low"], match=h["match"])
    baseline = a.system == "BASELINE"
    tf = floor = None
    if not baseline:
        base, ov, tf, floor, _ = parse(a.system)
        spec = spec_from_dict(dict(ov, name=base))
    out = Path(a.root) / "MOT17-val" / a.name / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows, audit, tr, layer, video = {}, [], None, None, None
    for im in ann["images"]:
        seq, fid = im["file_name"].split("/")[0], int(im["frame_id"])
        if fid == 1:
            BaseTrack.reset_id()
            tr = OCSortHost(h) if a.host == "ocsort" else ByteTrackAdapter(
                high=h["assoc"], low=h["low"], new=h["birth"], buffer=30, match=h["match"], fuse=True)
            layer = None if baseline else V7Layer(spec, host)
            rows[seq] = []
        d = dets[im["file_name"]].copy()
        if tf is not None:
            d[:, 4] = TRANSFORMS[tf](np.clip(d[:, 4], 1e-6, 1 - 1e-6))
        if floor is not None:
            d = d[d[:, 4] >= floor]
        if baseline:
            keep, sc, assoc, birth, match, log = np.arange(len(d)), d[:, 4], h["assoc"], h["birth"], h["match"], {}
        else:
            dec = layer.step(d[:, :4], d[:, 4], None, classes=np.zeros(len(d), int))
            keep, sc, assoc, birth, match, log = dec.keep, dec.scores, dec.assoc, dec.birth, dec.match, dec.log
        det_list = [Detection(x1=float(d[k, 0]), y1=float(d[k, 1]), x2=float(d[k, 2]), y2=float(d[k, 3]),
                              confidence=float(v), class_id=0) for k, v in zip(keep, sc)]
        tr.set_association_tolerance(match)
        if a.host == "ocsort":
            # single-threshold host: V7 assoc and birth coincide for it (birth = assoc contract)
            tracks = tr.update([[x.x1, x.y1, x.x2, x.y2, x.confidence] for x in det_list],
                               (im["height"], im["width"]), min(assoc, birth))
        else:
            tracks = tr.update(det_list, (im["height"], im["width"]), association_threshold=assoc,
                               birth_threshold=birth)
        if layer is not None:
            layer.observe([[t.x1, t.y1, t.x2, t.y2] for t in tracks], [t.track_id for t in tracks])
        n_out = 0
        for t in tracks:
            w, hh = t.x2 - t.x1, t.y2 - t.y1
            if h["filt"] and (w / max(hh, 1e-9) > 1.6 or w * hh <= h.get("min_area", 100)):
                continue
            rows[seq].append(f"{fid},{t.track_id},{t.x1:.2f},{t.y1:.2f},{w:.2f},{hh:.2f},{t.confidence:.2f},-1,-1,-1\n")
            n_out += 1
        rec = dict(seq=seq, frame=fid, n_det=len(d), n_out=n_out, tracks=len(tracks))
        rec.update({k: v for k, v in log.items() if isinstance(v, (int, float, str))})
        audit.append(rec)
    for s, r in rows.items():
        open(out / f"{s}.txt", "w").writelines(r)
    json.dump(dict(audit=audit, system=a.system, stream=a.stream, host=a.host, motion_cue=None,
                   code_sha=__import__("hashlib").sha256((ACMOT / "acmot_v7.py").read_bytes()).hexdigest()),
              open(out.parent / "per_frame.json", "w"))
    print("wrote", a.name)


if __name__ == "__main__":
    main()
