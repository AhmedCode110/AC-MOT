"""
POST-FREEZE external system S1: PD-SORT (Wang et al., IEEE Transactions on
Consumer Electronics 2025; github.com/Wangyc2000/PD_SORT @ af21db6), MOT17
val-half, predeclared in research/final/V7_EXTERNAL_SELECTION.md.

Both arms replay the authors' RELEASED tracker-input detections
(res_mot/MOT17-val/yolox_x_ablation_results/<run>/MOT17-*_detections.txt) through
the unmodified PD-SORT tracker with the README MOT17-val arguments; the
evaluator loop of yolox/evaluators/mot_evaluator.py is reproduced (tracker
re-created at frame 1, update skipped on frames without detections, output
filter area > min_box_area 100 and w/h <= 1.6, write_results_no_score).
Frames are unreachable: the tracker reads pixels only through CMC, which is
served from the shipped GMC files, so a zero image of the right shape is
passed (verified by reproducing the authors' released outputs).

  python pdsort_v7.py --system BASELINE|<frozen system> --name N [--dets-run oc-pd-cmc-pdlev5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ACMOT = Path(os.environ.get("ACMOT_ROOT", Path(__file__).resolve().parents[3]))
EXT = Path(os.environ.get("ACMOT_EXT", "/Users/ahmedgouda/Desktop/acmot_external"))
PD = EXT / "PD_SORT"
RES = PD / "res_mot/MOT17-val/yolox_x_ablation_results"
ARGS = dict(track_thresh=0.6, iou_thresh=0.3, asso="iou", deltat=3, inertia=0.2,
            min_box_area=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dets-run", default="oc-pd-cmc-pdlev5")
    ap.add_argument("--root", default=str(EXT / "runs/pdsort"))
    a = ap.parse_args()
    sys.path.insert(0, str(ACMOT))
    os.chdir(PD)                       # CMC reads ./cache/cmc_files
    sys.path.insert(0, str(PD))
    from trackers.ocsort_tracker.ocsort import OCSort
    baseline = a.system == "BASELINE"
    if not baseline:
        from acmot_v7 import HostContract, V7Layer, spec_from_dict
        cfg = json.loads((ACMOT / "configs/universal_acmot_policy_v7.json").read_text())
        assert a.system == cfg["system"], "post-freeze: only the frozen policy may run"
        spec = spec_from_dict(cfg["spec"])
        host = HostContract(assoc=ARGS["track_thresh"], birth=ARGS["track_thresh"],
                            low=ARGS["track_thresh"], match=1.0 - ARGS["iou_thresh"])
    ann = json.load(open(EXT / "data_mirror/MOT17/annotations/val_half.json"))
    by_id = {im["id"]: im for im in ann["images"]}
    out_dir = Path(a.root) / "MOT17-val" / a.name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = []
    for seq in sorted({im["file_name"].split("/")[0] for im in ann["images"]}):
        rows = np.loadtxt(RES / a.dets_run / f"{seq}_detections.txt", delimiter=",", ndmin=2)
        dets = defaultdict(list)
        for r in rows:
            dets[int(r[0])].append(r)
        ims = sorted((im for im in ann["images"] if im["file_name"].startswith(seq + "/")),
                     key=lambda im: im["frame_id"])
        tracker = layer = None
        results = []
        for im in ims:
            fid, h, w = int(im["frame_id"]), im["height"], im["width"]
            if fid == 1:
                tracker = OCSort(det_thresh=ARGS["track_thresh"], iou_threshold=ARGS["iou_thresh"],
                                 asso_func=ARGS["asso"], delta_t=ARGS["deltat"], inertia=ARGS["inertia"])
                layer = None if baseline else V7Layer(spec, host)
            d = dets.get(im["id"])
            if not d:                   # the official loop skips frames without detections
                continue
            d = np.asarray(d)
            o = np.c_[d[:, 2], d[:, 3], d[:, 2] + d[:, 4], d[:, 3] + d[:, 5], d[:, 6]].astype(np.float64)
            if layer is not None:
                dec = layer.step(o[:, :4], o[:, 4], None, classes=np.zeros(len(o), int))
                o = np.c_[o[dec.keep, :4], np.asarray(dec.scores, np.float64)]
                tracker.det_thresh = float(min(dec.assoc, dec.birth))
                tracker.iou_threshold = 1.0 - float(dec.match)
                audit.append(dict(seq=seq, frame=fid, **{k: v for k, v in dec.log.items()
                                                          if isinstance(v, (int, float, str))}))
                o = o.reshape(-1, 5)
            img = np.zeros((h, w, 3), np.uint8)
            tag = f"{seq}:{fid}"
            targets = tracker.update(o.copy(), None, img, (h, w), (h, w), tag)
            if layer is not None:
                layer.observe(targets[:, :4] if len(targets) else np.zeros((0, 4)),
                              targets[:, 4].astype(int) if len(targets) else [])
            tl, ids = [], []
            for t in targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                if tlwh[2] * tlwh[3] > ARGS["min_box_area"] and not tlwh[2] / tlwh[3] > 1.6:
                    tl.append(tlwh)
                    ids.append(t[4])
            results.append((fid, tl, ids))
        with open(out_dir / f"{seq}.txt", "w") as f:
            for fid, tls, ids in results:
                for (x1, y1, ww, hh), tid in zip(tls, ids):
                    if tid < 0:
                        continue
                    f.write(f"{fid},{tid},{round(x1, 1)},{round(y1, 1)},{round(ww, 1)},{round(hh, 1)},-1,-1,-1,-1\n")
    json.dump(audit, open(out_dir.parent / "audit.json", "w"))
    print("wrote", a.name)


if __name__ == "__main__":
    main()
