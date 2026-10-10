"""
POST-FREEZE external system S2: Hybrid-SORT (Yang et al., AAAI 2024;
github.com/ymzis69/HybridSORT @ 396f8d3), configuration
exps/example/mot/yolox_x_ablation_hybrid_sort.py (no ReID), MOT17 val-half,
predeclared in research/final/V7_EXTERNAL_SELECTION.md.

Tracker input: the YOLOX-X `ocsort_mot17_ablation` MOT17 val-half detections
(conf 0.1, NMS 0.7, 800x1440) released with PD-SORT. The official evaluator
loop (yolox/evaluators/mot_evaluator_dance.py::evaluate_hybrid_sort) is
reproduced: tracker re-created at frame 1, update on every frame (None when
the frame has no detection), output filter area > min_box_area and
w/h <= 1.6, write_results_no_score. Arguments: the repository parser defaults
merged with the experiment file (utils.args.args_merge_params_form_exp).

  python hybridsort_v7.py --system BASELINE|<frozen system> --name N
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
HS = EXT / "HybridSORT"
DETS = EXT / "PD_SORT/res_mot/MOT17-val/yolox_x_ablation_results/oc-pd-cmc-pdlev5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default=str(EXT / "runs/hybridsort"))
    a = ap.parse_args()
    os.chdir(HS)
    sys.path.insert(0, str(HS))
    from utils.args import make_parser
    from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort
    args = make_parser().parse_args(["-f", "exps/example/mot/yolox_x_ablation_hybrid_sort.py"])
    # args_merge_params_form_exp(args, exp) without importing the detector stack:
    # the experiment file's literal `self.<k> = <v>` assignments that name a
    # parser argument (identical effect for these keys; yolox_base adds only
    # `dataset`, which the experiment overrides with "mot17").
    import ast
    tree = ast.parse(open(args.exp_file).read())
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
                and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "self"
                and n.targets[0].attr in args.__dict__):
            try:
                setattr(args, n.targets[0].attr, ast.literal_eval(n.value))
            except ValueError:
                pass
    args.dataset = "mot17"
    sys.path.insert(0, str(ACMOT))
    baseline = a.system == "BASELINE"
    if not baseline:
        from acmot_v7 import HostContract, V7Layer, spec_from_dict
        cfg = json.loads((ACMOT / "configs/universal_acmot_policy_v7.json").read_text())
        assert a.system == cfg["system"], "post-freeze: only the frozen policy may run"
        spec = spec_from_dict(cfg["spec"])
        host = HostContract(assoc=args.track_thresh, birth=args.track_thresh,
                            low=0.1 if args.use_byte else args.track_thresh,
                            match=1.0 - args.iou_thresh)
    ann = json.load(open(EXT / "data_mirror/MOT17/annotations/val_half.json"))
    out_dir = Path(a.root) / "MOT17-val" / a.name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_args = {k: getattr(args, k) for k in ("track_thresh", "iou_thresh", "asso", "deltat", "inertia",
                                              "use_byte", "min_box_area", "TCM_first_step",
                                              "TCM_byte_step", "TCM_first_step_weight",
                                              "TCM_byte_step_weight", "hybrid_sort_with_reid")}
    audit = []
    for seq in sorted({im["file_name"].split("/")[0] for im in ann["images"]}):
        rows = np.loadtxt(DETS / f"{seq}_detections.txt", delimiter=",", ndmin=2)
        dets = defaultdict(list)
        for r in rows:
            dets[int(r[0])].append(r)
        ims = sorted((im for im in ann["images"] if im["file_name"].startswith(seq + "/")),
                     key=lambda im: im["frame_id"])
        results, tracker, layer = [], None, None
        for im in ims:
            fid, h, w = int(im["frame_id"]), im["height"], im["width"]
            if fid == 1:
                tracker = Hybrid_Sort(args, det_thresh=args.track_thresh, iou_threshold=args.iou_thresh,
                                      asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia,
                                      use_byte=args.use_byte)
                layer = None if baseline else V7Layer(spec, host)
            d = dets.get(im["id"])
            o = None
            if d:
                d = np.asarray(d)
                o = np.c_[d[:, 2], d[:, 3], d[:, 2] + d[:, 4], d[:, 3] + d[:, 5], d[:, 6]].astype(np.float64)
                if layer is not None:
                    dec = layer.step(o[:, :4], o[:, 4], None, classes=np.zeros(len(o), int))
                    o = np.c_[o[dec.keep, :4], np.asarray(dec.scores, np.float64)].reshape(-1, 5)
                    tracker.det_thresh = float(min(dec.assoc, dec.birth))
                    tracker.iou_threshold = 1.0 - float(dec.match)
                    audit.append(dict(seq=seq, frame=fid, **{k: v for k, v in dec.log.items()
                                                              if isinstance(v, (int, float, str))}))
            targets = tracker.update(o.copy() if o is not None else None, (h, w), (h, w))
            if layer is not None and o is not None:
                layer.observe(targets[:, :4] if len(targets) else np.zeros((0, 4)),
                              targets[:, 4].astype(int) if len(targets) else [])
            tl, ids = [], []
            for t in targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                if tlwh[2] * tlwh[3] > args.min_box_area and not tlwh[2] / tlwh[3] > 1.6:
                    tl.append(tlwh)
                    ids.append(t[4])
            results.append((fid, tl, ids))
        with open(out_dir / f"{seq}.txt", "w") as f:
            for fid, tls, ids in results:
                for (x1, y1, ww, hh), tid in zip(tls, ids):
                    if tid < 0:
                        continue
                    f.write(f"{fid},{tid},{round(x1, 1)},{round(y1, 1)},{round(ww, 1)},{round(hh, 1)},-1,-1,-1,-1\n")
    json.dump(dict(args=run_args, audit=audit), open(out_dir.parent / "audit.json", "w"), default=str)
    print("wrote", a.name, run_args)


if __name__ == "__main__":
    main()
