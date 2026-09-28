"""
POST-FREEZE recent external system: TOPICTrack (Cao et al., "TOPIC: A Parallel
Association Paradigm for Multi-Object Tracking Under Complex Motions and
Diverse Scenes", IEEE Transactions on Image Processing 2025;
github.com/holmescao/TOPICTrack @ e7b260f).

The loop of main.py::main is reproduced (official detector with its own
cache, official loader and preprocessing, OCSort(**oc_sort_args) re-created at
frame 1, `update(pred, img, np_img, tag, AARM, TOPIC)`, utils.filter_targets,
write_results_no_score, utils.dti for `_post`), with the official arguments of
run/mot17_val.sh / run/mot20_val.sh.

Compatibility (no GPU on the runner): `.cuda()` is a no-op, `.half()` keeps
float32 and `torch.load` maps to CPU, for the detector and the ReID model alike (the authors ran CUDA fp16).
Both arms use the same detector cache.

V7f arm: the detector output of the frame (N x 5, network scale) is mapped to
original coordinates exactly as `OCSort.extract_detections` does, passed
through the frozen layer with the frame's image cue, and the kept candidates
with the layer's scores are handed to `update` in the network scale.
`det_thresh <- decision.assoc`, `iou_threshold <- 1 - decision.match`
(V7_RECENT_EXTERNAL_PROTOCOL.md section 5: assoc 0.6, birth 0.4, low 0.4,
match 0.8; the 0.4 band is hard-coded in the tracker and is not changed).
The V7f arm keeps its own ReID-embedding cache.

  python topictrack_v7.py --dataset mot17|mot20 --system BASELINE|V7f --name N [--seq MOT17-02-FRCNN]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ACMOT = Path(os.environ.get("ACMOT_ROOT", Path(__file__).resolve().parents[3]))
EXT = Path(os.environ.get("ACMOT_EXT", Path.home() / "acmot_work/acmot_external"))
TOPIC = Path(os.environ.get("TOPIC_ROOT", EXT / "TOPICTrack"))
OFFICIAL = {  # run/mot17_val.sh, run/mot20_val.sh
    "mot17": ["--alpha_gate", "0.3", "--gate", "0.3", "--aspect_ratio_thresh", "1.6",
              "--w_assoc_emb", "0.75", "--AARM", "--TOPIC", "--dataset", "mot17"],
    "mot20": ["--alpha_gate", "0", "--gate", "0.3", "--aspect_ratio_thresh", "1.6",
              "--w_assoc_emb", "0.75", "--AARM", "--TOPIC", "--dataset", "mot20"],
}


def cpu_shims():
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.half = lambda self, *a, **k: self
    torch.nn.Module.half = lambda self, *a, **k: self
    load = torch.load

    def cpu_load(*a, **k):
        k.setdefault("map_location", "cpu")
        k.setdefault("weights_only", False)
        return load(*a, **k)
    torch.load = cpu_load
    torch.set_num_threads(os.cpu_count() or 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(OFFICIAL))
    ap.add_argument("--system", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seq", default=None)
    ap.add_argument("--root", default=str(EXT / "runs/topictrack"))
    a = ap.parse_args()
    baseline = a.system == "BASELINE"
    out_root = Path(a.root).resolve()
    cpu_shims()
    os.chdir(TOPIC)
    for p in [TOPIC / "external/deep-person-reid", TOPIC / "external", TOPIC / "external/YOLOX", TOPIC]:
        sys.path.insert(0, str(p))
    sys.argv = ["main.py", "--exp_name", a.name] + OFFICIAL[a.dataset]
    from fast_reid.fastreid.config import defaults as reid_defaults
    reid_defaults._C.MODEL.DEVICE = "cpu"      # compatibility: FastReID builds on cfg.MODEL.DEVICE ("cuda")
    import main as tmain
    import dataset
    import utils
    from external.adaptors import detector
    from trackers import ocsort_embedding as tracker_module
    args = tmain.get_main_args()
    if not baseline:
        sys.path.insert(0, str(ACMOT))
        from acmot_v7 import HostContract, V7Layer, spec_from_dict
        cfg = json.loads((ACMOT / "configs/universal_acmot_policy_v7.json").read_text())
        assert a.system == cfg["system"], "post-freeze: only the frozen policy may run"
        spec = spec_from_dict(cfg["spec"])
        host = HostContract(assoc=args.track_thresh, birth=0.4, low=0.4, match=1.0 - args.iou_thresh)
    if args.dataset == "mot17":
        detector_path, size = "external/weights/topictrack_ablation.pth.tar", (800, 1440)
    else:
        detector_path, size = "external/weights/topictrack_mot17.pth.tar", (800, 1440)
    det = detector.Detector("yolox", detector_path, args.dataset)
    loader = dataset.get_mot_loader(args.dataset, args.test_dataset, size=size, workers=2)
    oc_sort_args = dict(args=args, det_thresh=args.track_thresh, alpha_gate=args.alpha_gate, gate=args.gate,
                        gate2=args.gate2, iou_threshold=args.iou_thresh, asso_func=args.asso,
                        delta_t=args.deltat, inertia=args.inertia, w_association_emb=args.w_assoc_emb,
                        new_kf_off=args.new_kf_off)
    emb_cache = "./cache/embeddings/{}_embedding.pkl" if baseline else "./cache/embeddings_v7/{}_embedding.pkl"
    os.makedirs(os.path.dirname(emb_cache), exist_ok=True)

    def new_tracker():
        t = tracker_module.ocsort.OCSort(**oc_sort_args)
        t.embedder.cache_path = emb_cache
        return t

    tracker = new_tracker()
    layer = None
    results, audit = {}, []
    for (img, np_img), label, info, idx in loader:
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]
        if a.seq is not None and video_name != a.seq:
            continue
        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []
        if frame_id == 1:
            tracker.dump_cache()
            tracker = new_tracker()
            layer = None if baseline else V7Layer(spec, host)
        pred = det(img, tag)
        if pred is None:
            continue
        if layer is not None:
            p = pred.cpu().numpy().astype(np.float64)
            raw = np_img[0].numpy()
            scale = min(img.shape[2] / raw.shape[0], img.shape[3] / raw.shape[1])
            dec = layer.step(p[:, :4] / scale, p[:, 4], layer.image_motion(raw),
                             classes=np.zeros(len(p), int))
            keep = np.asarray(dec.keep, int)
            tracker.det_thresh = float(dec.assoc)
            tracker.iou_threshold = 1.0 - float(dec.match)
            audit.append(dict(seq=video_name, frame=frame_id,
                              **{k: v for k, v in dec.log.items() if isinstance(v, (int, float, str))}))
            if len(keep) == 0:            # as the official loop does when the detector returns nothing
                layer.observe(np.zeros((0, 4)), [])
                continue
            pred = torch.from_numpy(np.c_[p[keep, :4], np.asarray(dec.scores, np.float64)].astype(np.float32))
        targets = tracker.update(pred, img, np_img[0].numpy(), tag, args.AARM, args.TOPIC)
        if layer is not None:
            layer.observe(targets[:, :4] if len(targets) else np.zeros((0, 4)),
                          targets[:, 4].astype(int) if len(targets) else [])
        tlwhs, ids = utils.filter_targets(targets, args.aspect_ratio_thresh, args.min_box_area, args.dataset)
        results[video_name].append((frame_id, tlwhs, ids))
    det.dump_cache()
    tracker.dump_cache()
    split = "MOT17-val" if args.dataset == "mot17" else "MOT20-val"
    folder = out_root / split / a.name / "data"
    folder.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        utils.write_results_no_score(str(folder / f"{name}.txt"), res)
    post = out_root / split / (a.name + "_post")
    post.mkdir(parents=True, exist_ok=True)
    (post / "data").mkdir(exist_ok=True)
    for name in results:
        shutil.copy(folder / f"{name}.txt", post / "data" / f"{name}.txt")
    utils.dti(str(post / "data"), str(post / "data"))
    if audit:
        json.dump(audit, open(folder.parent / f"audit_{a.seq or 'all'}.json", "w"))
    print("wrote", folder, list(results))


if __name__ == "__main__":
    main()
