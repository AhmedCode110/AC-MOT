"""
Runtime of the SCI + V7f pipeline on real frames, live detector, batch 1,
sequential, one device. Stages per frame (milliseconds):

  decode    cv2.imread (reported separately, as in the V7 record)
  analyzer  image statistics of the scene layer (edges, gray level, blur on a
            1/4-scale frame: run_universal_acmot.analyze_visual)
  scene     SceneLayer.decide + observe (compute level)
  detector  forward pass + postprocessing at the chosen resolution
  score     V7f motion cue + step + observe
  tracker   ByteTrack update
On CUDA every stage boundary is synchronized (torch.cuda.synchronize).

Arms: 'native:<res>' (host alone), 'fixed:<res>' (V7f at a fixed
resolution = G1 at that setting) and 'sci:<profile key>' (scene layer + V7f). The first --warmup frames of every
sequence are excluded.

  python tools/sci_v7/benchmark.py --weights yolov8n.pt --det yolov8 \
      --dataset <VisDrone2019-MOT-val> --seqs uav0000086_00000_v uav0000339_00001_v \
      --arms fixed:736 sci:resolution --out bench.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_arm(arm, frames, det, det_name, warmup, torch):
    import cv2
    from acmot_sci import SceneLayer
    from acmot_v7 import HostContract, V7Layer, spec_from_dict
    from adapters.detectors.compute_profile import ComputeProfileAdapter
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from adapters.types import Detection
    from run_universal_acmot import analyze_visual
    from tools.sci_v7.build_sweep_cache import NATIVE_NMS
    from tools.v7.systems import HOST_BYTETRACK as H, SYSTEMS
    kind, arg = arm.split(":")
    scene = SceneLayer() if kind == "sci" else None
    adapter = ComputeProfileAdapter.from_config(det_name, key=arg) if kind == "sci" else None
    layer = None if kind == "native" else V7Layer(spec_from_dict(dict(SYSTEMS["V7f"], name="V7f")),
                                                  HostContract(**H))
    tr = ByteTrackAdapter(high=H["assoc"], low=H["low"], new=H["birth"], buffer=30, match=H["match"], fuse=True)
    nms = NATIVE_NMS[det_name]
    rec = []
    for i, fp in enumerate(frames, start=1):
        t0 = time.perf_counter()
        img = cv2.imread(str(fp))
        t1 = time.perf_counter()
        if scene is not None:
            st = analyze_visual(img)
            t2 = time.perf_counter()
            res = adapter.resolution(scene.decide(i, st).level)
        else:                              # 'fixed:<px>' (V7f) or 'native:<px>' (host alone)
            t2 = time.perf_counter()
            res = int(arg)
        t3 = time.perf_counter()
        raw = det.detect(img, confidence=0.01, suppression=nms, resolution=res)
        _sync(torch)
        t4 = time.perf_counter()
        if layer is not None:
            m = layer.image_motion(img)
            b = np.array([[d.x1, d.y1, d.x2, d.y2] for d in raw]).reshape(-1, 4)
            s = np.array([d.confidence for d in raw])
            dec = layer.step(b, s, m, classes=[d.class_id for d in raw])
            dets = [Detection(raw[k].x1, raw[k].y1, raw[k].x2, raw[k].y2, float(v), raw[k].class_id)
                    for k, v in zip(dec.keep, dec.scores)]
            tr.set_association_tolerance(dec.match)
            assoc, birth = dec.assoc, dec.birth
        else:
            dets, assoc, birth = raw, H["assoc"], H["birth"]
        t5 = time.perf_counter()
        tracks = tr.update(dets, img.shape[:2], association_threshold=assoc, birth_threshold=birth)
        t6 = time.perf_counter()
        boxes = [[t.x1, t.y1, t.x2, t.y2] for t in tracks]
        if layer is not None:
            layer.observe(boxes, [t.track_id for t in tracks])
        t7 = time.perf_counter()
        if scene is not None:
            scene.observe(boxes)
        t8 = time.perf_counter()
        if i > warmup:
            rec.append(dict(decode=t1 - t0, analyzer=t2 - t1, scene=(t3 - t2) + (t8 - t7),
                            detector=t4 - t3, score=(t5 - t4) + (t7 - t6), tracker=t6 - t5,
                            total=t8 - t1, res=res))
    return rec


def stats(rec):
    out = {}
    for k in ("decode", "analyzer", "scene", "detector", "score", "tracker", "total"):
        v = 1000 * np.array([r[k] for r in rec])
        out[k] = dict(mean_ms=float(v.mean()), p50_ms=float(np.percentile(v, 50)), p95_ms=float(np.percentile(v, 95)))
    out["controller_ms"] = dict(mean=out["analyzer"]["mean_ms"] + out["scene"]["mean_ms"] + out["score"]["mean_ms"])
    out["fps_excl_decode"] = 1000.0 / out["total"]["mean_ms"]
    out["mean_resolution"] = float(np.mean([r["res"] for r in rec]))
    out["frames"] = len(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=0, help="first N frames per sequence (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from tools.sci_v7.build_sweep_cache import make_detector
    det = make_detector(a.weights)
    dev = str(getattr(det, "device", "cpu"))
    hw = dict(cpu=platform.processor() or platform.machine(), os=platform.platform(), torch=torch.__version__,
              threads=torch.get_num_threads(), cuda=torch.cuda.is_available(),
              gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, device=dev)
    frames = {s: sorted((Path(a.dataset) / "sequences" / s).glob("*.jpg")) for s in a.seqs}
    if a.max_frames:
        frames = {s: f[:a.max_frames] for s, f in frames.items()}
    for fl in frames.values():          # warm page cache: decode timed under equal conditions
        for fp in fl:
            fp.read_bytes()
    res = dict(det=a.det, weights=Path(a.weights).name, seqs=a.seqs, warmup=a.warmup, hardware=hw,
               precision="fp32", batch=1, arms={})
    for arm in a.arms:
        rec = []
        for s in a.seqs:
            rec += run_arm(arm, frames[s], det, a.det, a.warmup, torch)
        res["arms"][arm] = stats(rec)
        print(arm, json.dumps(res["arms"][arm]["total"]), f"controller {res['arms'][arm]['controller_ms']['mean']:.2f} ms",
              flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
