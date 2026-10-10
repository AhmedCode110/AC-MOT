"""
Resolution-sweep detection cache for the General-SCI cycle (development
split val-7 only), with per-resolution detector latency.

Recipe = the native-cache recipe of the V7 record (tools/cache_detections.py,
tools/mac_cache_queue_v5tf.sh): confidence floor 0.01, detector-native
suppression (YOLO NMS 0.7, RT-DETR none, Faster R-CNN 0.5), COCO person / car
/ bus / truck. One device for every resolution, so the accuracy and runtime
curves are not confounded by the device that built the cache.

Output: <out>/<det>/<res>/<seq>.npz  (det = N x 7: frame, x1, y1, x2, y2,
score, class), <out>/<det>/<res>/<seq>.timing.json (per-frame detector
milliseconds: forward pass + postprocessing, batch 1; decode excluded).

  python tools/sci_v7/build_sweep_cache.py --weights yolov8n.pt --det yolov8 \
      --dataset <VisDrone2019-MOT-val> --out <dir> --res 512 640 736
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NATIVE_NMS = {"yolov8": 0.7, "rtdetr": None, "fasterrcnn": 0.5, "retinanet": None}
VAL7 = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
        "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]


def make_detector(weights):
    """Detector adapter for a weights file. adapters/detectors/factory.py is
    part of the V6 policy lock, so detectors added in this cycle are created
    here."""
    if "retinanet" in Path(weights).name.lower():
        from adapters.detectors.retinanet import RetinaNetAdapter
        return RetinaNetAdapter(weights)
    from adapters.detectors.factory import create_detector
    return create_detector(weights, family="auto")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--det", required=True, choices=sorted(NATIVE_NMS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, nargs="+", required=True)
    ap.add_argument("--seqs", nargs="*", default=VAL7)
    ap.add_argument("--floor", type=float, default=0.01)
    a = ap.parse_args()
    if any(s not in VAL7 for s in a.seqs):
        raise SystemExit("val-7 sequences only")
    import cv2
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    det = make_detector(a.weights)
    nms = NATIVE_NMS[a.det]
    for seq in a.seqs:
        frames = sorted((Path(a.dataset) / "sequences" / seq).glob("*.jpg"))
        for r in a.res:
            d = Path(a.out) / a.det / str(r)
            d.mkdir(parents=True, exist_ok=True)
            if (d / f"{seq}.npz").exists():
                continue
            rows, ms, shape = [], [], None
            for i, fp in enumerate(frames, start=1):
                img = cv2.imread(str(fp))
                shape = img.shape[:2]
                t0 = time.perf_counter()
                out = det.detect(img, confidence=a.floor, suppression=nms, resolution=r)
                ms.append(1000 * (time.perf_counter() - t0))
                rows += [(i, x.x1, x.y1, x.x2, x.y2, x.confidence, x.class_id) for x in out]
            np.savez_compressed(d / f"{seq}.npz", det=np.asarray(rows, np.float64).reshape(-1, 7),
                                shape=np.asarray(shape), frames=len(frames))
            (d / f"{seq}.timing.json").write_text(json.dumps(dict(
                det=a.det, res=r, seq=seq, frames=len(frames), threads=torch.get_num_threads(),
                device=str(getattr(det, "device", "cpu")), ms=ms)))
            print(f"{a.det} {r} {seq}: {len(frames)} frames, mean {np.mean(ms):.1f} ms, "
                  f"p95 {np.percentile(ms, 95):.1f} ms", flush=True)


if __name__ == "__main__":
    main()
