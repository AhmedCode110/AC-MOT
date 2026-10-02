"""Detector latency on ONE host for the G2 compute axis: full-frame calls at
every setting and one tile call (adapters/detectors/tiling.py) at the
reference setting; real val-7 frames, batch 1, first frames as warm-up.

  python tools/g2/latency.py --weights w --det d --dataset D --seq S --frames 60 \
      --full 512 576 ... --tile 736 --out latency_<det>.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--full", type=int, nargs="+", required=True)
    ap.add_argument("--tile", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cv2
    import torch
    from adapters.detectors.tiling import tile_boxes
    from tools.sci_v7.build_sweep_cache import NATIVE_NMS, make_detector
    det = make_detector(a.weights)
    nms = NATIVE_NMS[a.det]
    imgs = [cv2.imread(str(p)) for p in sorted((Path(a.dataset) / "sequences" / a.seq).glob("*.jpg"))[:a.frames]]

    def timed(fn):
        ms = []
        for i, im in enumerate(imgs):
            t0 = time.perf_counter()
            fn(im)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if i >= a.warmup:
                ms.append(1000 * (time.perf_counter() - t0))
        return dict(mean_ms=float(np.mean(ms)), p95_ms=float(np.percentile(ms, 95)), n=len(ms))
    res = dict(det=a.det, seq=a.seq, host=dict(cpu=platform.processor() or platform.machine(),
               threads=torch.get_num_threads(), cuda=torch.cuda.is_available()), full={}, tile={})
    for r in a.full:
        res["full"][str(r)] = timed(lambda im: det.detect(im, 0.01, nms, r))
    tb = tile_boxes(imgs[0].shape[1], imgs[0].shape[0])
    res["tile"][str(a.tile)] = timed(lambda im: det.detect(np.ascontiguousarray(im[tb[0][1]:tb[0][3], tb[0][0]:tb[0][2]]),
                                                         0.01, nms, a.tile))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
