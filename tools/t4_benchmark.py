"""
Official end-to-end timing of the FROZEN pipeline (run on Colab T4 only;
Mac/MPS numbers are development-only and never reported).

Wraps the detector and tracker adapters in timing proxies (the locked
policy code is untouched) and times, per frame:
  detector   : DetectorAdapter.detect (includes pre/post-processing)
  tracker    : TrackerAdapter.update
  adaptive   : everything else inside UniversalACMOT (normalizer, gate,
               thresholds, bookkeeping) = total - detector - tracker
  total      : UniversalACMOT(frame) wall time (frame already decoded)
Reports mean / P95 per component, FPS = 1 / mean total, adaptive share,
and peak CUDA memory.

Example (Colab):
  python tools/t4_benchmark.py --dataset /content/VisDrone2019-MOT-val \
     --weights yolov8n.pt rtdetr-l.pt weights/fasterrcnn_...pth \
     --levels 640 736 832 --trackers bytetrack botsort --frames 300 \
     --out outputs/t4_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch


class Timed:
    def __init__(self, inner, method):
        self.inner, self.method, self.times = inner, method, []

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if name != self.method:
            return attr

        def wrapped(*a, **k):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = perf_counter()
            out = attr(*a, **k)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.times.append(perf_counter() - t)
            return out
        return wrapped


def stats(x):
    x = np.asarray(x) * 1000
    return dict(mean_ms=float(x.mean()), p95_ms=float(np.percentile(x, 95)),
                median_ms=float(np.median(x)))


def run(weights, tracker_name, level, frames, warmup):
    from adapters.detectors.factory import create_detector
    from adapters.trackers.botsort import BoTSORTAdapter
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from universal_acmot import UniversalACMOT

    det = Timed(create_detector(weights), "detect")
    cls = {"bytetrack": ByteTrackAdapter, "botsort": BoTSORTAdapter}[
        tracker_name]
    trk = Timed(cls(buffer=45, match=0.86), "update")
    system = UniversalACMOT(det, trk, resolution=level)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total = []
    for i, img in enumerate(frames):
        t = perf_counter()
        system(img)
        dt = perf_counter() - t
        if i >= warmup:
            total.append(dt)
    d, k = det.times[warmup:], trk.times[warmup:]
    adaptive = np.asarray(total) - np.asarray(d) - np.asarray(k)
    return dict(
        detector=Path(weights).name, tracker=tracker_name, level=level,
        frames=len(total), total=stats(total), detector_time=stats(d),
        tracker_time=stats(k), adaptive_time=stats(adaptive),
        fps=float(1.0 / np.mean(total)),
        adaptive_share_of_total=float(adaptive.mean() / np.mean(total)),
        peak_cuda_mem_mb=(torch.cuda.max_memory_allocated() / 2 ** 20
                          if torch.cuda.is_available() else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sequence", default="uav0000137_00458_v")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--levels", nargs="+", type=int, default=[640, 736, 832])
    ap.add_argument("--trackers", nargs="+", default=["bytetrack"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = sorted((Path(a.dataset) / "sequences" / a.sequence)
                   .glob("*.jpg"))[:a.frames + a.warmup]
    frames = [cv2.imread(str(p)) for p in paths]      # decode excluded
    env = dict(platform=platform.platform(), torch=torch.__version__,
               cuda=torch.cuda.is_available(),
               gpu=(torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
               git=subprocess.run(["git", "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip())
    if not env["gpu"] or "T4" not in env["gpu"]:
        print("WARNING: not a T4 — numbers are NOT official.")
    rows = []
    for w in a.weights:
        for tr in a.trackers:
            for lv in a.levels:
                r = run(w, tr, lv, frames, a.warmup)
                rows.append(r)
                print(f"{r['detector']:<45} {tr:<9} {lv} FPS {r['fps']:6.1f} "
                      f"total {r['total']['mean_ms']:6.1f}/"
                      f"{r['total']['p95_ms']:6.1f} ms  det "
                      f"{r['detector_time']['mean_ms']:6.1f}  trk "
                      f"{r['tracker_time']['mean_ms']:5.2f}  adaptive "
                      f"{r['adaptive_time']['mean_ms']:5.2f} ms "
                      f"({100 * r['adaptive_share_of_total']:.2f}%)",
                      flush=True)
    json.dump(dict(env=env, results=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
