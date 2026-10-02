"""
G2 S3 cache (development split val-7 only): per frame, detections of each
of the four tiles (adapters/detectors/tiling.py) at the reference input
setting, in frame coordinates, and full-frame detections at extra settings;
per-call detector latency.

Output: <out>/<det>/tiles<res>/<seq>.npz  det: N x 8 (frame, x1, y1, x2, y2,
        score, class, tile); timing json alongside
        <out>/<det>/<res>/<seq>.npz       (full frame, same format as the sweep cache)
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
from tools.sci_v7.build_sweep_cache import NATIVE_NMS, VAL7, make_detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--tile-res", type=int, default=0)
    ap.add_argument("--full-res", type=int, nargs="*", default=[])
    ap.add_argument("--floor", type=float, default=0.01)
    a = ap.parse_args()
    if any(s not in VAL7 for s in a.seqs):
        raise SystemExit("val-7 sequences only")
    import cv2
    from adapters.detectors.tiling import tile_boxes
    det = make_detector(a.weights)
    nms = NATIVE_NMS[a.det]
    for seq in a.seqs:
        frames = sorted((Path(a.dataset) / "sequences" / seq).glob("*.jpg"))
        jobs = ([("tiles", a.tile_res)] if a.tile_res else []) + [("full", r) for r in a.full_res]
        for kind, r in jobs:
            d = Path(a.out) / a.det / (f"tiles{r}" if kind == "tiles" else str(r))
            d.mkdir(parents=True, exist_ok=True)
            if (d / f"{seq}.npz").exists():
                continue
            rows, ms, shape = [], [], None
            for i, fp in enumerate(frames, start=1):
                img = cv2.imread(str(fp))
                shape = img.shape[:2]
                if kind == "full":
                    t0 = time.perf_counter()
                    out = det.detect(img, confidence=a.floor, suppression=nms, resolution=r)
                    ms.append(1000 * (time.perf_counter() - t0))
                    rows += [(i, x.x1, x.y1, x.x2, x.y2, x.confidence, x.class_id) for x in out]
                else:
                    for k, (x0, y0, x1, y1) in enumerate(tile_boxes(shape[1], shape[0])):
                        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
                        t0 = time.perf_counter()
                        out = det.detect(crop, confidence=a.floor, suppression=nms, resolution=r)
                        ms.append(1000 * (time.perf_counter() - t0))
                        rows += [(i, x.x1 + x0, x.y1 + y0, x.x2 + x0, x.y2 + y0, x.confidence, x.class_id, k)
                                 for x in out]
            w = 8 if kind == "tiles" else 7
            np.savez_compressed(d / f"{seq}.npz", det=np.asarray(rows, np.float64).reshape(-1, w),
                                shape=np.asarray(shape), frames=len(frames))
            (d / f"{seq}.timing.json").write_text(json.dumps(dict(det=a.det, kind=kind, res=r, seq=seq, ms=ms)))
            print(f"{a.det} {kind} {r} {seq}: mean {np.mean(ms):.1f} ms per call", flush=True)


if __name__ == "__main__":
    main()
