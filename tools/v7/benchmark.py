"""
POST-FREEZE runtime benchmark: Baseline vs Baseline + frozen Universal AC-MOT
(V7f) on ONE fixed device, batch 1, sequential, LIVE detector.

Pipeline per frame: decode frame -> detector (native operating point: conf
floor 0.01, NMS 0.7, 736 px, fp32) -> [AC-MOT: image motion cue + layer.step
+ observe] -> tracker (ultralytics ByteTrack, host defaults). The two arms run
the same frames back to back; the first `--warmup` frames of every sequence
are excluded from the statistics.

  python tools/v7/benchmark.py --weights yolov8n.pt --seqs 0001 0009 0019 --out research/final/V7_REALTIME.json
Data: KITTI tracking training frames (the only frames reachable here).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(arm, frames, det, cfg, warmup):
    import cv2
    from acmot_v7 import HostContract, V7Layer, spec_from_dict
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from adapters.types import Detection
    from tools.v7.systems import HOST_BYTETRACK as H
    tr = ByteTrackAdapter(high=H["assoc"], low=H["low"], new=H["birth"], buffer=30, match=H["match"], fuse=True)
    layer = V7Layer(spec_from_dict(cfg["spec"]), HostContract(**H)) if arm == "v7" else None
    rec = []
    for i, fp in enumerate(frames):
        t0 = time.perf_counter()
        img = cv2.imread(str(fp))
        t1 = time.perf_counter()
        raw = det.detect(img, confidence=0.01, suppression=0.7, resolution=736)
        t2 = time.perf_counter()
        if layer is not None:
            m = layer.image_motion(img)
            b = np.array([[d.x1, d.y1, d.x2, d.y2] for d in raw]).reshape(-1, 4)
            s = np.array([d.confidence for d in raw])
            dec = layer.step(b, s, m, classes=[d.class_id for d in raw])
            dets = [Detection(x1=raw[k].x1, y1=raw[k].y1, x2=raw[k].x2, y2=raw[k].y2,
                              confidence=float(v), class_id=raw[k].class_id)
                    for k, v in zip(dec.keep, dec.scores)]
            tr.set_association_tolerance(dec.match)
            assoc, birth = dec.assoc, dec.birth
        else:
            dets, assoc, birth = raw, H["assoc"], H["birth"]
        t3 = time.perf_counter()
        tracks = tr.update(dets, img.shape[:2], association_threshold=assoc, birth_threshold=birth)
        t4 = time.perf_counter()
        if layer is not None:
            layer.observe([[t.x1, t.y1, t.x2, t.y2] for t in tracks], [t.track_id for t in tracks])
        t5 = time.perf_counter()
        if i >= warmup:
            rec.append(dict(read=t1 - t0, det=t2 - t1, ctrl=(t3 - t2) + (t5 - t4), trk=t4 - t3,
                            e2e=t5 - t0, pipe=t5 - t1))
    return rec


def stats(rec):
    out = {}
    for k in ("read", "det", "ctrl", "trk", "pipe", "e2e"):
        v = 1000 * np.array([r[k] for r in rec])
        out[k] = dict(mean_ms=float(v.mean()), p50_ms=float(np.percentile(v, 50)),
                      p95_ms=float(np.percentile(v, 95)))
    out["fps_pipe"] = 1000.0 / out["pipe"]["mean_ms"]
    out["fps_e2e"] = 1000.0 / out["e2e"]["mean_ms"]
    out["frames"] = len(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--view", default=str(Path.home() / "acmot_work/kitti/view/sequences"))
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from adapters.detectors.factory import create_detector
    cfg = json.loads((ROOT / "configs/universal_acmot_policy_v7.json").read_text())
    det = create_detector(a.weights, family="auto")
    res = dict(policy=cfg["system"], weights=Path(a.weights).name, seqs=a.seqs, warmup=a.warmup,
               hardware=dict(cpu=platform.processor() or platform.machine(), os=platform.platform(),
                             torch=torch.__version__, threads=torch.get_num_threads(),
                             cuda=torch.cuda.is_available(), device=str(getattr(det, "device", "?"))),
               precision="fp32", resolution=736, batch=1, arms={})
    # warm the page cache for every frame once, so that frame decoding is timed
    # under the same (warm) conditions in both arms
    for s in a.seqs:
        for fp in sorted((Path(a.view) / s).glob("*.jpg")):
            fp.read_bytes()
    res["page_cache"] = "pre-warmed (every frame read once before timing)"
    for arm in ("baseline", "v7"):
        rec = []
        for s in a.seqs:
            rec += run(arm, sorted((Path(a.view) / s).glob("*.jpg")), det, cfg, a.warmup)
        res["arms"][arm] = stats(rec)
    b, v = res["arms"]["baseline"], res["arms"]["v7"]
    res["overhead"] = dict(ctrl_mean_ms=v["ctrl"]["mean_ms"], ctrl_p95_ms=v["ctrl"]["p95_ms"],
                           pipe_delta_ms=v["pipe"]["mean_ms"] - b["pipe"]["mean_ms"],
                           pipe_delta_pct=100 * (v["pipe"]["mean_ms"] / b["pipe"]["mean_ms"] - 1))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
