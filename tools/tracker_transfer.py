"""
Tracker-portability test: the FROZEN policy (configs/universal_acmot_policy.json)
driving a different tracker through the generic TrackerAdapter interface,
with no policy change. Baselines on the same tracker: default thresholds
(res 640) and shared-static (raw 0.5, res 832).

Detections come from the verified cache; frames are loaded because the
tracker's camera-motion compensation needs them.
"""
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SYSTEMS = ["universal", "default", "shared_static"]
STATIC = {"shared_static": dict(res=832, high=0.5, low=0.1, new=0.5),
          "default": dict(res=640, high=0.25, low=0.1, new=0.25)}


def run_one(job):
    system, tracker_name, det, seq, dataset, cache_root, out = job
    target = Path(out) / tracker_name / system / det / f"{seq}.pkl"
    if target.exists():
        return
    from adapters.trackers.botsort import BoTSORTAdapter
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from run_universal_acmot import build_config
    from tools.run_policy_validation import CachedDetector
    from tools.seqstats import sequence_stats
    from universal_acmot import load_policy
    from universal_policy_pipeline import UniversalPolicyPipeline

    cls = {"botsort": BoTSORTAdapter, "bytetrack": ByteTrackAdapter}[
        tracker_name]
    cd = CachedDetector(Path(cache_root) / det / f"{seq}.npz")
    frames = sorted((Path(dataset) / "sequences" / seq).glob("*.jpg"))
    lines = []
    if system == "universal":
        cfg = build_config()
        policy, dk = load_policy()
        tr = cls(high=cfg.high, low=cfg.low, new=cfg.new, buffer=cfg.buffer,
                 match=cfg.match, fuse=cfg.fuse)
        pipe = UniversalPolicyPipeline(cfg, cd, tr, policy,
                                       density_kwargs=dk)
    else:
        c = STATIC[system]
        tr = cls(high=c["high"], low=c["low"], new=c["new"], buffer=30,
                 match=0.8)
    for i, fp in enumerate(frames, start=1):
        cd.frame = i
        img = cv2.imread(str(fp))
        if system == "universal":
            v = cd.visual[i - 1]
            tracks = pipe.process(i, img, dict(edges=v[0], brightness=v[1],
                                               blur=v[2]))["tracks"]
        else:
            dets = cd.detect(None, 0.01, 0.45, STATIC[system]["res"])
            tracks = tr.update(dets, img.shape[:2], image=img)
        for t in tracks:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},"
                         f"{max(0.0, t.x2 - t.x1):.3f},"
                         f"{max(0.0, t.y2 - t.y1):.3f},"
                         f"{t.confidence:.6f},{t.class_id},-1,-1\n")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.writelines(lines)
    st = sequence_stats(dataset, seq, f.name)
    os.unlink(f.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(target, "wb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trackers", nargs="+", default=["botsort"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    seqs = sorted(p.name for p in Path(a.dataset, "sequences").iterdir()
                  if p.is_dir())
    jobs = [(s, t, d, q, a.dataset, a.cache_root, a.out) for s, t, d, q in
            itertools.product(SYSTEMS, a.trackers, ["yolov8", "rtdetr"],
                              seqs)]
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(run_one, jobs, chunksize=1))
    from tools.seqstats import combine
    for t, s in itertools.product(a.trackers, SYSTEMS):
        row = f"{t:<9}{s:<14}"
        for d in ["yolov8", "rtdetr"]:
            st = [pickle.load(open(Path(a.out) / t / s / d / f"{q}.pkl",
                                   "rb")) for q in seqs]
            m = combine(st)
            ncat = sum(combine([x])["MOTA"] < 0 for x in st)
            row += (f"| {d} MOTA {m['MOTA']:6.2f} HOTA {m['HOTA']:6.2f} "
                    f"IDF1 {m['IDF1']:6.2f} IDS {m['IDS']:5d} ncat {ncat} ")
        print(row, flush=True)


if __name__ == "__main__":
    main()
