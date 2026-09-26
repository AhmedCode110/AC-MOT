"""
Non-adaptive baselines on raw detector scores (cache replay):

* default : detector at resolution 640 + ByteTrack with ultralytics
            defaults (high 0.25, low 0.1, new 0.25, buffer 30, match 0.8).
* oracle  : DETECTOR-SPECIFIC reference (NOT the universal method):
            per detector, sweep resolution x raw score threshold and pick
            the best setting on development data with the same selection
            rule as the universal policy (lexicographic: fewest (sequence)
            cells with MOTA < 0, then max ½(HOTA+IDF1)). Reported both
            in-sample (upper bound) and leave-one-sequence-out.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RESOLUTIONS = [640, 736, 832]
HIGHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def configs():
    cfgs = [dict(id="default", res=640, high=0.25, low=0.1, new=0.25,
                 buffer=30, match=0.8)]
    for r, h in itertools.product(RESOLUTIONS, HIGHS):
        cfgs.append(dict(id=f"oracle_r{r}_h{h:.1f}", res=r, high=h,
                         low=min(0.1, h / 2), new=h, buffer=30, match=0.8))
    return cfgs


def run_one(job):
    cfg, det, seq, cache_root, dataset, out_root = job
    target = Path(out_root) / cfg["id"] / det / f"{seq}.pkl"
    if target.exists():
        return
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from tools.run_policy_validation import CachedDetector
    from tools.seqstats import sequence_stats

    cd = CachedDetector(Path(cache_root) / det / f"{seq}.npz")
    tr = ByteTrackAdapter(high=cfg["high"], low=cfg["low"], new=cfg["new"],
                          buffer=cfg["buffer"], match=cfg["match"],
                          fuse=True)
    lines = []
    for i in range(1, cd.frames + 1):
        cd.frame = i
        dets = cd.detect(None, confidence=0.01, suppression=0.45,
                         resolution=cfg["res"])
        for t in tr.update(dets, cd.shape):
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
    ap.add_argument("--detectors", nargs="+", default=["yolov8", "rtdetr"])
    ap.add_argument("--only", nargs="*", help="config ids to run")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    seqs = sorted(p.name for p in Path(a.dataset, "sequences").iterdir()
                  if p.is_dir())
    cfgs = [c for c in configs() if not a.only or c["id"] in a.only]
    jobs = [(c, d, s, a.cache_root, a.dataset, a.out)
            for c in cfgs for d in a.detectors for s in seqs]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(cfgs, open(Path(a.out) / "configs.json", "w"), indent=1)
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(run_one, jobs, chunksize=1))
    print("done", len(jobs))


if __name__ == "__main__":
    main()
