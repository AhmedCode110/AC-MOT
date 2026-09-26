"""
Score-calibration stress test (declared before running, val only).

Question: when a detector's confidence calibration changes (a new detector
family with a different score distribution), which FROZEN system keeps
working without re-tuning?

Systems (all frozen, none re-tuned):
  universal     : configs/universal_acmot_policy.json
  shared_static : raw threshold 0.5, res 832, ByteTrack (the best single
                  shared raw setting on val, same selection protocol)
  default       : raw, res 640, ByteTrack defaults
Transforms: temp2, temp05, scale05, pow3 (see CachedDetector.TRANSFORMS)
"""
from __future__ import annotations

import itertools
import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATASET = "/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val"
OUT = Path("outputs/stress")
DETS = ["yolov8", "rtdetr"]
TRANSFORMS = [None, "temp2", "temp05", "scale05", "pow3"]
SYSTEMS = ["universal", "shared_static", "default"]
STATIC = {"shared_static": dict(res=832, high=0.5, low=0.1, new=0.5,
                                buffer=30, match=0.8),
          "default": dict(res=640, high=0.25, low=0.1, new=0.25, buffer=30,
                          match=0.8)}


def run_one(job):
    system, tf, det, seq = job
    target = OUT / system / str(tf) / det / f"{seq}.pkl"
    if target.exists():
        return
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from tools.run_policy_validation import CachedDetector, make_tracker
    from tools.seqstats import sequence_stats

    cd = CachedDetector(f"outputs/det_cache/{det}/{seq}.npz", transform=tf)
    lines = []

    def emit(i, tracks):
        for t in tracks:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},"
                         f"{max(0.0, t.x2 - t.x1):.3f},"
                         f"{max(0.0, t.y2 - t.y1):.3f},"
                         f"{t.confidence:.6f},{t.class_id},-1,-1\n")

    if system == "universal":
        from run_universal_acmot import build_config
        from universal_acmot import load_policy
        from universal_policy_pipeline import UniversalPolicyPipeline
        policy, dk = load_policy()
        cfg = build_config()
        pipe = UniversalPolicyPipeline(cfg, cd, make_tracker(cfg), policy,
                                       density_kwargs=dk)
        img = np.empty(cd.shape + (0,), np.uint8)
        for i in range(1, cd.frames + 1):
            cd.frame = i
            v = cd.visual[i - 1]
            emit(i, pipe.process(i, img, dict(edges=v[0], brightness=v[1],
                                              blur=v[2]))["tracks"])
    else:
        c = STATIC[system]
        tr = ByteTrackAdapter(high=c["high"], low=c["low"], new=c["new"],
                              buffer=c["buffer"], match=c["match"])
        for i in range(1, cd.frames + 1):
            cd.frame = i
            emit(i, tr.update(cd.detect(None, 0.01, 0.45, c["res"]),
                              cd.shape))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.writelines(lines)
    st = sequence_stats(DATASET, seq, f.name)
    os.unlink(f.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(target, "wb"))


def report():
    from tools.seqstats import combine
    seqs = sorted(p.name for p in Path(DATASET, "sequences").iterdir())
    print(f"{'transform':<9}{'system':<15}" + "".join(
        f"| {d}: MOTA   HOTA   IDF1  ncat " for d in DETS))
    for tf in TRANSFORMS:
        for sy in SYSTEMS:
            row = f"{str(tf):<9}{sy:<15}"
            for d in DETS:
                st = [pickle.load(open(OUT / sy / str(tf) / d / f"{s}.pkl",
                                       "rb")) for s in seqs]
                m = combine(st)
                ncat = sum(combine([x])["MOTA"] < 0 for x in st)
                row += (f"| {m['MOTA']:7.2f} {m['HOTA']:6.2f} "
                        f"{m['IDF1']:6.2f} {ncat:4d} ")
            print(row)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        seqs = sorted(p.name for p in Path(DATASET, "sequences").iterdir())
        jobs = list(itertools.product(SYSTEMS, TRANSFORMS, DETS, seqs))
        with ProcessPoolExecutor(8) as ex:
            list(ex.map(run_one, jobs, chunksize=1))
        report()
