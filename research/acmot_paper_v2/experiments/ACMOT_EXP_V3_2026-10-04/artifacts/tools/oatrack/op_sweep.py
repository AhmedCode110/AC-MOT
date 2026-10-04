"""
Step 4 (development, val-7, in-sample): does calibrating the operating
point close the gap between Ultralytics ByteTrack and OATrack, and does it
add to OATrack? Byte-identical cached detections.

Systems: 'bytetrack:<c>@<res>'  confidence filter c before ByteTrack (the
AC-MOT detector operating point; c=0 is the host alone), official tracker
settings; 'oatrack:<m>@<res>'  OATrack with min_conf m (0.40 = paper).

  python tools/oatrack/op_sweep.py <det> <out.json> <system> [<system> ...]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SCI_CACHE", "sweep")
from tools.g2 import dev  # noqa: E402  (val-7 guard)


def parse(system):
    name, rest = system.split(":")
    val, res = rest.split("@")
    return name, float(val), int(res)


def run(job):
    system, det, seq = job
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from adapters.trackers.oatrack import OATrackAdapter
    from adapters.types import Detection
    from tools.seqstats import sequence_stats
    import numpy as np
    name, val, res = parse(system)
    SPLITS, _ = dev._splits()
    z = np.load(Path(os.environ["SCI_SWEEP_ROOT"]) / det / str(res) / f"{seq}.npz")
    a, frames = z["det"], int(z["frames"])
    if name == "bytetrack":
        tr, floor = ByteTrackAdapter(0.25, 0.10, 0.25, 30, 0.80, True), val
    else:
        tr, floor = OATrackAdapter(min_conf=val), 0.0
    by = {int(k): a[a[:, 0] == k] for k in np.unique(a[:, 0])}
    lines = []
    for t in range(1, frames + 1):
        rows = by.get(t, ())
        dets = [Detection(x1=float(r[1]), y1=float(r[2]), x2=float(r[3]), y2=float(r[4]), confidence=float(r[5]),
                          class_id=int(r[6])) for r in rows if r[5] >= floor]
        for k in tr.update(dets, tuple(int(v) for v in z["shape"])):
            lines.append(f"{t},{k.track_id},{k.x1:.3f},{k.y1:.3f},{k.x2 - k.x1:.3f},{k.y2 - k.y1:.3f},"
                         f"{k.confidence:.6f},{k.class_id},-1,-1\n")
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.writelines(lines)
    f.close()
    st = sequence_stats(SPLITS[dev.SPLIT]["data"], seq, f.name)
    os.unlink(f.name)
    return system, seq, st


def main(det, out, *systems):
    _, seqs_of = dev._splits()
    seqs = seqs_of(dev.SPLIT)
    jobs = [(s, det, q) for s in systems for q in seqs]
    res = {}
    with ProcessPoolExecutor(int(os.environ.get("V7_WORKERS", 4))) as ex:
        for system, seq, st in ex.map(run, jobs, chunksize=1):
            res.setdefault(system, {})[seq] = st
    report = {}
    for s in systems:
        m = dev.combine([res[s][q] for q in seqs])
        report[s] = {k: float(m[k]) for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDS", "FP", "FN") if k in m}
        r = report[s]
        print(f"{det:7s} {s:22s} HOTA {r['HOTA']:6.2f} DetA {r['DetA']:6.2f} AssA {r['AssA']:6.2f} MOTA {r['MOTA']:7.2f} "
              f"IDF1 {r['IDF1']:6.2f} IDS {int(r['IDS']):5d} FP {int(r['FP']):6d} FN {int(r['FN']):6d}", flush=True)
    import pickle
    Path(out).write_text(json.dumps(report, indent=1))
    pickle.dump(res, open(str(out) + ".seq.pkl", "wb"))


if __name__ == "__main__":
    main(*sys.argv[1:])
