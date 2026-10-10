"""
Mechanism check of the OATrack re-implementation on cached detections
(development split val-7 only): Ultralytics ByteTrack (official settings
0.25/0.10/0.25/30/0.80, score fusion) vs OATrack on byte-identical
detections, internal protocol plus HOTA/DetA/AssA.

  python tools/oatrack/val_check.py <det> <resolution> [out.json]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SCI_CACHE", "sweep")
from tools.g2 import dev  # noqa: E402  (val-7 guard)


def run(tracker_name, det, res, seq):
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from adapters.trackers.oatrack import OATrackAdapter
    from adapters.types import Detection
    from tools.sci_v7.sweep_cache import SweepCache
    SPLITS, _ = dev._splits()
    nat = SPLITS[dev.SPLIT]["native"]
    cd = SweepCache(os.environ["SCI_SWEEP_ROOT"], det, seq, f"{nat}/visual_cues/{seq}.npz")
    tr = (ByteTrackAdapter(0.25, 0.10, 0.25, 30, 0.80, True) if tracker_name == "bytetrack" else OATrackAdapter())
    lines = []
    rows_all = cd.by_res[res]
    for t in range(1, cd.frames + 1):
        rows = rows_all.get(t, ())
        dets = [Detection(x1=float(r[1]), y1=float(r[2]), x2=float(r[3]), y2=float(r[4]), confidence=float(r[5]),
                          class_id=int(r[6])) for r in rows]
        for k in tr.update(dets, cd.shape):
            lines.append(f"{t},{k.track_id},{k.x1:.3f},{k.y1:.3f},{k.x2 - k.x1:.3f},{k.y2 - k.y1:.3f},"
                         f"{k.confidence:.6f},{k.class_id},-1,-1\n")
    from tools.seqstats import sequence_stats
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.writelines(lines)
    f.close()
    st = sequence_stats(SPLITS[dev.SPLIT]["data"], seq, f.name)
    os.unlink(f.name)
    st["tracks_txt"] = "".join(lines)
    return st


def main(det, res, out=None):
    _, seqs_of = dev._splits()
    res = int(res)
    report = {}
    for name in ("bytetrack", "oatrack"):
        sts = [run(name, det, res, s) for s in seqs_of(dev.SPLIT)]
        m = dev.combine(sts)
        report[name] = {k: m[k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDS", "FP", "FN") if k in m}
        print(name, json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in report[name].items()}))
    if out:
        Path(out).write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])
