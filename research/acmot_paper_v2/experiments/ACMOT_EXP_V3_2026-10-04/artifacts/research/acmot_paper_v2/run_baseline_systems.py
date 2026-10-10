"""
Baseline systems (no AC-MOT) for the OATrack-comparison four-system table:
  1. YOLO11m + Ultralytics ByteTrack
  2. YOLO11m + frozen OATrack (min_conf=0.40, unchanged)

Both use the SAME frozen public detector cache, at the single predeclared
default detector operating point (imgsz=1536, NMS IoU=0.70 -- "0.70 =
Ultralytics default", per notebooks/lightning/make_cache_script.py and
OATRACK_DESIGN.md). This is a fixed, predeclared choice, not a tuned one:
no AC-MOT controller is involved in these two systems, so there is nothing
to calibrate or tune here.

Reuses, unmodified:
  adapters/trackers/bytetrack.py (ByteTrackAdapter)
  adapters/trackers/oatrack.py   (OATrackAdapter, min_conf=0.40 default)
  tools/v6/eval_official.py      (official_sequence_stats / combine_official)
  tools/v7/bootstrap.py          (paired, for later cross-system deltas)

Systems 3/4 (AC-MOT + YOLO11m + {ByteTrack,OATrack}) are BLOCKED: no ready,
oracle-gated, detector-side AC-MOT controller exists for this host
(tools/g2/controllers.py::REGISTRY is empty; GAP_AUDIT.md X3/X4 gate not
run). Flagged, not worked around.

  python research/acmot_paper_v2/run_baseline_systems.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# tools/eval_local.py hardcodes a Mac-only TrackEval checkout path; this Studio copy is inserted
# into sys.path too so the (nonexistent, silently-skipped) Mac path still resolves to the real package.
sys.path.insert(0, "/teamspace/studios/this_studio/TrackEval")

from adapters.trackers.bytetrack import ByteTrackAdapter  # noqa: E402
from adapters.trackers.oatrack import OATrackAdapter  # noqa: E402
from adapters.types import Detection  # noqa: E402
from tools.v6.eval_official import official_sequence_stats, combine_official  # noqa: E402

BASE = "/teamspace/studios/this_studio"
VAL_ROOT = f"{BASE}/data/VisDrone2019-MOT-val"
CACHE_ROOT = f"{BASE}/outputs/acmot_oatrack/cache/yolo11m_vd/visdrone_val"
RES, NMS = 1536, 70  # predeclared default operating point (not tuned)

VAL_SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
            "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]

# eval5 cache class index -> VisDrone official raw class id (pedestrian=1,car=4,van=5,truck=6,bus=9)
EVAL5_TO_VISDRONE = {0: 1, 1: 4, 2: 5, 3: 6, 4: 9}
BOOTSTRAP_KEYS = ["MOTA", "HOTA", "IDF1", "IDS", "FP", "FN", "Precision", "Recall"]


def load_cache(seq: str):
    npz = np.load(f"{CACHE_ROOT}/r{RES}_n{NMS}/{seq}.npz")
    det = npz["det"]  # rows: [frame, x1, y1, x2, y2, score, eval5_class]
    shape = tuple(int(x) for x in npz["shape"])  # (H, W), as cached during detection
    return det, shape


def run_tracker(adapter_factory, det_rows: np.ndarray, frame_shape) -> np.ndarray:
    tracker = adapter_factory()
    out = []
    n_frames = int(det_rows[:, 0].max()) if len(det_rows) else 0
    for t in range(1, n_frames + 1):
        rows = det_rows[det_rows[:, 0] == t]
        dets = [Detection(x1=r[1], y1=r[2], x2=r[3], y2=r[4], confidence=r[5], class_id=int(r[6])) for r in rows]
        tracks = tracker.update(dets, frame_shape)  # positional: ByteTrackAdapter's kw is frame_shape, OATrackAdapter's is shape
        for tr in tracks:
            out.append([t, tr.track_id, tr.x1, tr.y1, tr.x2 - tr.x1, tr.y2 - tr.y1, tr.confidence, tr.class_id, -1, -1])
    return np.asarray(out, dtype=float).reshape(-1, 10) if out else np.zeros((0, 10))


def run_system(name: str, adapter_factory):
    per_seq = []
    t0 = time.time()
    for seq in VAL_SEQS:
        det_rows, frame_shape = load_cache(seq)
        tracks = run_tracker(adapter_factory, det_rows, frame_shape)
        stats = official_sequence_stats(VAL_ROOT, seq, tracks, class_map=EVAL5_TO_VISDRONE)
        per_seq.append(stats)
        print(name, seq, "tracks:", len(tracks), "dets:", len(det_rows))
    agg = combine_official(per_seq)
    agg["wall_seconds"] = time.time() - t0
    print(name, "AGGREGATE:", json.dumps(agg, indent=1))
    return per_seq, agg


def main():
    results = {}
    seq1, agg1 = run_system("YOLO11m+ByteTrack", lambda: ByteTrackAdapter())
    seq2, agg2 = run_system("YOLO11m+OATrack", lambda: OATrackAdapter())  # min_conf=0.40 default, untouched
    results["YOLO11m+ByteTrack"] = dict(aggregate=agg1, per_sequence=dict(zip(VAL_SEQS, seq1)))
    results["YOLO11m+OATrack"] = dict(aggregate=agg2, per_sequence=dict(zip(VAL_SEQS, seq2)))
    results["meta"] = dict(detector_sha256="c6f98247e7c731a567aaf37ca7b808beff588b687623e028b25a2e1d846e96e7",
                            resolution=RES, nms_iou=NMS / 100, class_map=EVAL5_TO_VISDRONE,
                            val_sequences=VAL_SEQS, protocol="official (tools/v6/eval_official.py)",
                            note="systems 3/4 (AC-MOT-controlled) not run: no ready detector-side "
                                 "AC-MOT controller exists for this host (see GAP_AUDIT.md)")
    out_path = f"{BASE}/outputs/acmot_oatrack/BASELINE_SYSTEMS_RESULT.json"
    json.dump(results, open(out_path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
