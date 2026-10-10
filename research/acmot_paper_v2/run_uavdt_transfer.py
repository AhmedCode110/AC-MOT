"""
UAVDT_TRANSFER: frozen, no-retuning cross-dataset transfer evaluation of
the v3 genuinely-adaptive AC-MOT controller (FREEZE_MANIFEST_V3_ADAPTIVE.json)
vs the predeclared host baseline r1536_n70, on the UAVDT test-20 split.

Uses the existing, pre-built UAVDT adapter (tools/build_uavdt_view.py's
view, class-agnostic single "vehicle" GT class=4) and its matching
class-agnostic scorer (tools/seqstats.py -- the one already designed for
this adapter's ignore-region rule). No new evaluator methodology
invented. Detector cache from the Kaggle-generated UAVDT_CACHE_STATS run
(3 action pairs only: the exact ones the frozen v3 policy and the
r1536_n70 baseline use).

No retuning: the v3 action mapping, thresholds, and cue subset are
identical to the frozen VisDrone policy. Vehicle classes (car/van/truck/
bus, eval5 indices 1-4) are kept; pedestrian (index 0, not present in
UAVDT) is dropped; surviving tracks are written with class=4 to match
the adapter's single placeholder GT class.
"""
from __future__ import annotations

import glob
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from acmot_sci import SceneLayer, SceneSpec  # noqa: E402
from adapters.trackers.bytetrack import ByteTrackAdapter  # noqa: E402
from adapters.trackers.oatrack import OATrackAdapter  # noqa: E402
from adapters.types import Detection  # noqa: E402
from tools.seqstats import sequence_stats, combine  # noqa: E402

CACHE_ROOT = ROOT / "research/acmot_paper_v2/cache/uavdt"
VIEW = ROOT / "outputs/uavdt_view"
SEQS = "M0203 M0205 M0208 M0209 M0403 M0601 M0602 M0606 M0701 M0801 M0802 M1001 M1004 M1007 M1009 M1101 M1301 M1302 M1303 M1401".split()

V3_ACTION_MAP = {"LOW": (1536, 0.70, 0.10), "MEDIUM": (1280, 0.45, 0.40), "HIGH": (1088, 0.60, 0.40)}
V3_SPEC = SceneSpec(t_med=0.29747709701135764, t_high=0.5114928923560124,
                     w_crowd=1.0, w_tiny=0.0, w_edge=0.0, w_dark=0.0, w_blur=0.0)
BASELINE_RES_NMS = (1536, 0.70)
BASELINE_CONF = 0.0  # cache floor (0.01), no extra filter -- matches systems 1/2's own definition

VEHICLE_EVAL5 = {1, 2, 3, 4}  # car, van, truck, bus (drop 0=pedestrian, not present in UAVDT)
UAVDT_GT_CLASS = 4  # the adapter's single placeholder class


class SeqCache:
    def __init__(self, seq):
        self.seq = seq
        self.det = {}
        n_imgs = len(glob.glob(str(VIEW / "sequences" / seq / "*.jpg")))
        self.n_frames = n_imgs
        self._stats_cache = {}
        self.shape = None

    def _load(self, res, nms):
        path = CACHE_ROOT / f"r{res}_n{int(round(100 * nms))}" / f"{self.seq}.npz"
        npz = np.load(path)
        rows = npz["det"]
        by_frame = {}
        for t in range(1, self.n_frames + 1):
            by_frame[t] = rows[rows[:, 0] == t][:, 1:]
        self.det[(res, nms)] = by_frame
        if self.shape is None:
            self.shape = tuple(int(x) for x in npz["shape"])

    def detections(self, t, res, nms, conf_floor=0.0):
        if (res, nms) not in self.det:
            self._load(res, nms)
        rows = self.det[(res, nms)][t]
        if conf_floor > 0.0:
            rows = rows[rows[:, 4] >= conf_floor]
        dets = []
        for r in rows:
            cls = int(r[5])
            if cls not in VEHICLE_EVAL5:
                continue
            dets.append(Detection(x1=r[0], y1=r[1], x2=r[2], y2=r[3], confidence=r[4], class_id=cls))
        return dets

    def image_stats(self, t):
        if t not in self._stats_cache:
            import cv2
            img_path = sorted(glob.glob(str(VIEW / "sequences" / self.seq / "*.jpg")))[t - 1]
            img = cv2.imread(img_path)
            small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            self._stats_cache[t] = dict(edges=float(cv2.Canny(gray, 50, 120).mean() / 255.0),
                                         brightness=float(gray.mean()),
                                         blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        return self._stats_cache[t]


def make_tracker(host):
    return ByteTrackAdapter() if host == "bytetrack" else OATrackAdapter()


def run_v3(seq, host):
    sd = SeqCache(seq)
    scene = SceneLayer(V3_SPEC)
    tracker = make_tracker(host)
    out = []
    for t in range(1, sd.n_frames + 1):
        decision = scene.decide(t, sd.image_stats(t))
        res, nms, conf = V3_ACTION_MAP[decision.level]
        dets = sd.detections(t, res, nms, conf)
        tracks = tracker.update(dets, sd.shape)
        for tr in tracks:
            out.append([t, tr.track_id, tr.x1, tr.y1, tr.x2 - tr.x1, tr.y2 - tr.y1, 1, UAVDT_GT_CLASS, 0, 0])
        scene.observe([[x.x1, x.y1, x.x2, x.y2] for x in tracks])
    return np.asarray(out, dtype=float).reshape(-1, 10) if out else np.zeros((0, 10))


def run_baseline(seq, host):
    sd = SeqCache(seq)
    tracker = make_tracker(host)
    out = []
    for t in range(1, sd.n_frames + 1):
        dets = sd.detections(t, *BASELINE_RES_NMS, BASELINE_CONF)
        tracks = tracker.update(dets, sd.shape)
        for tr in tracks:
            out.append([t, tr.track_id, tr.x1, tr.y1, tr.x2 - tr.x1, tr.y2 - tr.y1, 1, UAVDT_GT_CLASS, 0, 0])
    return np.asarray(out, dtype=float).reshape(-1, 10) if out else np.zeros((0, 10))


def score(seq, tracks_arr):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        if len(tracks_arr):
            np.savetxt(f.name, tracks_arr, delimiter=",", fmt="%g")
        tmp_path = f.name
    try:
        return sequence_stats(VIEW, seq, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_system(name, run_fn, host):
    per_seq = {}
    t0 = time.time()
    for seq in SEQS:
        tracks = run_fn(seq, host)
        per_seq[seq] = score(seq, tracks)
        print("UAVDT_TRANSFER", name, host, seq, "tracks:", len(tracks), "elapsed", round(time.time() - t0, 1))
    agg = combine(list(per_seq.values()))
    agg["wall_seconds"] = time.time() - t0
    print("UAVDT_TRANSFER", name, host, "AGGREGATE:", json.dumps(agg, indent=1))
    return per_seq, agg


def main():
    results = {"_label": "UAVDT_TRANSFER", "_note": "frozen, no-retuning transfer evaluation; "
               "class-agnostic (single vehicle class) internal protocol via tools/seqstats.py, "
               "matching the pre-existing UAVDT adapter (tools/build_uavdt_view.py)."}
    for host in ("bytetrack", "oatrack"):
        seq_base, agg_base = run_system("baseline_r1536_n70", run_baseline, host)
        seq_v3, agg_v3 = run_system("v3_adaptive", run_v3, host)
        results[f"baseline_r1536_n70+{host}"] = dict(aggregate=agg_base,
                                                       per_sequence={s: v["counts"] for s, v in seq_base.items()})
        results[f"v3_adaptive+{host}"] = dict(aggregate=agg_v3,
                                               per_sequence={s: v["counts"] for s, v in seq_v3.items()})
    out_path = ROOT / "research/acmot_paper_v2/UAVDT_TRANSFER_RESULT.json"
    json.dump(results, open(out_path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
