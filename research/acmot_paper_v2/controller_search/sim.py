"""
Shared simulation harness for the detector-side AC-MOT controller search.
Calibration-only. Nothing here may read or be scored against VisDrone-val.

Causal pipeline per calibration sequence, reusing frozen, unchanged pieces:
  acmot_sci.SceneLayer         frame image stats (t) + reported boxes (<t) -> level
  run_universal_acmot.analyze_visual   frozen image-stats function
  adapters/trackers/{bytetrack,oatrack}  frozen tracker adapters
  tools/v6/eval_official        frozen official VisDrone evaluator

The only new, calibration-selected artifact is the LEVEL -> (resolution,
NMS) action mapping (confidence excluded; see CONFIDENCE_EXACTNESS_CHECK.json).
Detections for level L come directly from the real cached detections at
the resolution/NMS the action mapping names for L -- exact, not simulated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/teamspace/studios/this_studio/TrackEval")  # harmless no-op on the Mac

from acmot_sci import SceneLayer, SceneSpec  # noqa: E402
from run_universal_acmot import analyze_visual  # noqa: E402
from adapters.trackers.bytetrack import ByteTrackAdapter  # noqa: E402
from adapters.trackers.oatrack import OATrackAdapter  # noqa: E402
from adapters.types import Detection  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "research/acmot_paper_v2/cache/yolo11m_vd"
TRAIN_ROOT = Path("/Users/ahmedgouda/Downloads/VisDrone2019-MOT-train")
VAL_ROOT = Path("/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgoda1@gmail.com/My Drive/"
                 "AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-val")

RESOLUTIONS = [1088, 1280, 1536]
NMS_VALUES = [0.45, 0.60, 0.70]
LEVELS = ("LOW", "MEDIUM", "HIGH")

CALIBRATION = ["uav0000316_01288_v", "uav0000289_06922_v", "uav0000013_00000_v", "uav0000076_00720_v",
               "uav0000307_00000_v", "uav0000020_00406_v", "uav0000315_00000_v", "uav0000243_00001_v"]
VAL_SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
            "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]

EVAL5_TO_VISDRONE = {0: 1, 1: 4, 2: 5, 3: 6, 4: 9}


def _nms_tag(nms: float) -> str:
    return f"n{int(round(100 * nms))}"


class SequenceData:
    """Loads, once per sequence, everything a trial needs: per-(res,nms)
    detection rows keyed by frame, frozen image stats per frame, GT root."""

    def __init__(self, split: str, seq: str, data_root: Path):
        self.split = split
        self.seq = seq
        self.data_root = data_root
        frame_dir = data_root / "sequences" / seq
        self.frame_paths = sorted(frame_dir.glob("*.jpg"))
        self.n_frames = len(self.frame_paths)
        self.det = {}  # (res,nms) -> {frame:int -> rows (x1,y1,x2,y2,score,eval5_cls)}; lazy per (res,nms)
        self.shape = None
        self._stats_cache = {}

    def _load(self, res: int, nms: float):
        npz = np.load(CACHE_ROOT / self.split / f"r{res}_{_nms_tag(nms)}" / f"{self.seq}.npz")
        rows = npz["det"]
        by_frame = {}
        for t in range(1, self.n_frames + 1):
            by_frame[t] = rows[rows[:, 0] == t][:, 1:]
        self.det[(res, nms)] = by_frame
        if self.shape is None:
            self.shape = tuple(int(x) for x in npz["shape"])

    def image_stats(self, t: int) -> dict:
        if t not in self._stats_cache:
            img = cv2.imread(str(self.frame_paths[t - 1]))
            self._stats_cache[t] = analyze_visual(img)
        return self._stats_cache[t]

    def detections(self, t: int, res: int, nms: float):
        if (res, nms) not in self.det:
            self._load(res, nms)
        rows = self.det[(res, nms)][t]
        return [Detection(x1=r[0], y1=r[1], x2=r[2], y2=r[3], confidence=r[4], class_id=int(r[5])) for r in rows]


_SEQ_CACHE: dict[str, SequenceData] = {}


def get_sequence(split: str, seq: str) -> SequenceData:
    key = f"{split}/{seq}"
    if key not in _SEQ_CACHE:
        root = TRAIN_ROOT if split in ("visdrone_calib", "train") else VAL_ROOT
        _SEQ_CACHE[key] = SequenceData(split, seq, root)
    return _SEQ_CACHE[key]


def make_tracker(host: str):
    if host == "bytetrack":
        return ByteTrackAdapter()
    if host == "oatrack":
        return OATrackAdapter()  # min_conf=0.40, untouched
    raise ValueError(host)


def run_policy_on_sequence(split: str, seq: str, action_map: dict[str, tuple[int, float]], host: str,
                            scene_spec: SceneSpec = SceneSpec(), record_levels: bool = False):
    """action_map: {'LOW': (res, nms), 'MEDIUM': (...), 'HIGH': (...)}.
    Fully causal: level(t) uses image stats of t and tracker boxes of frames < t only."""
    sd = get_sequence(split, seq)
    scene = SceneLayer(scene_spec)
    tracker = make_tracker(host)
    out = []
    levels_used = []
    for t in range(1, sd.n_frames + 1):
        decision = scene.decide(t, sd.image_stats(t))
        res, nms = action_map[decision.level]
        if record_levels:
            levels_used.append(decision.level)
        dets = sd.detections(t, res, nms)
        tracks = tracker.update(dets, sd.shape)
        for tr in tracks:
            out.append([t, tr.track_id, tr.x1, tr.y1, tr.x2 - tr.x1, tr.y2 - tr.y1, tr.confidence, tr.class_id, -1, -1])
        scene.observe([[x.x1, x.y1, x.x2, x.y2] for x in tracks])
    tracks_arr = np.asarray(out, dtype=float).reshape(-1, 10) if out else np.zeros((0, 10))
    return tracks_arr, levels_used


def run_static_on_sequence(split: str, seq: str, res: int, nms: float, host: str):
    sd = get_sequence(split, seq)
    tracker = make_tracker(host)
    out = []
    for t in range(1, sd.n_frames + 1):
        dets = sd.detections(t, res, nms)
        tracks = tracker.update(dets, sd.shape)
        for tr in tracks:
            out.append([t, tr.track_id, tr.x1, tr.y1, tr.x2 - tr.x1, tr.y2 - tr.y1, tr.confidence, tr.class_id, -1, -1])
    return np.asarray(out, dtype=float).reshape(-1, 10) if out else np.zeros((0, 10))


def score_sequence(split: str, seq: str, tracks: np.ndarray):
    from tools.v6.eval_official import official_sequence_stats
    root = TRAIN_ROOT if split in ("visdrone_calib", "train") else VAL_ROOT
    return official_sequence_stats(root, seq, tracks, class_map=EVAL5_TO_VISDRONE)
