"""SCI + V7f layer: separation, historical equivalence, causality, guards,
frozen-V7f integrity."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acmot_sci import LEVELS, SceneLayer, SceneSpec  # noqa: E402
from acmot_sci_v7 import LevelSource, SciV7Pipeline  # noqa: E402
from adapters.detectors.compute_profile import ComputeProfileAdapter  # noqa: E402

GENERIC = ["acmot_sci.py", "acmot_sci_v7.py"]
FORBIDDEN = [r"yolo", r"detr", r"rcnn", r"bytetrack", r"byte_track", r"botsort", r"bot_sort",
             r"ocsort", r"oc_sort", r"sparsetrack", r"boosttrack", r"visdrone", r"uavdt", r"mot17",
             r"mot20", r"kitti", r"ultralytics", r"\b640\b", r"\b736\b", r"\b832\b"]


def test_v7_policy_lock_unchanged():
    lock = json.loads((ROOT / "research/V7_POLICY_LOCK.json").read_text())
    for f, h in lock["file_sha256"].items():
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h, f


def test_v7_layer_identical_to_freeze_tag():
    r = subprocess.run(["git", "show", "universal-acmot-v7-freeze:acmot_v7.py"], cwd=ROOT,
                       capture_output=True)
    if r.returncode:
        pytest.skip("freeze tag not available in this checkout")
    assert r.stdout == (ROOT / "acmot_v7.py").read_bytes()


@pytest.mark.parametrize("name", GENERIC)
def test_generic_layer_has_no_detector_tracker_or_dataset_names(name):
    src = (ROOT / name).read_text().lower()
    hits = [p for p in FORBIDDEN if re.search(p, src)]
    assert not hits, f"{name} contains {hits}"


def test_scene_layer_reads_no_score_and_sets_no_threshold():
    src = (ROOT / "acmot_sci.py").read_text()
    code = re.sub(r'"""[\s\S]*?"""', "", src)
    code = re.sub(r"#.*", "", code)
    for tok in ("confidence", "score", "assoc", "birth", "nms", "acmot_v7", "threshold("):
        assert tok not in code.lower(), tok


def _historical(n_steps, seed):
    from core import Config, Controller
    rng = np.random.default_rng(seed)
    ctl = Controller(Config("hist", policy="adaptive", stable=True, recovery=True,
                            detector_feedback=True))
    sl = SceneLayer()
    size_of = {"LOW": 640, "MEDIUM": 736, "HIGH": 832}
    prev = np.zeros((0, 6))
    for f in range(1, n_steps + 1):
        vis = dict(edges=float(rng.uniform(0, 0.2)), brightness=float(rng.uniform(40, 200)),
                   blur=float(rng.uniform(50, 2500)))
        a = ctl.choose(f, vis, prev)
        b = sl.decide(f, vis)
        assert size_of[b.level] == a["size"], (f, a, b)
        assert b.sci == pytest.approx(a["sci"]) and b.scene == a["scene"]
        # objects of frame f, used from f+1 on (scores 1: the historical score gate never binds)
        k = int(rng.choice([0, 2, 8, 25, 45]) if rng.random() < 0.3 else rng.integers(0, 40))
        xy = rng.uniform(0, 1000, (k, 2))
        wh = np.where(rng.random((k, 1)) < 0.5, rng.uniform(5, 30, (k, 2)), rng.uniform(30, 120, (k, 2)))
        boxes = np.hstack([xy, xy + wh])
        prev = np.hstack([boxes, np.ones((k, 1)), np.zeros((k, 1))])
        sl.observe(boxes)


@pytest.mark.parametrize("seed", range(6))
def test_scene_layer_equals_historical_controller(seed):
    _historical(1500, seed)


def test_adapter_refuses_unsupported_setting():
    with pytest.raises(ValueError):
        ComputeProfileAdapter({"LOW": 512, "MEDIUM": 736, "HIGH": 832}, supported={640, 736, 832})
    with pytest.raises(ValueError):
        ComputeProfileAdapter({"LOW": 640, "MEDIUM": 736})
    a = ComputeProfileAdapter.from_config("yolov8", supported={640, 736, 832})
    assert [a.resolution(lv) for lv in LEVELS] == [640, 736, 832]


@pytest.mark.parametrize("split", ["conf16", "testdev", "dev40", "uavdt", "kitti", "train"])
def test_protected_splits_refused(split):
    from tools.sci_v7.dev import guard
    with pytest.raises(SystemExit):
        guard(split)
    env = dict(os.environ, V7_SPLIT=split)
    r = subprocess.run([sys.executable, "-c", "import tools.sci_v7.dev"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "PROTECTED" in (r.stderr + r.stdout)


# ---- synthetic stream for causality ---------------------------------------
class _Det:
    def __init__(self, x1, y1, x2, y2, confidence, class_id):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.confidence, self.class_id = confidence, class_id


def _stream(n_frames, seed):
    rng = np.random.default_rng(seed)
    objs = rng.uniform([0, 0, 10, 10], [900, 500, 60, 60], (30, 4))
    vel = rng.normal(0, 3, (30, 2))
    frames = []
    for f in range(n_frames):
        per_res = {}
        for r in (640, 736, 832):
            keep = rng.random(30) < (0.5 + 0.1 * (r - 640) / 96)
            dets = []
            for o, v, k in zip(objs, vel, keep):
                if k:
                    x, y = o[0] + v[0] * f, o[1] + v[1] * f
                    dets.append(_Det(x, y, x + o[2], y + o[3], float(rng.uniform(0.05, 0.95)), 0))
            for _ in range(rng.integers(0, 15)):
                x, y = rng.uniform(0, 1000, 2)
                dets.append(_Det(x, y, x + 20, y + 20, float(rng.uniform(0.01, 0.4)), 0))
            per_res[r] = dets
        vis = dict(edges=float(rng.uniform(0.05, 0.2)), brightness=float(rng.uniform(60, 200)),
                   blur=float(rng.uniform(100, 2000)))
        frames.append((per_res, vis, float(rng.uniform(0.5, 3))))
    return frames


def _pipeline(trace=False):
    from acmot_v7 import HostContract, V7Layer, spec_from_dict
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from adapters.types import Detection
    from tools.v7.systems import HOST_BYTETRACK as h, SYSTEMS
    return SciV7Pipeline(
        levels=LevelSource(scene=SceneLayer(SceneSpec(dwell=10))),
        adapter=ComputeProfileAdapter({"LOW": 640, "MEDIUM": 736, "HIGH": 832}),
        layer=V7Layer(spec_from_dict(dict(SYSTEMS["V7f"], name="V7f")), HostContract(**h)),
        tracker=ByteTrackAdapter(high=h["assoc"], low=h["low"], new=h["birth"], buffer=30,
                                 match=h["match"], fuse=True),
        make_detection=lambda d, v: Detection(d.x1, d.y1, d.x2, d.y2, v, d.class_id),
        record_trace=trace)


def _run(frames, trace=False):
    p = _pipeline(trace)
    out = []
    for per_res, vis, mot in frames:
        tracks, a = p.step(vis, lambda r, pr=per_res: pr[r], (1080, 1920), motion=mot)
        out.append((a, [(t.track_id, round(t.x1, 6)) for t in tracks]))
    return out, p


def _decision_keys(a):
    return {k: a.get(k) for k in ("level", "setting", "sci", "scene", "regime", "t1", "t2",
                                  "rho_bar", "assoc", "birth")}


def test_future_perturbation_does_not_change_past_decisions():
    frames = _stream(120, 0)
    base, _ = _run(frames)
    levels = {a["level"] for a, _ in base}
    assert len(levels) > 1, "synthetic stream must exercise level switching"
    t0 = 70
    pert = list(frames[:t0]) + _stream(120, 99)[t0:]
    other, _ = _run(pert)
    for t in range(t0):
        assert base[t] == other[t], t


def test_frame_t_detections_do_not_change_frame_t_scene_or_thresholds():
    frames = _stream(120, 1)
    base, _ = _run(frames)
    for t0 in (35, 61, 90):
        alt = list(frames)
        per_res, vis, mot = frames[t0]
        noise = _stream(1, 1000 + t0)[0][0]
        alt[t0] = (noise, vis, mot)
        other, _ = _run(alt)
        # scene level and V7f thresholds/regime of frame t0 come from frames < t0
        assert _decision_keys(base[t0][0]) == _decision_keys(other[t0][0]), t0


def test_trace_order():
    frames = _stream(25, 2)
    _, p = _run(frames, trace=True)
    ev = p.trace
    assert len(ev) == 5 * 25
    for t in range(1, 26):
        names = [e[0] for e in ev[5 * (t - 1): 5 * t]]
        assert names == ["scene", "detect", "v7", "track", "observe"]
        assert all(e[1] == t for e in ev[5 * (t - 1): 5 * t])


def test_fixed_level_reproduces_v7_record_if_present():
    """V7f+MEDIUM through the new pipeline == the frozen V7f runner at 736."""
    import pickle
    a = ROOT / "outputs/sci_v7/val7/V7f+MEDIUM/yolov8/uav0000086_00000_v.pkl"
    b = ROOT / "outputs/v7/val7/V7f/yolov8/uav0000086_00000_v.pkl"
    if not (a.exists() and b.exists()):
        pytest.skip("val-7 outputs not present")
    assert pickle.load(open(a, "rb"))["tracks_txt"] == pickle.load(open(b, "rb"))["tracks_txt"]


def test_g1_lock_unchanged():
    lock = json.loads((ROOT / "research/GENERAL_ACMOT_G1_LOCK.json").read_text())
    bad = [f for f, h in lock["file_sha256"].items()
           if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    assert not bad, bad
