"""
Separation tests: the AC policy must not know detector or tracker identity.

1. Static: policy/controller modules contain no detector/tracker names and
   do not import detector/tracker implementations.
2. Behavioural: two different detector adapter classes that return the
   same detections produce identical tracks (the policy only sees the
   generic Detection list).
3. Tracker contract: the policy only uses generic tracker arguments.
Run: PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_universal_separation.py
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY_MODULES = [
    "universal_policy_pipeline.py",
    "universal_acmot.py",
    "adapters/detectors/candidate_density.py",
    "adapters/detectors/generic_controls.py",
    "adapters/detectors/online_normalizer.py",
]
FORBIDDEN = re.compile(r"yolo|rtdetr|rt-detr|rt_detr|detr|rcnn|bytetrack|"
                       r"byte_tracker|botsort|bot_sort", re.I)


def _code_without_docstrings(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef,
                             ast.AsyncFunctionDef)) and node.body and \
                isinstance(node.body[0], ast.Expr) and \
                isinstance(getattr(node.body[0], "value", None), ast.Constant):
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_policy_modules_have_no_detector_or_tracker_identity():
    for rel in POLICY_MODULES:
        code = _code_without_docstrings(ROOT / rel)
        code = re.sub(r"#.*", "", code)
        hits = FORBIDDEN.findall(code)
        assert not hits, f"{rel} references {set(hits)}"


class _FakeDetectorA:
    def __init__(self, dets):
        self.dets = dets
        self.i = 0

    def detect(self, image, confidence, suppression, resolution):
        out = [d for d in self.dets[self.i] if d.confidence >= confidence]
        self.i += 1
        return out


class _FakeDetectorB(_FakeDetectorA):
    """Different class, different name, same outputs."""


def _synthetic_stream(n_frames=40, seed=0):
    from adapters.types import Detection
    rng = np.random.default_rng(seed)
    objs = rng.uniform(50, 900, size=(12, 2))
    frames = []
    for t in range(n_frames):
        dets = []
        for k, (x, y) in enumerate(objs):
            x += t * (k % 3)
            dets.append(Detection(x, y, x + 30, y + 20,
                                  float(0.9 - 0.05 * k), 2))
        for _ in range(30):   # clutter
            x, y = rng.uniform(0, 1000, 2)
            dets.append(Detection(x, y, x + 10, y + 10,
                                  float(rng.uniform(0.01, 0.3)), 2))
        frames.append(dets)
    return frames


def _run(detector_cls, policy_overrides):
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from universal_acmot import UniversalACMOT
    from universal_policy_pipeline import POLICIES, replace

    stream = _synthetic_stream()
    system = UniversalACMOT(detector_cls(stream), ByteTrackAdapter(),
                            policy=replace(POLICIES["V2cA"],
                                           **policy_overrides),
                            density_kwargs={})
    out = []
    img = np.zeros((1080, 1920, 3), np.uint8)
    for _ in stream:
        out.append([(t.track_id, round(t.x1, 3), round(t.y1, 3))
                    for t in system(img)])
    return out


def test_detector_identity_does_not_change_behaviour():
    for ov in ({}, {"leader_rho": 0.5}):
        assert _run(_FakeDetectorA, ov) == _run(_FakeDetectorB, ov)


def test_policy_uses_only_generic_tracker_arguments():
    src = (ROOT / "universal_policy_pipeline.py").read_text()
    assert "association_threshold=" in src and "birth_threshold=" in src
    assert "high_thresh=" not in src and "new_track_thresh=" not in src
