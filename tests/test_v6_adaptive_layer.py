"""V6-TF adaptive layer: statistical primitives, causality, invariance,
state reset and absence of detector-specific logic (Amendment 9)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from online_calibration import (dedup, exact_otsu2, exact_otsu3,
                                exact_otsu3_fast, logits, nested_otsu)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/det_cache_val_native"
SEQ = "uav0000086_00000_v"


def policy():
    from universal_policy_pipeline import POLICIES, replace
    ov = json.loads((ROOT / "configs/universal_acmot_policy_v6tf.json").read_text())["overrides"]
    return replace(POLICIES["V1"], **ov)


def run(det, frames, transform=None, mutate=None, pipe=None):
    """Replay the cached detector through the V6-TF pipeline; returns
    per-frame (track tuples, audit)."""
    from run_universal_acmot import build_config
    from tools.run_policy_validation import CachedDetector, make_tracker
    from universal_policy_pipeline import UniversalPolicyPipeline
    from ultralytics.trackers.basetrack import BaseTrack
    BaseTrack.reset_id()                          # process-global ID counter
    cd = CachedDetector(str(CACHE / det / f"{SEQ}.npz"), transform=transform)
    if mutate:
        mutate(cd)
    pol = policy()
    if pipe is None:
        cfg = build_config()
        pipe = UniversalPolicyPipeline(cfg, cd, make_tracker(cfg, pol), pol)
    else:
        pipe.detector = cd
        pipe.reset()
    img = np.empty(cd.shape + (0,), np.uint8)
    out = []
    for i in range(1, frames + 1):
        cd.frame = i
        r = pipe.process(i, img, cd.visual_dict(i))
        out.append(([(t.track_id, round(t.x1, 3), round(t.y1, 3), round(t.x2, 3),
                      round(t.y2, 3), round(t.confidence, 6)) for t in r["tracks"]],
                    {k: v for k, v in r["audit"].items() if k.startswith("tf_")}))
    return out, pipe


needs_cache = pytest.mark.skipif(not (CACHE / "yolov8" / f"{SEQ}.npz").exists(),
                                 reason="development cache not present")


def test_exact_otsu_fast_equals_reference():
    rng = np.random.default_rng(1)
    for _ in range(50):
        v = np.concatenate([rng.normal(-3, 1, rng.integers(3, 80)),
                            rng.normal(1, .5, rng.integers(0, 20))])
        a, b = exact_otsu3(v), exact_otsu3_fast(v)
        assert (a is None) == (b is None)
        if a is not None:
            assert np.allclose(a, b)


def test_nested_otsu_affine_equivariant_and_ordered():
    rng = np.random.default_rng(2)
    v = np.concatenate([rng.normal(-4, 1, 300), rng.normal(0, .7, 40), rng.normal(2, .5, 15)])
    t1, t2, eta = nested_otsu(v)
    assert t1 <= t2 and 0 <= eta <= 1
    a, b = 2.5, -1.3                              # Platt / temperature map
    u1, u2, e2 = nested_otsu(a * v + b)
    assert np.isclose(u1, a * t1 + b) and np.isclose(u2, a * t2 + b) and np.isclose(eta, e2)
    assert nested_otsu(np.ones(10)) is None and exact_otsu2([1.0]) is None


def test_dedup_suppresses_only_redundant_boxes():
    boxes = [[0, 0, 10, 10], [1, 0, 11, 10], [20, 20, 30, 30], [0, 0, 10, 4]]
    keep = dedup(boxes, [0.9, 0.8, 0.7, 0.95], 0.5)
    assert sorted(keep) == [0, 2, 3]              # box 1 has IoU 0.82 with box 0
    keep2 = dedup(boxes[::-1], [0.95, 0.7, 0.8, 0.9], 0.5)
    assert sorted(3 - k for k in keep2) == [0, 2, 3]


def test_no_detector_or_dataset_names_in_adaptive_layer():
    pat = re.compile(r"yolo|rtdetr|rt-detr|detr|faster|rcnn|visdrone|uavdt", re.I)
    for f in ("online_calibration.py",):
        assert not pat.search((ROOT / f).read_text())
    src = (ROOT / "universal_policy_pipeline.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    body = code.split('"""', 2)[-1]               # skip the module docstring
    assert not pat.search(body)


@needs_cache
@pytest.mark.parametrize("det", ["yolov8", "rtdetr"])
def test_causality_future_frames_do_not_change_past(det):
    k, n = 30, 45
    base, _ = run(det, n)

    def mutate(cd):                               # scramble frames >= k
        rng = np.random.default_rng(0)
        for f in range(k, n + 1):
            a = cd.by_res[736].get(f)
            if a is not None and len(a):
                a = a.copy()
                a[:, 5] = rng.uniform(0.01, 0.99, len(a))
                a[:, 1:5] += rng.normal(0, 5, (len(a), 4))
                cd.by_res[736][f] = a
    pert, _ = run(det, n, mutate=mutate)
    assert base[:k - 1] == pert[:k - 1]
    # frame-k thresholds come from frames < k only
    for key in ("tf_otsu_t1", "tf_otsu_t2", "tf_band_lo"):
        assert base[k - 1][1][key] == pert[k - 1][1][key]
    assert base[k - 1][0] != pert[k - 1][0]       # frame k itself does change


@needs_cache
@pytest.mark.parametrize("tr", ["temp2", "temp05"])
def test_platt_invariance_exact(tr):
    a, _ = run("rtdetr", 40)
    b, _ = run("rtdetr", 40, transform=tr)
    assert [x[0] for x in a] == [x[0] for x in b]


@needs_cache
def test_state_reset_between_sequences():
    a, pipe = run("yolov8", 40)
    run("rtdetr", 25, pipe=pipe)                  # contaminate state
    b, _ = run("yolov8", 40, pipe=pipe)           # reset() inside run
    assert [x[0] for x in a] == [x[0] for x in b]
