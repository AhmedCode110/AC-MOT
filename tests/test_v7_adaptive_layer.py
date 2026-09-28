"""V7 adaptive layer (acmot_v7.py): causality as stated in the handoff (§1),
reset, pass-through identity, absence of detector/tracker/dataset names,
noisy-regime invariance to logit-affine score recalibration, emission floors,
crowd/duplicate cases, determinism and the E13 pooling option."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pytest

from acmot_v7 import HostContract, V7Layer, V7Spec, logit, sigmoid, spec_from_dict
from tools.v7.systems import SYSTEMS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/det_cache_val_native"
SEQ = "uav0000086_00000_v"
HOST = HostContract(assoc=0.25, birth=0.25, low=0.1, match=0.8)


def spec(name, **kw):
    return spec_from_dict(dict(SYSTEMS[name], name=name, **kw))


def synth_frame(rng, n_obj=12, n_bg=30, noisy=False):
    """One frame: confident + ambiguous object candidates and low-score
    background. Clean: most of the foreground is confident; noisy: most of
    it is ambiguous."""
    n_conf, n_amb = (3, n_obj) if noisy else (n_obj, 3)
    s = np.r_[rng.uniform(0.85, 0.97, n_conf), rng.uniform(0.35, 0.55, n_amb),
              rng.uniform(0.01, 0.12, n_bg)]
    xy = rng.uniform(0, 900, (len(s), 2))
    b = np.c_[xy, xy + rng.uniform(15, 60, (len(s), 2))]
    cls = rng.integers(0, 3, len(s))
    return b, s, cls


def stream(seed=0, frames=40, noisy=False):
    rng = np.random.default_rng(seed)
    return [synth_frame(rng, noisy=noisy) + (float(rng.uniform(0.001, 0.02)),)
            for _ in range(frames)]


def replay(layer, frames, tf=lambda s: s):
    out = []
    for b, s, cls, m in frames:
        d = layer.step(b, tf(s), m, classes=cls)
        layer.observe(b[d.keep][:5], np.arange(min(5, len(d.keep))))
        out.append(d)
    return out


def cached_frames(det, n=60):
    from tools.run_policy_validation import CachedDetector
    cd = CachedDetector(str(CACHE / det / f"{SEQ}.npz"))
    fr = []
    for i in range(1, n + 1):
        cd.frame = i
        raw = cd.detect(None, 0.0, None, 736)
        fr.append((np.array([[d.x1, d.y1, d.x2, d.y2] for d in raw]).reshape(-1, 4),
                   np.array([d.confidence for d in raw]),
                   np.array([d.class_id for d in raw]),
                   cd.visual_dict(i).get("motion")))
    return fr


needs_cache = pytest.mark.skipif(not (CACHE / "yolov8" / f"{SEQ}.npz").exists(),
                                 reason="development cache not present")
SYS = ["V6EMU", "V7c", "V7d"]


# ------------------------------------------------------------- causality
@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("name", SYS)
def test_thresholds_and_regime_use_only_past_frames(name, noisy):
    """Changing frame t's candidates leaves assoc/birth/regime/bands of
    frame t unchanged (they come from frames < t)."""
    fr = stream(1, noisy=noisy)
    layer = V7Layer(spec(name), HOST)
    replay(layer, fr[:25])
    a, b = copy.deepcopy(layer), copy.deepcopy(layer)
    rng = np.random.default_rng(99)
    b1, s1, c1 = synth_frame(rng, n_obj=40, n_bg=5)
    d0 = a.step(fr[25][0], fr[25][1], fr[25][3], classes=fr[25][2])
    d1 = b.step(b1, s1, fr[25][3], classes=c1)
    for k in ("regime", "t1", "t2", "rho_bar"):
        assert d0.log.get(k) == d1.log.get(k)
    assert (d0.assoc, d0.birth) == (d1.assoc, d1.birth)


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("name", SYS)
def test_match_tolerance_depends_on_frame_t_motion_only_through_the_cue(name, noisy):
    fr = stream(2, noisy=noisy)
    layer = V7Layer(spec(name), HOST)
    replay(layer, fr[:25])
    a, b = copy.deepcopy(layer), copy.deepcopy(layer)
    bx, s, c, m = fr[25]
    d0, d1 = a.step(bx, s, m, classes=c), b.step(bx, s, 50 * m, classes=c)
    np.testing.assert_array_equal(d0.keep, d1.keep)
    np.testing.assert_array_equal(d0.scores, d1.scores)
    assert (d0.assoc, d0.birth) == (d1.assoc, d1.birth)
    assert d0.log["regime"] == ("noisy" if noisy or name == "V6EMU" else "clean")
    if d0.log["regime"] == "noisy" or SYSTEMS[name].get("motion_regime", "always") == "always":
        assert d1.match > d0.match
    else:
        assert d1.match == d0.match == HOST.match


@pytest.mark.parametrize("name", SYS)
def test_future_frames_do_not_change_past_decisions(name):
    fr = stream(3)
    full = replay(V7Layer(spec(name), HOST), fr)
    part = replay(V7Layer(spec(name), HOST), fr[:20])
    for x, y in zip(full[:20], part):
        np.testing.assert_array_equal(x.keep, y.keep)
        np.testing.assert_array_equal(x.scores, y.scores)
        assert (x.assoc, x.birth, x.match) == (y.assoc, y.birth, y.match)


# ------------------------------------------------------------- reset / determinism
@pytest.mark.parametrize("name", SYS)
def test_reset_equals_fresh_layer_and_reruns_are_deterministic(name):
    fr, other = stream(4), stream(5)
    layer = V7Layer(spec(name), HOST)
    replay(layer, other)
    layer.reset()
    x = replay(layer, fr)
    y = replay(V7Layer(spec(name), HOST), fr)
    for p, q in zip(x, y):
        np.testing.assert_array_equal(p.keep, q.keep)
        np.testing.assert_array_equal(p.scores, q.scores)
        assert (p.assoc, p.birth, p.match, p.log.get("regime")) == \
               (q.assoc, q.birth, q.match, q.log.get("regime"))


# ------------------------------------------------------------- pass-through
def test_native_is_exact_pass_through():
    layer = V7Layer(spec("NATIVE"), HOST)
    for b, s, c, m in stream(6):
        d = layer.step(b, s, m, classes=c)
        np.testing.assert_array_equal(d.keep, np.arange(len(s)))
        np.testing.assert_array_equal(d.scores, s)
        assert (d.assoc, d.birth, d.match) == (pytest.approx(HOST.assoc),
                                               pytest.approx(HOST.birth), HOST.match)


def test_clean_regime_never_raises_host_thresholds():
    """clean=upper: in clean frames the host threshold is only lowered."""
    for name in ("V7c", "V7d"):
        for d in replay(V7Layer(spec(name), HOST), stream(7)):
            if d.log["regime"] in ("clean", "cold"):
                assert d.assoc <= HOST.assoc + 1e-12 and d.birth <= HOST.birth + 1e-12
                assert (d.scores <= 1).all()


def test_clean_stream_passes_every_candidate_raw():
    """V7c/V7d on a clean stream: no same-class duplicate removal, raw
    scores, nothing discarded (cross-class overlaps are absent here)."""
    rng = np.random.default_rng(8)
    fr = []
    for _ in range(40):
        b, s, _ = synth_frame(rng)
        fr.append((b, s, np.zeros(len(s), int), 0.01))
    for name in ("V7c", "V7d"):
        out = replay(V7Layer(spec(name), HOST), fr)
        for d, (b, s, c, m) in zip(out[1:], fr[1:]):
            assert d.log["regime"] == "clean"
            dup = V7Layer(spec(name), HOST)._duplicates(b, s, c, rule="xclass")
            np.testing.assert_array_equal(d.keep, dup)
            np.testing.assert_array_equal(d.scores, s[d.keep])


# ------------------------------------------------------------- names
def test_no_detector_tracker_dataset_or_sequence_names_in_layer():
    code = (ROOT / "acmot_v7.py").read_text().lower()
    code = re.sub(r'""".*?"""', "", code, flags=re.S)     # docstrings
    code = re.sub(r"#.*", "", code)                       # comments
    for w in ("yolo", "detr", "rcnn", "faster", "sparse", "boost", "bytetrack", "botsort",
              "visdrone", "mot17", "uavdt", "uav0", "topic", "yolox"):
        assert w not in code, w


def test_host_contract_has_no_identity_field():
    assert set(HostContract.__dataclass_fields__) == {"assoc", "birth", "low", "match", "cmc"}


# ------------------------------------------------------------- invariance
@pytest.mark.parametrize("a,b", [(0.5, 0.0), (2.0, 0.0), (1.0, -1.0), (1.5, 0.7)])
def test_noisy_regime_invariant_to_logit_affine_recalibration(a, b):
    """Forced-noisy V7 (Otsu bands + rank remap + track duplicates): the
    passed set, the remapped scores and the regime are unchanged when the
    detector's logits are recalibrated by any increasing affine map."""
    tf = lambda s: sigmoid(a * logit(s) + b)
    fr = stream(9, noisy=True)
    sp = spec("V7c", regime="noisy")
    x = replay(V7Layer(sp, HOST), fr)
    y = replay(V7Layer(sp, HOST), fr, tf)
    for p, q in zip(x, y):
        np.testing.assert_array_equal(p.keep, q.keep)
        assert p.log.get("regime") == q.log.get("regime")
        if p.log.get("regime") == "noisy":
            np.testing.assert_allclose(p.scores, q.scores, atol=1e-9)
            assert (p.assoc, p.birth) == (q.assoc, q.birth)
            np.testing.assert_allclose(q.log["t2"], a * p.log["t2"] + b, atol=1e-6)


@pytest.mark.parametrize("a,b", [(0.5, 0.0), (2.0, 0.3)])
def test_rho_regime_decision_invariant_to_logit_affine_recalibration(a, b):
    tf = lambda s: sigmoid(a * logit(s) + b)
    for noisy in (False, True):
        fr = stream(10, noisy=noisy)
        x = replay(V7Layer(spec("V7c", pool="raw"), HOST), fr)
        y = replay(V7Layer(spec("V7c", pool="raw"), HOST), fr, tf)
        assert [p.log.get("regime") for p in x] == [q.log.get("regime") for q in y]


# ------------------------------------------------------------- floors
@pytest.mark.parametrize("floor", [0.01, 0.05, 0.1, 0.2])
@pytest.mark.parametrize("name", SYS + ["NATIVE"])
def test_emission_floors_are_handled(floor, name):
    fr = [(b[s >= floor], s[s >= floor], c[s >= floor], m) for b, s, c, m in stream(11)]
    fr[5] = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), 0.01)      # empty frame
    for d, (b, s, c, m) in zip(replay(V7Layer(spec(name), HOST), fr), fr):
        assert np.all(np.isfinite([d.assoc, d.birth, d.match]))
        assert 0 < d.match <= 0.95 or d.match == HOST.match
        assert len(d.keep) == len(d.scores) <= len(s)
        assert np.all((d.scores > 0) & (d.scores < 1))
        if name == "NATIVE":
            assert len(d.keep) == len(s)


# ------------------------------------------------------------- duplicates / crowds
def _layer(**kw):
    return V7Layer(V7Spec(**kw), HOST)


def test_track_rule_keeps_overlap_claimed_by_a_different_track():
    """Two people overlapping (IoU > 0.5), each continuing its own track."""
    L = _layer(dup="track")
    p1, p2 = [100, 100, 140, 200], [110, 100, 150, 200]
    L.observe([p1, p2], [1, 2])
    L.frame = 1
    keep = L._duplicates(np.array([p1, p2], float), np.array([0.9, 0.8]))
    assert list(keep) == [0, 1]


def test_track_rule_removes_unclaimed_or_same_track_duplicate():
    L = _layer(dup="track")
    p1, dup = [100, 100, 140, 200], [102, 101, 141, 203]
    L.observe([p1], [1])
    assert list(L._duplicates(np.array([p1, dup], float), np.array([0.9, 0.6]))) == [0]
    L.observe(np.zeros((0, 4)), [])
    assert list(L._duplicates(np.array([p1, dup], float), np.array([0.9, 0.6]))) == [0]


def test_iou_rule_removes_overlapping_people_and_xclass_only_cross_class():
    p1, p2 = [100, 100, 140, 200], [110, 100, 150, 200]
    b, s = np.array([p1, p2], float), np.array([0.9, 0.8])
    assert list(_layer(dup="iou")._duplicates(b, s)) == [0]
    assert list(_layer()._duplicates(b, s, np.array([0, 0]), rule="xclass")) == [0, 1]
    assert list(_layer()._duplicates(b, s, np.array([0, 1]), rule="xclass")) == [0]


def test_clean_frames_use_the_clean_duplicate_rule():
    """dup_regime=noisy: in clean frames same-class overlaps survive even
    without track context (crowd-safe), cross-class ones are removed."""
    L = V7Layer(spec("V7c"), HOST)
    rng = np.random.default_rng(12)
    for _ in range(20):
        b, s, _ = synth_frame(rng)
        d = L.step(b, s, 0.01, classes=np.zeros(len(s), int))
    p1, p2, p3 = [100, 100, 140, 200], [110, 100, 150, 200], [101, 100, 141, 201]
    b = np.array([p1, p2, p3], float)
    d = L.step(b, np.array([0.9, 0.85, 0.8]), 0.01, classes=np.array([0, 0, 1]))
    assert d.log["regime"] == "clean"
    assert list(d.keep) == [0, 1]


# ------------------------------------------------------------- E13 pooling
def test_pool_raw_uses_every_emitted_candidate():
    fr = stream(13)
    for pool, expect_all in (("raw", True), ("post", False)):
        L = V7Layer(spec("V6EMU", pool=pool), HOST)
        b, s, c, m = fr[0]
        b = np.r_[b, b[:3] + 1]
        s = np.r_[s, s[:3] - 0.01]
        c = np.r_[c, c[:3]]
        d = L.step(b, s, m, classes=c)
        assert len(L.window[-1]) == (len(s) if expect_all else len(s) - d.log["n_dup"])
    assert V7Spec().pool == "post"                # V6EMU identity default


# ------------------------------------------------------------- real caches
@needs_cache
@pytest.mark.parametrize("det", ["yolov8", "rtdetr"])
def test_cached_stream_causality_and_determinism(det):
    fr = cached_frames(det)
    for name in SYS:
        x = replay(V7Layer(spec(name), HOST), fr)
        y = replay(V7Layer(spec(name), HOST), fr)
        for p, q in zip(x, y):
            np.testing.assert_array_equal(p.keep, q.keep)
            assert (p.assoc, p.birth, p.match) == (q.assoc, q.birth, q.match)
        part = replay(V7Layer(spec(name), HOST), fr[:30])
        for p, q in zip(x[:30], part):
            np.testing.assert_array_equal(p.keep, q.keep)
