"""Paper 2 (post-freeze): the causality, reset, host-anchoring and
invariance statements of the manuscript checked on the FROZEN V7f policy
itself, loaded from configs/universal_acmot_policy_v7.json. The earlier
causality tests in test_v7_adaptive_layer.py are parametrised over the
development variants V6EMU/V7c/V7d; this file adds V7f. Nothing here changes
the frozen code or configuration (the lock test below re-checks them)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from acmot_v7 import HostContract, V7Layer, logit, sigmoid, spec_from_dict

ROOT = Path(__file__).resolve().parents[1]
V7F = spec_from_dict(json.loads((ROOT / "configs/universal_acmot_policy_v7.json").read_text())["spec"])
TWO_STAGE = HostContract(assoc=0.6, birth=0.7, low=0.1, match=0.8)      # e.g. an official ByteTrack setting
SINGLE = HostContract(assoc=0.6, birth=0.6, low=0.6, match=0.7)         # e.g. an OC-SORT setting
HOSTS = [TWO_STAGE, SINGLE]


def frame(rng, noisy, n_obj=12, n_bg=30):
    n_conf, n_amb = (3, n_obj) if noisy else (n_obj, 3)
    s = np.r_[rng.uniform(0.85, 0.97, n_conf), rng.uniform(0.35, 0.55, n_amb),
              rng.uniform(0.01, 0.12, n_bg)]
    xy = rng.uniform(0, 900, (len(s), 2))
    b = np.c_[xy, xy + rng.uniform(15, 60, (len(s), 2))]
    return b, s, rng.integers(0, 3, len(s)), float(rng.uniform(0.001, 0.02))


def stream(seed, noisy, n=40):
    rng = np.random.default_rng(seed)
    return [frame(rng, noisy) for _ in range(n)]


def replay(layer, frames, tf=lambda s: s):
    out = []
    for b, s, c, m in frames:
        d = layer.step(b, tf(s), m, classes=c)
        layer.observe(b[d.keep][:5], np.arange(min(5, len(d.keep))))
        out.append(d)
    return out


def test_frozen_files_match_the_lock():
    lock = json.loads((ROOT / "research/V7_POLICY_LOCK.json").read_text())
    bad = [p for p, h in lock["file_sha256"].items()
           if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    assert not bad, bad


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("noisy", [False, True])
def test_frame_t_operating_point_and_regime_use_only_frames_before_t(host, noisy):
    """Replacing frame t's candidates leaves the association/birth thresholds,
    the Otsu bands, the regime statistic and the regime of frame t unchanged."""
    fr = stream(1, noisy)
    layer = V7Layer(V7F, host)
    replay(layer, fr[:25])
    a, b = copy.deepcopy(layer), copy.deepcopy(layer)
    alt = frame(np.random.default_rng(99), not noisy, n_obj=40, n_bg=5)
    d0 = a.step(fr[25][0], fr[25][1], fr[25][3], classes=fr[25][2])
    d1 = b.step(alt[0], alt[1], fr[25][3], classes=alt[2])
    for k in ("regime", "t1", "t2", "rho", "rho_bar"):
        assert d0.log.get(k) == d1.log.get(k)
    assert (d0.assoc, d0.birth) == (d1.assoc, d1.birth)


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("noisy", [False, True])
def test_future_frames_do_not_change_past_decisions(host, noisy):
    fr = stream(3, noisy)
    full = replay(V7Layer(V7F, host), fr)
    part = replay(V7Layer(V7F, host), fr[:20])
    for x, y in zip(full[:20], part):
        np.testing.assert_array_equal(x.keep, y.keep)
        np.testing.assert_array_equal(x.scores, y.scores)
        assert (x.assoc, x.birth, x.match) == (y.assoc, y.birth, y.match)


@pytest.mark.parametrize("host", HOSTS)
def test_reset_between_sequences_equals_a_fresh_layer(host):
    fr, other = stream(4, False), stream(5, True)
    layer = V7Layer(V7F, host)
    replay(layer, other)
    layer.reset()
    for p, q in zip(replay(layer, fr), replay(V7Layer(V7F, host), fr)):
        np.testing.assert_array_equal(p.keep, q.keep)
        np.testing.assert_array_equal(p.scores, q.scores)
        assert (p.assoc, p.birth, p.match, p.log.get("regime")) == \
               (q.assoc, q.birth, q.match, q.log.get("regime"))


@pytest.mark.parametrize("host", HOSTS)
def test_cold_first_frame_is_host_pass_through_up_to_cross_class_duplicates(host):
    b, s, c, m = stream(6, False)[0]
    d = V7Layer(V7F, host).step(b, s, m, classes=c)
    assert d.log["regime"] == "cold"
    ref = V7Layer(V7F, host)._duplicates(b, s, c, rule="xclass")
    np.testing.assert_array_equal(d.keep, ref)
    np.testing.assert_array_equal(d.scores, s[d.keep])
    assert (d.assoc, d.birth, d.match) == (pytest.approx(host.assoc), pytest.approx(host.birth), host.match)


@pytest.mark.parametrize("host", HOSTS)
def test_clean_and_cold_frames_never_raise_host_thresholds_or_change_match(host):
    for d in replay(V7Layer(V7F, host), stream(7, False)):
        if d.log["regime"] in ("clean", "cold"):
            assert d.assoc <= host.assoc + 1e-12 and d.birth <= host.birth + 1e-12
            assert d.match == host.match


@pytest.mark.parametrize("host", HOSTS)
def test_match_tolerance_changes_only_in_noisy_frames_and_only_through_the_cue(host):
    for noisy in (False, True):
        fr = stream(2, noisy)
        layer = V7Layer(V7F, host)
        replay(layer, fr[:25])
        a, b = copy.deepcopy(layer), copy.deepcopy(layer)
        bx, s, c, m = fr[25]
        d0, d1 = a.step(bx, s, m, classes=c), b.step(bx, s, 50 * m, classes=c)
        np.testing.assert_array_equal(d0.keep, d1.keep)
        np.testing.assert_array_equal(d0.scores, d1.scores)
        if d0.log["regime"] == "noisy":
            assert d1.match > d0.match
        else:
            assert d1.match == d0.match == host.match


@pytest.mark.parametrize("a,b", [(0.5, 0.0), (2.0, 0.3), (1.0, -1.0)])
@pytest.mark.parametrize("host", HOSTS)
def test_regime_decisions_invariant_to_increasing_logit_affine_recalibration(a, b, host):
    """The regime sequence (and hence whether V7f intervenes) is unchanged
    when the detector logits are recalibrated by an increasing affine map;
    clean-frame thresholds stay anchored to the host in raw score units."""
    tf = lambda s: sigmoid(a * logit(s) + b)
    for noisy in (False, True):
        fr = stream(10, noisy)
        x = replay(V7Layer(V7F, host), fr)
        y = replay(V7Layer(V7F, host), fr, tf)
        assert [p.log.get("regime") for p in x] == [q.log.get("regime") for q in y]


@pytest.mark.parametrize("a,b", [(0.5, 0.0), (2.0, 0.3)])
def test_noisy_frames_pass_the_same_candidates_under_logit_affine_recalibration(a, b):
    tf = lambda s: sigmoid(a * logit(s) + b)
    fr = stream(9, True)
    x = replay(V7Layer(V7F, TWO_STAGE), fr)
    y = replay(V7Layer(V7F, TWO_STAGE), fr, tf)
    for p, q in zip(x, y):
        if p.log.get("regime") == "noisy":
            np.testing.assert_array_equal(p.keep, q.keep)
            np.testing.assert_allclose(q.log["t2"], a * p.log["t2"] + b, atol=1e-6)
