"""tools/v7/bootstrap.py: paired resampling mechanics (toy pooled metric)."""
import numpy as np

from tools.v7.bootstrap import paired


def toy_combine(cells):
    tp = sum(c["tp"] for c in cells)
    gt = sum(c["gt"] for c in cells)
    return {"R": 100.0 * tp / gt}


def test_identical_systems_give_zero_interval():
    A = [dict(tp=t, gt=100) for t in (50, 60, 70, 80)]
    r = paired(A, list(A), toy_combine, n=500, keys=["R"])["R"]
    assert r["diff"] == r["ci_lo"] == r["ci_hi"] == 0 and r["p_le0"] == 1 and r["p_ge0"] == 1


def test_uniform_gain_is_detected_and_seeded():
    A = [dict(tp=t, gt=100) for t in (50, 60, 70, 80, 55, 65)]
    B = [dict(tp=t + 5, gt=100) for t in (50, 60, 70, 80, 55, 65)]
    r1 = paired(A, B, toy_combine, n=1000, keys=["R"])["R"]
    r2 = paired(A, B, toy_combine, n=1000, keys=["R"])["R"]
    assert {k: v for k, v in r1.items() if k != "dz"} == {k: v for k, v in r2.items() if k != "dz"}
    assert np.isclose(r1["diff"], 5) and r1["ci_lo"] > 0 and r1["p_le0"] == 0
    assert r1["seq_wins"] == 6 and r1["seq_losses"] == 0
