import numpy as np

from tools.g2.gsci_audit import entropy_weights, minmax, pca_weights, sdc


def test_entropy_weight_favours_concentrated_indicator():
    rng = np.random.default_rng(0)
    spread = rng.uniform(0.4, 0.6, 200)
    concentrated = np.zeros(200)
    concentrated[:5] = 1.0
    w = entropy_weights(minmax(np.column_stack([spread, concentrated])))
    assert abs(w.sum() - 1) < 1e-12
    assert w[1] > w[0]


def test_constant_indicator_gets_no_entropy_weight():
    rng = np.random.default_rng(1)
    X = minmax(np.column_stack([rng.uniform(size=50), np.full(50, 3.0)]))
    w = entropy_weights(X)
    assert w[1] == 0 and abs(w[0] - 1) < 1e-12


def test_pca_weights_sum_to_one_and_index_in_unit_range():
    rng = np.random.default_rng(2)
    a = rng.normal(size=300)
    X = minmax(np.column_stack([a, a + 0.1 * rng.normal(size=300), rng.normal(size=300)]))
    wp = pca_weights(X)
    assert abs(wp.sum() - 1) < 1e-12 and (wp > 0).all()
    idx, w = sdc(X)
    assert abs(w.sum() - 1) < 1e-12
    assert idx.min() >= 0 and idx.max() <= 1


def test_frozen_parameters_reproduce_the_development_indices():
    import json
    from pathlib import Path
    from tools.g2.gsci_audit import indices
    rows = json.loads((Path(__file__).resolve().parents[1] / "research/final/sci_v7f/general_sci/cue_audit_V7f_512_960.json").read_text())["rows"]
    fit, _, params = indices(rows)
    params = json.loads(json.dumps(params))
    again, _, _ = indices(rows, params)
    for k in fit:
        assert np.allclose(fit[k], again[k], rtol=0, atol=1e-12)
    shifted = [dict(r, img_edges=r["img_edges"] * 2) for r in rows]
    moved, _, _ = indices(shifted, params)
    assert not np.allclose(moved["G"], fit["G"])
