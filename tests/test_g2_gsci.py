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
