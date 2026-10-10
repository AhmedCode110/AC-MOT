import numpy as np

from adapters.trackers.oatrack import OATrack


def run(frames, **kw):
    t = OATrack(**kw)
    return [t.update_arrays(np.array(b, float).reshape(-1, 4), np.array(s, float), [0] * len(s)) for b, s in frames]


def test_identity_kept_and_recovered_after_gap():
    box = [100, 100, 110, 120]
    frames = [([box], [0.9])] * 5 + [([], [])] * 4 + [([box], [0.9])] * 3
    out = run(frames)
    ids = {o[1] for f in out for o in f}
    assert ids == {1}
    assert all(len(f) == 0 for f in out[5:9])


def test_lost_track_removed_after_max_lost():
    box = [100, 100, 110, 120]
    frames = [([box], [0.9])] + [([], [])] * 31 + [([box], [0.9])]
    out = run(frames)
    assert out[0][0][1] == 1 and out[-1][0][1] == 2


def test_birth_and_candidate_gates():
    out = run([([[0, 0, 10, 10], [50, 50, 60, 60]], [0.5, 0.3])])
    assert out[0] == []                      # 0.5 < tau_high, 0.3 < min_conf
    out = run([([[0, 0, 10, 10]], [0.9]), ([[0, 0, 10, 10]], [0.45])])
    assert len(out[1]) == 1 and out[1][0][1] == 1   # weak detection continues a track (stage 2)


def test_two_objects_do_not_swap():
    a, b = [0, 0, 10, 20], [30, 0, 40, 20]
    frames = [([a, b], [0.9, 0.9])]
    for k in range(1, 6):
        frames.append(([[a[0] + k, 0, a[2] + k, 20], [b[0] - k, 0, b[2] - k, 20]], [0.9, 0.9]))
    out = run(frames)
    first = {o[1]: o[0][0] for o in out[0]}
    last = {o[1]: o[0][0] for o in out[-1]}
    assert last[1] < last[2] and first[1] < first[2]
