"""
Shuffled-control validation for the selected adaptive candidate (v3 trial 7):
same per-sequence action-use counts, random (seed=44) temporal order
within each sequence instead of the causal SceneLayer order. Reported as
supporting evidence, not a gate.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.acmot_paper_v2.controller_search.sim import CALIBRATION, get_sequence, make_tracker, score_sequence  # noqa: E402
from tools.v6.eval_official import combine_official  # noqa: E402
from research.acmot_paper_v2.controller_search.run_v3_joint_search import (  # noqa: E402
    FOLDS, baseline_fold_hota, spec_for, run_policy_with_conf,
)

HERE = Path(__file__).resolve().parent
ACTION_MAP = {"LOW": (1536, 0.70, 0.10), "MEDIUM": (1280, 0.45, 0.40), "HIGH": (1088, 0.60, 0.40)}
SPEC = spec_for(0.29747709701135766, 0.5114928923560124)
SEED = 44


def run_shuffled_on_sequence(seq, levels_causal):
    rng = np.random.default_rng(SEED + hash(seq) % 1000)
    shuffled = list(levels_causal)
    rng.shuffle(shuffled)
    sd = get_sequence("visdrone_calib", seq)
    tracker = make_tracker("bytetrack")
    out = []
    for t in range(1, sd.n_frames + 1):
        res, nms, conf = ACTION_MAP[shuffled[t - 1]]
        dets = sd.detections(t, res, nms, conf)
        tracks = tracker.update(dets, sd.shape)
        for tr in tracks:
            out.append([t, tr.track_id, tr.x1, tr.y1, tr.x2 - tr.x1, tr.y2 - tr.y1, tr.confidence, tr.class_id, -1, -1])
    return np.asarray(out, dtype=float).reshape(-1, 10) if out else np.zeros((0, 10))


def main():
    t0 = time.time()
    causal_stats, shuffled_stats, levels_by_seq = {}, {}, {}
    for seq in CALIBRATION:
        tr_causal, levels = run_policy_with_conf("visdrone_calib", seq, ACTION_MAP, "bytetrack", SPEC)
        causal_stats[seq] = score_sequence("visdrone_calib", seq, tr_causal)
        levels_by_seq[seq] = levels
        tr_shuf = run_shuffled_on_sequence(seq, levels)
        shuffled_stats[seq] = score_sequence("visdrone_calib", seq, tr_shuf)
        print(seq, "causal", combine_official([causal_stats[seq]])["HOTA"], "shuffled",
              combine_official([shuffled_stats[seq]])["HOTA"], "elapsed", round(time.time() - t0, 1))

    causal_fold, shuffled_fold = [], []
    for i, fold in enumerate(FOLDS):
        c = combine_official([causal_stats[s] for s in fold])["HOTA"]
        s = combine_official([shuffled_stats[s] for s in fold])["HOTA"]
        base = baseline_fold_hota(i)
        causal_fold.append(c - base)
        shuffled_fold.append(s - base)
        print("fold", i, "causal_delta", c - base, "shuffled_delta", s - base)

    result = dict(action_map=ACTION_MAP, seed=SEED, causal_cv_mean=sum(causal_fold) / 4,
                  shuffled_cv_mean=sum(shuffled_fold) / 4, causal_per_fold=causal_fold,
                  shuffled_per_fold=shuffled_fold, causal_beats_shuffled=sum(causal_fold) > sum(shuffled_fold),
                  wall_seconds=time.time() - t0)
    json.dump(result, open(HERE / "SHUFFLED_CONTROL_RESULT.json", "w"), indent=1)
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
