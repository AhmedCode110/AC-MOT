"""
STAGE 3A -- exhaustive 31-config grid over the cue SUBSET (which of
crowd/tiny/edge/dark/blur feed the SCI), weights renormalized within the
subset preserving historical ratios. Thresholds and action mapping fixed
at their Stage1/2-selected / historical values. See
PREDECLARATION_STAGE3PLUS.json.
"""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from acmot_sci import SceneSpec  # noqa: E402
from research.acmot_paper_v2.controller_search.sim import CALIBRATION, run_policy_on_sequence, score_sequence  # noqa: E402
from tools.v6.eval_official import combine_official  # noqa: E402
from research.acmot_paper_v2.controller_search.run_stage1 import FOLDS, matched_static_fold_hota  # noqa: E402

HERE = Path(__file__).resolve().parent
STAGE1 = json.loads((HERE / "STAGE1_RESULT.json").read_text())
STAGE2 = json.loads((HERE / "STAGE2_RESULT.json").read_text())
RES = STAGE1["best_trial"]["params"]
NMS = STAGE2["best_trial"]["params"]
ACTION_MAP = {"LOW": (RES["res_low"], NMS["nms_low"]), "MEDIUM": (RES["res_medium"], NMS["nms_medium"]),
              "HIGH": (RES["res_high"], NMS["nms_high"])}

CUES = ["crowd", "tiny", "edge", "dark", "blur"]
ORIGINAL_W = dict(crowd=0.30, tiny=0.30, edge=0.20, dark=0.10, blur=0.05)


def spec_for_subset(subset):
    total = sum(ORIGINAL_W[c] for c in subset)
    kwargs = {f"w_{c}": (ORIGINAL_W[c] / total if c in subset else 0.0) for c in CUES}
    return SceneSpec(**kwargs)


def evaluate_candidate(spec):
    per_seq_stats = {}
    for seq in CALIBRATION:
        tr, _ = run_policy_on_sequence("visdrone_calib", seq, ACTION_MAP, "bytetrack", scene_spec=spec)
        per_seq_stats[seq] = score_sequence("visdrone_calib", seq, tr)
    per_fold_delta = []
    for i, fold in enumerate(FOLDS):
        cand_hota = combine_official([per_seq_stats[s] for s in fold])["HOTA"]
        per_fold_delta.append(cand_hota - matched_static_fold_hota(i))
    cv_mean = sum(per_fold_delta) / len(per_fold_delta)
    return cv_mean, per_fold_delta


def main():
    t0 = time.time()
    subsets = []
    for k in range(1, len(CUES) + 1):
        subsets.extend(itertools.combinations(CUES, k))
    assert len(subsets) == 31

    trials = []
    for i, subset in enumerate(subsets):
        spec = spec_for_subset(subset)
        cv_mean, per_fold = evaluate_candidate(spec)
        trials.append(dict(trial=i, subset=list(subset), cv_mean_delta=cv_mean, per_fold_delta=per_fold,
                            elapsed_s=time.time() - t0))
        print("trial", i, "subset", subset, "cv_mean_delta", round(cv_mean, 4), "elapsed", round(time.time() - t0, 1))

    best = max(trials, key=lambda x: x["cv_mean_delta"])
    result = dict(action_map_fixed=ACTION_MAP, n_trials=31, folds=FOLDS, all_trials=trials,
                  best_trial=best, wall_seconds=time.time() - t0)
    json.dump(result, open(HERE / "STAGE3A_RESULT.json", "w"), indent=1, default=str)
    print("BEST:", best["subset"], "cv_mean_delta", best["cv_mean_delta"])


if __name__ == "__main__":
    main()
