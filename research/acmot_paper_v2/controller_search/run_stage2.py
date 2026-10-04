"""
STAGE 2 -- exhaustive 27-config grid over the level -> NMS mapping,
resolution fixed at Stage 1's winner (1088 for all three levels, per
STAGE1_RESULT.json). Calibration only. Deterministic Optuna GridSampler,
full trial log, same 4-fold CV as Stage 1.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.acmot_paper_v2.controller_search.sim import (  # noqa: E402
    CALIBRATION, NMS_VALUES, run_policy_on_sequence, score_sequence,
)
from tools.v6.eval_official import combine_official  # noqa: E402
from research.acmot_paper_v2.controller_search.run_stage1 import (  # noqa: E402
    FOLDS, matched_static_fold_hota,
)

HERE = Path(__file__).resolve().parent
STAGE1 = json.loads((HERE / "STAGE1_RESULT.json").read_text())
RES_FIXED = STAGE1["best_trial"]["params"]  # {'res_low':1088,'res_medium':1088,'res_high':1088}


def evaluate_candidate(action_map):
    per_seq_stats = {}
    for seq in CALIBRATION:
        tr, _ = run_policy_on_sequence("visdrone_calib", seq, action_map, "bytetrack")
        per_seq_stats[seq] = score_sequence("visdrone_calib", seq, tr)
    per_fold_delta = []
    for i, fold in enumerate(FOLDS):
        cand_hota = combine_official([per_seq_stats[s] for s in fold])["HOTA"]
        per_fold_delta.append(cand_hota - matched_static_fold_hota(i))
    cv_mean = sum(per_fold_delta) / len(per_fold_delta)
    per_seq_hota = {s: combine_official([per_seq_stats[s]])["HOTA"] for s in CALIBRATION}
    return cv_mean, per_fold_delta, per_seq_hota


def main():
    t0 = time.time()
    search_space = {"nms_low": NMS_VALUES, "nms_medium": NMS_VALUES, "nms_high": NMS_VALUES}
    sampler = optuna.samplers.GridSampler(search_space, seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="stage2_nms")

    trials_log = []

    def objective(trial):
        nms_low = trial.suggest_categorical("nms_low", NMS_VALUES)
        nms_med = trial.suggest_categorical("nms_medium", NMS_VALUES)
        nms_high = trial.suggest_categorical("nms_high", NMS_VALUES)
        action_map = {"LOW": (RES_FIXED["res_low"], nms_low), "MEDIUM": (RES_FIXED["res_medium"], nms_med),
                      "HIGH": (RES_FIXED["res_high"], nms_high)}
        cv_mean, per_fold, per_seq = evaluate_candidate(action_map)
        trials_log.append(dict(trial=trial.number, nms_low=nms_low, nms_medium=nms_med, nms_high=nms_high,
                                cv_mean_delta=cv_mean, per_fold_delta=per_fold, per_seq_hota=per_seq,
                                elapsed_s=time.time() - t0))
        print("trial", trial.number, "nms", (nms_low, nms_med, nms_high), "cv_mean_delta", round(cv_mean, 4),
              "elapsed", round(time.time() - t0, 1))
        return cv_mean

    n_trials = len(list(itertools.product(NMS_VALUES, NMS_VALUES, NMS_VALUES)))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    result = dict(
        search_space=search_space, sampler="optuna.samplers.GridSampler", seed=42, n_trials=n_trials,
        resolution_fixed=RES_FIXED, folds=FOLDS,
        best_trial=dict(number=best.number, params=best.params, cv_mean_delta=best.value),
        all_trials=trials_log, wall_seconds=time.time() - t0,
    )
    json.dump(result, open(HERE / "STAGE2_RESULT.json", "w"), indent=1, default=str)
    print("BEST:", best.params, "cv_mean_delta", best.value)
    print("wrote", HERE / "STAGE2_RESULT.json")


if __name__ == "__main__":
    main()
