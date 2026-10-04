"""
Joint Stage 3B+4 search (see PREDECLARATION_AMENDMENT.json): thresholds
(t_med, t_high) AND the action mapping (level -> resolution, NMS)
searched together, cue subset fixed at Stage 3A's winner. Optuna TPE,
60-trial budget, calibration-only, 4-fold CV.
"""
import json
import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from acmot_sci import SceneSpec  # noqa: E402
from research.acmot_paper_v2.controller_search.sim import (  # noqa: E402
    CALIBRATION, RESOLUTIONS, NMS_VALUES, run_policy_on_sequence, score_sequence,
)
from tools.v6.eval_official import combine_official  # noqa: E402
from research.acmot_paper_v2.controller_search.run_stage1 import FOLDS, matched_static_fold_hota  # noqa: E402
from research.acmot_paper_v2.controller_search.run_stage3a import ORIGINAL_W  # noqa: E402

HERE = Path(__file__).resolve().parent
STAGE3A = json.loads((HERE / "STAGE3A_RESULT.json").read_text())
CUE_SUBSET = STAGE3A["best_trial"]["subset"]


def spec_for(t_med, t_high):
    total = sum(ORIGINAL_W[c] for c in CUE_SUBSET)
    kwargs = {f"w_{c}": (ORIGINAL_W[c] / total if c in CUE_SUBSET else 0.0) for c in ORIGINAL_W}
    return SceneSpec(t_med=t_med, t_high=t_high, **kwargs)


def evaluate_candidate(spec, action_map):
    per_seq_stats = {}
    for seq in CALIBRATION:
        tr, levels = run_policy_on_sequence("visdrone_calib", seq, action_map, "bytetrack", scene_spec=spec,
                                             record_levels=True)
        per_seq_stats[seq] = (score_sequence("visdrone_calib", seq, tr), levels)
    per_fold_delta = []
    for i, fold in enumerate(FOLDS):
        cand_hota = combine_official([per_seq_stats[s][0] for s in fold])["HOTA"]
        per_fold_delta.append(cand_hota - matched_static_fold_hota(i))
    cv_mean = sum(per_fold_delta) / len(per_fold_delta)
    level_counts = {lv: sum(l.count(lv) for _, l in per_seq_stats.values()) for lv in ("LOW", "MEDIUM", "HIGH")}
    return cv_mean, per_fold_delta, level_counts


def main():
    t0 = time.time()
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="stage3b4_joint")
    trials_log = []

    def objective(trial):
        t_med = trial.suggest_float("t_med", 0.10, 0.50)
        t_high = trial.suggest_float("t_high", t_med + 0.05, min(t_med + 0.40, 0.90))
        res_low = trial.suggest_categorical("res_low", RESOLUTIONS)
        res_med = trial.suggest_categorical("res_medium", RESOLUTIONS)
        res_high = trial.suggest_categorical("res_high", RESOLUTIONS)
        nms_low = trial.suggest_categorical("nms_low", NMS_VALUES)
        nms_med = trial.suggest_categorical("nms_medium", NMS_VALUES)
        nms_high = trial.suggest_categorical("nms_high", NMS_VALUES)
        spec = spec_for(t_med, t_high)
        action_map = {"LOW": (res_low, nms_low), "MEDIUM": (res_med, nms_med), "HIGH": (res_high, nms_high)}
        cv_mean, per_fold, level_counts = evaluate_candidate(spec, action_map)
        trials_log.append(dict(trial=trial.number, t_med=t_med, t_high=t_high, res_low=res_low, res_medium=res_med,
                                res_high=res_high, nms_low=nms_low, nms_medium=nms_med, nms_high=nms_high,
                                cv_mean_delta=cv_mean, per_fold_delta=per_fold, level_counts=level_counts,
                                elapsed_s=time.time() - t0))
        print("trial", trial.number, "t_med/t_high", round(t_med, 3), round(t_high, 3), "res",
              (res_low, res_med, res_high), "nms", (nms_low, nms_med, nms_high), "cv_mean_delta", round(cv_mean, 4),
              "levels", level_counts, "elapsed", round(time.time() - t0, 1))
        return cv_mean

    study.optimize(objective, n_trials=60)
    best = study.best_trial
    result = dict(cue_subset=CUE_SUBSET, sampler="optuna.samplers.TPESampler", seed=42, n_trials=60,
                  folds=FOLDS, best_trial=dict(number=best.number, params=best.params, cv_mean_delta=best.value),
                  all_trials=trials_log, wall_seconds=time.time() - t0)
    json.dump(result, open(HERE / "STAGE3B4_JOINT_RESULT.json", "w"), indent=1, default=str)
    print("BEST:", best.params, "cv_mean_delta", best.value)


if __name__ == "__main__":
    main()
