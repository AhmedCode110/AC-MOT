"""
STAGE 1 -- exhaustive 27-config grid over the level -> resolution mapping
(NMS fixed at 0.70 for all levels, per PREDECLARATION.json). Calibration
only. Deterministic Optuna GridSampler, full trial log, 4-fold CV.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.acmot_paper_v2.controller_search.sim import (  # noqa: E402
    CALIBRATION, RESOLUTIONS, run_policy_on_sequence, score_sequence,
)
from tools.v6.eval_official import combine_official  # noqa: E402

HERE = Path(__file__).resolve().parent
NMS_FIXED = 0.70
FOLDS = [CALIBRATION[0:2], CALIBRATION[2:4], CALIBRATION[4:6], CALIBRATION[6:8]]
MATCHED_STATIC_KEY = "r1088_n45"
ORACLE = json.loads((HERE / "ORACLE_GATE_RESULT.json").read_text())

# matched-static per-fold pooled HOTA, reused from the oracle gate's raw per-sequence stats
# (recomputed here cheaply from the already-cached per-seq pooled values is NOT exact for
# fold-pooling, so we recompute matched-static tracks once per fold too, cached below).
_matched_static_fold_cache = {}


def matched_static_fold_hota(fold_idx):
    if fold_idx not in _matched_static_fold_cache:
        stats = [score_sequence("visdrone_calib", seq,
                                 run_policy_on_sequence_static(seq)) for seq in FOLDS[fold_idx]]
        _matched_static_fold_cache[fold_idx] = combine_official(stats)["HOTA"]
    return _matched_static_fold_cache[fold_idx]


def run_policy_on_sequence_static(seq):
    from research.acmot_paper_v2.controller_search.sim import run_static_on_sequence
    return run_static_on_sequence("visdrone_calib", seq, 1088, 0.45, "bytetrack")


def evaluate_candidate(action_map):
    """Returns (cv_mean_delta, per_fold_delta, per_seq_hota)."""
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
    all_candidates = list(itertools.product(RESOLUTIONS, RESOLUTIONS, RESOLUTIONS))  # (LOW,MEDIUM,HIGH)
    search_space = {"res_low": RESOLUTIONS, "res_medium": RESOLUTIONS, "res_high": RESOLUTIONS}
    sampler = optuna.samplers.GridSampler(search_space, seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="stage1_resolution")

    trials_log = []

    def objective(trial):
        res_low = trial.suggest_categorical("res_low", RESOLUTIONS)
        res_med = trial.suggest_categorical("res_medium", RESOLUTIONS)
        res_high = trial.suggest_categorical("res_high", RESOLUTIONS)
        action_map = {"LOW": (res_low, NMS_FIXED), "MEDIUM": (res_med, NMS_FIXED), "HIGH": (res_high, NMS_FIXED)}
        cv_mean, per_fold, per_seq = evaluate_candidate(action_map)
        trials_log.append(dict(trial=trial.number, res_low=res_low, res_medium=res_med, res_high=res_high,
                                cv_mean_delta=cv_mean, per_fold_delta=per_fold, per_seq_hota=per_seq,
                                elapsed_s=time.time() - t0))
        print("trial", trial.number, "res", (res_low, res_med, res_high), "cv_mean_delta", round(cv_mean, 4),
              "elapsed", round(time.time() - t0, 1))
        return cv_mean

    n_trials = len(all_candidates)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    result = dict(
        search_space=search_space, sampler="optuna.samplers.GridSampler", seed=42, n_trials=n_trials,
        nms_fixed=NMS_FIXED, folds=FOLDS, matched_static_key=MATCHED_STATIC_KEY,
        best_trial=dict(number=best.number, params=best.params, cv_mean_delta=best.value),
        all_trials=trials_log, wall_seconds=time.time() - t0,
    )
    json.dump(result, open(HERE / "STAGE1_RESULT.json", "w"), indent=1, default=str)
    print("BEST:", best.params, "cv_mean_delta", best.value)
    print("wrote", HERE / "STAGE1_RESULT.json")


if __name__ == "__main__":
    main()
