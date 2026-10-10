"""
V3 joint search: full adaptive AC-MOT candidate, A_t = (resolution_t,
NMS_t, detector_confidence_t), SCI thresholds searched jointly, cue
subset fixed at Stage 3A's winner. Objective vs r1536_n70 (NOT the
static matched-static point). Optuna TPE, 60-trial predeclared budget.
See PREDECLARATION_V3_FINAL.json.
"""
import json
import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from acmot_sci import SceneSpec  # noqa: E402
from research.acmot_paper_v2.controller_search.sim import (  # noqa: E402
    CALIBRATION, RESOLUTIONS, NMS_VALUES, run_policy_on_sequence, run_static_on_sequence, score_sequence,
)
from tools.v6.eval_official import combine_official  # noqa: E402
from research.acmot_paper_v2.controller_search.run_stage3a import ORIGINAL_W  # noqa: E402

HERE = Path(__file__).resolve().parent
CONF_VALUES = [0.10, 0.25, 0.40]
CUE_SUBSET = json.loads((HERE / "STAGE3A_RESULT.json").read_text())["best_trial"]["subset"]
FOLDS = [CALIBRATION[0:2], CALIBRATION[2:4], CALIBRATION[4:6], CALIBRATION[6:8]]
BASELINE_RES, BASELINE_NMS, BASELINE_CONF = 1536, 0.70, 0.0  # 0.0 = no extra filter beyond cache's own 0.01 floor


def spec_for(t_med, t_high):
    total = sum(ORIGINAL_W[c] for c in CUE_SUBSET)
    kwargs = {f"w_{c}": (ORIGINAL_W[c] / total if c in CUE_SUBSET else 0.0) for c in ORIGINAL_W}
    return SceneSpec(t_med=t_med, t_high=t_high, **kwargs)


_baseline_fold_cache = {}


def baseline_fold_hota(fold_idx):
    if fold_idx not in _baseline_fold_cache:
        stats = [score_sequence("visdrone_calib", seq,
                                 run_static_on_sequence("visdrone_calib", seq, BASELINE_RES, BASELINE_NMS, "bytetrack",
                                                         conf_floor=BASELINE_CONF))
                 for seq in FOLDS[fold_idx]]
        _baseline_fold_cache[fold_idx] = combine_official(stats)["HOTA"]
    return _baseline_fold_cache[fold_idx]


def run_policy_with_conf(split, seq, action_map, host, scene_spec):
    """action_map: {'LOW': (res,nms,conf), ...}. Thin wrapper around sim.py adding conf_floor."""
    from research.acmot_paper_v2.controller_search.sim import get_sequence, make_tracker
    from acmot_sci import SceneLayer
    import numpy as np
    sd = get_sequence(split, seq)
    scene = SceneLayer(scene_spec)
    tracker = make_tracker(host)
    out, levels_used = [], []
    for t in range(1, sd.n_frames + 1):
        decision = scene.decide(t, sd.image_stats(t))
        res, nms, conf = action_map[decision.level]
        levels_used.append(decision.level)
        dets = sd.detections(t, res, nms, conf)
        tracks = tracker.update(dets, sd.shape)
        for tr in tracks:
            out.append([t, tr.track_id, tr.x1, tr.y1, tr.x2 - tr.x1, tr.y2 - tr.y1, tr.confidence, tr.class_id, -1, -1])
        scene.observe([[x.x1, x.y1, x.x2, x.y2] for x in tracks])
    tracks_arr = np.asarray(out, dtype=float).reshape(-1, 10) if out else np.zeros((0, 10))
    return tracks_arr, levels_used


def switches_per_100(levels_used):
    if len(levels_used) < 2:
        return 0.0
    n_switch = sum(1 for i in range(1, len(levels_used)) if levels_used[i] != levels_used[i - 1])
    return 100.0 * n_switch / len(levels_used)


def evaluate_candidate(spec, action_map):
    per_seq_stats, per_seq_levels = {}, {}
    for seq in CALIBRATION:
        tr, levels = run_policy_with_conf("visdrone_calib", seq, action_map, "bytetrack", spec)
        per_seq_stats[seq] = score_sequence("visdrone_calib", seq, tr)
        per_seq_levels[seq] = levels
    per_fold_delta = []
    for i, fold in enumerate(FOLDS):
        cand_hota = combine_official([per_seq_stats[s] for s in fold])["HOTA"]
        per_fold_delta.append(cand_hota - baseline_fold_hota(i))
    cv_mean = sum(per_fold_delta) / len(per_fold_delta)
    total_frames = sum(len(l) for l in per_seq_levels.values())
    level_counts = {lv: sum(l.count(lv) for l in per_seq_levels.values()) for lv in ("LOW", "MEDIUM", "HIGH")}
    level_frac = {lv: c / total_frames for lv, c in level_counts.items()}
    switches = {seq: switches_per_100(l) for seq, l in per_seq_levels.items()}
    mean_switches = sum(switches.values()) / len(switches)
    n_actions_ge10pct = sum(1 for f in level_frac.values() if f >= 0.10)
    genuinely_adaptive = n_actions_ge10pct >= 2 and mean_switches > 0
    return dict(cv_mean=cv_mean, per_fold_delta=per_fold_delta, level_counts=level_counts, level_frac=level_frac,
                switches_per_100_frames=switches, mean_switches_per_100=mean_switches,
                n_actions_ge10pct=n_actions_ge10pct, genuinely_adaptive=genuinely_adaptive)


def main():
    t0 = time.time()
    # warm the baseline cache (prints progress)
    for i in range(4):
        print("baseline fold", i, baseline_fold_hota(i), "elapsed", round(time.time() - t0, 1))

    sampler = optuna.samplers.TPESampler(seed=43)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="v3_joint")
    trials_log = []

    def objective(trial):
        t_med = trial.suggest_float("t_med", 0.10, 0.50)
        t_high = trial.suggest_float("t_high", t_med + 0.05, min(t_med + 0.40, 0.90))
        res = {lv: trial.suggest_categorical(f"res_{lv.lower()}", RESOLUTIONS) for lv in ("LOW", "MEDIUM", "HIGH")}
        nms = {lv: trial.suggest_categorical(f"nms_{lv.lower()}", NMS_VALUES) for lv in ("LOW", "MEDIUM", "HIGH")}
        conf = {lv: trial.suggest_categorical(f"conf_{lv.lower()}", CONF_VALUES) for lv in ("LOW", "MEDIUM", "HIGH")}
        action_map = {lv: (res[lv], nms[lv], conf[lv]) for lv in ("LOW", "MEDIUM", "HIGH")}
        spec = spec_for(t_med, t_high)
        r = evaluate_candidate(spec, action_map)
        rec = dict(trial=trial.number, t_med=t_med, t_high=t_high, action_map=action_map, elapsed_s=time.time() - t0,
                   **r)
        trials_log.append(rec)
        print("trial", trial.number, "cv_mean", round(r["cv_mean"], 4), "adaptive", r["genuinely_adaptive"],
              "level_frac", {k: round(v, 3) for k, v in r["level_frac"].items()},
              "switches/100f", round(r["mean_switches_per_100"], 2), "elapsed", round(time.time() - t0, 1))
        return r["cv_mean"]

    study.optimize(objective, n_trials=60)

    eligible = [t for t in trials_log if t["genuinely_adaptive"]]
    best_eligible = max(eligible, key=lambda t: t["cv_mean"]) if eligible else None
    best_overall = max(trials_log, key=lambda t: t["cv_mean"])

    result = dict(baseline="r1536_n70", n_trials=60, sampler="TPESampler", seed=43, cue_subset=CUE_SUBSET,
                  folds=FOLDS, all_trials=trials_log, n_eligible=len(eligible),
                  best_eligible_trial=best_eligible, best_overall_trial=best_overall, wall_seconds=time.time() - t0)
    json.dump(result, open(HERE / "V3_JOINT_SEARCH_RESULT.json", "w"), indent=1, default=str)
    print("N eligible (genuinely adaptive):", len(eligible))
    print("BEST ELIGIBLE:", best_eligible)
    print("BEST OVERALL (may not be eligible):", best_overall["trial"], best_overall["cv_mean"],
          best_overall["genuinely_adaptive"])


if __name__ == "__main__":
    main()
