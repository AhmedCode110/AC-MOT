"""
Cross-detector, cross-sequence global parameter selection.

Stage 1 (`run`): evaluate every grid config on every (detector, sequence)
by cache replay; store per-sequence sufficient statistics.

Stage 2 (`select`): leave-one-sequence-out (LOSO) CV. For each held-out
sequence, choose the config maximizing objective J on the other 6
sequences with BOTH detectors, then score it on the held-out sequence.
Also reports the config chosen on all 7 sequences (the final choice).

The grid, objectives and constraint are fixed in this file BEFORE the run
(see research/OPTIMIZATION_PROTOCOL.md).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATASET = "/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val"
DETECTORS = ["yolov8", "rtdetr"]
OUT = Path("outputs/opt" if os.environ.get("ACMOT_GRID", "v1") == "v1"
           else f"outputs/opt_{os.environ['ACMOT_GRID']}")

RHOS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
KAPPAS = [0.5, 1.0, 2.0]


FAMILY = os.environ.get("ACMOT_GRID", "v1")
TAUS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def grid():
    if FAMILY == "v3":
        v3 = dict(feedback="accepted", normalizer="ecdf", gate_stat="zlogit",
                  policy_raw_floor=0.0)
        cfgs = [dict(id="v3_off", base="V1", selectable=False,
                     overrides=dict(v3, gate_tau=0.0), density_kwargs={})]
        cfgs += [dict(id=f"v3_t{t:.2f}", base="V1", selectable=True,
                      overrides=dict(v3, gate_tau=t), density_kwargs={})
                 for t in TAUS]
        cfgs += [dict(id=f"abl_ecdf_ratio_r{r:.1f}", base="V1",
                      selectable=False,
                      overrides=dict(feedback="accepted", normalizer="ecdf",
                                     leader_rho=r, policy_raw_floor=0.0),
                      density_kwargs={}) for r in (0.5, 0.6, 0.7)]
        return cfgs
    cfgs = []
    for rho in RHOS:
        cfgs.append(dict(id=f"gate_r{rho:.1f}", base="V1",
                         overrides=dict(leader_rho=rho,
                                        feedback="accepted"),
                         density_kwargs={}))
    for rho, k in itertools.product(RHOS, KAPPAS):
        cfgs.append(dict(
            id=f"dens_k{k:.1f}_r{rho:.1f}", base="V2cA",
            overrides=dict(leader_rho=rho),
            density_kwargs=dict(base_budget=12.0 * k, sci_budget=36.0 * k,
                                min_budget=int(round(12 * k)),
                                max_budget=int(round(48 * k)))))
    return cfgs


def run_one(args):
    cfg, det, seq = args
    target = OUT / "stats" / cfg["id"] / det / f"{seq}.pkl"
    if target.exists():
        return str(target)
    import tempfile

    from run_universal_acmot import build_config
    from tools.run_policy_validation import CachedDetector, make_tracker
    from tools.seqstats import sequence_stats
    from universal_policy_pipeline import (POLICIES, UniversalPolicyPipeline,
                                           replace)

    policy = replace(POLICIES[cfg["base"]], name=cfg["id"],
                     **cfg["overrides"])
    det_cache = CachedDetector(f"outputs/det_cache/{det}/{seq}.npz")
    cfg_ac = build_config()
    pipe = UniversalPolicyPipeline(cfg_ac, det_cache, make_tracker(cfg_ac),
                                   policy,
                                   density_kwargs=cfg["density_kwargs"])
    image = np.empty(det_cache.shape + (0,), dtype=np.uint8)
    lines, audit = [], []
    for i in range(1, det_cache.frames + 1):
        det_cache.frame = i
        v = det_cache.visual[i - 1]
        res = pipe.process(i, image, dict(edges=v[0], brightness=v[1],
                                          blur=v[2]))
        for t in res["tracks"]:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},"
                         f"{max(0.0, t.x2 - t.x1):.3f},"
                         f"{max(0.0, t.y2 - t.y1):.3f},"
                         f"{t.confidence:.6f},{t.class_id},-1,-1\n")
        audit.append(res["audit"])
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.writelines(lines)
        tmp = f.name
    st = sequence_stats(DATASET, seq, tmp)
    os.unlink(tmp)
    st["mean_sci"] = float(np.mean([a["sci"] for a in audit]))
    raw = np.array([a["raw_count"] for a in audit], float)
    acc = np.array([a["accepted_after_topk"] for a in audit], float)
    st["keep_pct"] = float(np.mean(np.where(raw > 0, 100 * acc /
                                            np.maximum(raw, 1), 0)))
    target.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(target, "wb"))
    return str(target)


def sequences():
    return sorted(p.name for p in Path(DATASET, "sequences").iterdir()
                  if p.is_dir())


def stage_run(workers):
    jobs = [(c, d, s) for c in grid() for d in DETECTORS for s in sequences()]
    json.dump(grid(), open(OUT / "grid.json", "w"), indent=1)
    with ProcessPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(run_one, jobs, chunksize=1), 1):
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)


# ---------------------------------------------------------------- selection
def load_all():
    from tools.seqstats import combine  # noqa: F401
    S = {}
    for c in grid():
        for d in DETECTORS:
            for s in sequences():
                S[(c["id"], d, s)] = pickle.load(
                    open(OUT / "stats" / c["id"] / d / f"{s}.pkl", "rb"))
    return S


def metrics_on(S, cid, det, seqs, root=None):
    from tools.seqstats import combine
    if root is not None:   # reference V1 always from the v1 grid
        return combine([pickle.load(open(Path(root) / "stats" / cid / det /
                                          f"{s}.pkl", "rb")) for s in seqs])
    return combine([S[(cid, det, s)] for s in seqs])


def objective(name, per_det, ref):
    """per_det: {det: metrics}; ref: {det: V1 metrics on same seqs}."""
    q = {d: 0.5 * (m["HOTA"] + m["IDF1"]) for d, m in per_det.items()}
    if name == "J1_mean":
        return float(np.mean(list(q.values())))
    if name == "J2_worst":
        return float(min(q.values()))
    if name == "J3_rel_gain":
        return float(np.mean([(q[d] - 0.5 * (ref[d]["HOTA"] + ref[d]["IDF1"]))
                              / (0.5 * (ref[d]["HOTA"] + ref[d]["IDF1"]))
                              for d in q]))
    if name == "J4_worst_rel_gain":
        return float(min((q[d] - 0.5 * (ref[d]["HOTA"] + ref[d]["IDF1"]))
                         / (0.5 * (ref[d]["HOTA"] + ref[d]["IDF1"]))
                         for d in q))
    raise ValueError(name)


def feasible(per_det):
    return all(m["MOTA"] > 0 for m in per_det.values())


def n_catastrophic(S, cid, seqs):
    return sum(metrics_on(S, cid, d, [s])["MOTA"] < 0
               for d in DETECTORS for s in seqs)


def choose(S, seqs, obj):
    lexi = obj.startswith("L_")
    base_obj = obj[2:] if lexi else obj
    ref = {d: metrics_on(S, "gate_r0.0", d, seqs, "outputs/opt") for d in DETECTORS}
    best, best_key = None, None
    for c in grid():
        if not c.get("selectable", True):
            continue
        per = {d: metrics_on(S, c["id"], d, seqs) for d in DETECTORS}
        if not feasible(per):
            continue
        v = objective(base_obj, per, ref)
        key = (-n_catastrophic(S, c["id"], seqs), v) if lexi else (v,)
        if best_key is None or key > best_key:
            best, best_key = c["id"], key
    return best, best_key[-1]


def stage_select():
    S = load_all()
    seqs = sequences()
    report = {}
    for obj in ["J2_worst", "J1_mean", "J3_rel_gain", "J4_worst_rel_gain",
                "L_J4_worst_rel_gain", "L_J3_rel_gain"]:
        folds = []
        for held in seqs:
            train = [s for s in seqs if s != held]
            cid, val = choose(S, train, obj)
            folds.append(dict(held_out=held, chosen=cid, train_obj=val))
        # CV estimate: held-out sequences scored with their fold's choice,
        # pooled over the 7 held-out sequences (exact protocol pooling).
        from tools.seqstats import combine
        cv = {d: combine([S[(f["chosen"], d, f["held_out"])]
                          for f in folds]) for d in DETECTORS}
        final, final_val = choose(S, seqs, obj)
        report[obj] = dict(folds=folds, cv_pooled=cv, final=final,
                           final_train_obj=final_val,
                           final_all7={d: metrics_on(S, final, d, seqs)
                                       for d in DETECTORS})
    json.dump(report, open(OUT / "selection_report.json", "w"), indent=1,
              default=float)
    for obj, r in report.items():
        print(f"== {obj}: final={r['final']}  "
              f"fold choices={[f['chosen'] for f in r['folds']]}")
        for d in DETECTORS:
            cv, fa = r["cv_pooled"][d], r["final_all7"][d]
            print(f"   {d:<7} LOSO-CV  MOTA {cv['MOTA']:7.2f} HOTA {cv['HOTA']:6.2f}"
                  f" IDF1 {cv['IDF1']:6.2f} IDS {cv['IDS']:5d} | final-on-7 "
                  f"MOTA {fa['MOTA']:7.2f} HOTA {fa['HOTA']:6.2f} IDF1 {fa['IDF1']:6.2f}")


def stage_surface():
    """Full response surface on all 7 sequences (for the paper)."""
    S = load_all()
    seqs = sequences()
    rows = []
    for c in grid():
        row = dict(config=c["id"])
        for d in DETECTORS:
            m = metrics_on(S, c["id"], d, seqs)
            for k in ("MOTA", "HOTA", "IDF1", "IDS", "FP", "FN",
                      "Precision", "Recall"):
                row[f"{d}_{k}"] = m[k]
            row[f"{d}_worst_seq_HOTA"] = min(
                metrics_on(S, c["id"], d, [s])["HOTA"] for s in seqs)
        rows.append(row)
    import csv
    with open(OUT / "response_surface.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['config']:<16} " + " | ".join(
            f"{d} MOTA {r[d + '_MOTA']:7.2f} HOTA {r[d + '_HOTA']:5.2f} "
            f"IDF1 {r[d + '_IDF1']:5.2f} worstSeqHOTA {r[d + '_worst_seq_HOTA']:5.2f}"
            for d in DETECTORS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "select", "surface"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"run": lambda: stage_run(a.workers), "select": stage_select,
     "surface": stage_surface}[a.stage]()
