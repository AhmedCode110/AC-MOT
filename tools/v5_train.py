"""
V5 on VisDrone-MOT-train, exactly as declared in Amendment 5d.

  python tools/v5_train.py s1         # fixed-value runs + headroom (development)
  python tools/v5_train.py s2         # cue utility, 5-fold sequence CV (development)
  python tools/v5_train.py s3         # C1/C2(/C3) outer 5-fold, end-to-end replays
  python tools/v5_train.py final      # fit chosen family on all 40 -> configs/..._v5.json
  python tools/v5_train.py confirm    # ONCE, after the freeze commit: 16 sequences

GT is used offline only (costs); the controller uses causal state only.
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scene_state import ALL_CUES

TRAIN = ("/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@"
         "gmail.com/My Drive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-train")
CACHE = "outputs/det_cache_train"
OUT = Path("outputs/v5_train")
SPLIT = json.load(open("research/TRAIN_SPLIT_V5.json"))
DEV, CONF = SPLIT["development"], SPLIT["confirmation"]
DETS = ["yolov8", "rtdetr"]
WINDOW = 30
TARGETS = {
    "sensitivity": ("fixed_sensitivity", [0.3, 0.4, 0.5, 0.6], 0.4),
    "assoc_offset": ("assoc_offset", [0.0, 0.05, 0.10, 0.18, 0.26], 0.10),
    "gate_tau": ("gate_tau", [0.5, 0.75, 1.0, 1.25, 1.5], 0.75),
    "retention": ("tracker_buffer", [15, 30, 45, 60, 90], 45),
}
HEADROOM_MIN = 0.5
N_NULL_S2, N_NULL_S3 = 200, 100
MIN_LEAF = 0.10


def folds5(seqs, seed=1):
    """Seeded, length-stratified assignment of sequences to 5 folds."""
    rng = np.random.default_rng(seed)
    order = sorted(seqs, key=lambda s: SPLIT["frames"][s])
    folds = [[] for _ in range(5)]
    for b in range(0, len(order), 5):
        block = order[b:b + 5]
        for k, s in zip(rng.permutation(5), block):
            folds[k].append(s)
    return folds


# ------------------------------------------------------------------ replays
def v4_overrides():
    o = json.load(open("configs/universal_acmot_policy_v4.json"))["overrides"]
    o["scene_state"] = True
    return o


def replay(job):
    """job = (tag, overrides_dict, det, seq). Stores stats + per-frame
    events (gt,tp,fp,fn,idsw,idtp,idfp,idfn) + logged states."""
    tag, ov, det, seq = job
    dest = OUT / "runs" / tag / det / f"{seq}.pkl"
    if dest.exists():
        return
    from run_universal_acmot import build_config
    from tools.run_policy_validation import CachedDetector, make_tracker
    from tools.seqstats import sequence_stats
    from tools.v5_s1_headroom import per_frame_events_ds
    from universal_policy_pipeline import (POLICIES, UniversalPolicyPipeline,
                                           replace)
    pol = replace(POLICIES["V1"], **ov)
    cd = CachedDetector(f"{CACHE}/{det}/{seq}.npz")
    cfg = build_config()
    pipe = UniversalPolicyPipeline(cfg, cd, make_tracker(cfg, pol), pol)
    img = np.empty(cd.shape + (0,), np.uint8)
    lines, states = [], []
    for i in range(1, cd.frames + 1):
        cd.frame = i
        r = pipe.process(i, img, cd.visual_dict(i))
        states.append({k[2:]: v for k, v in r["audit"].items()
                       if k.startswith("s_")})
        for t in r["tracks"]:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},"
                         f"{t.x2 - t.x1:.3f},{t.y2 - t.y1:.3f},"
                         f"{t.confidence:.6f},{t.class_id},-1,-1\n")
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.writelines(lines)
    f.close()
    st = sequence_stats(TRAIN, seq, f.name)
    st["frame_events"] = per_frame_events_ds(TRAIN, seq, f.name, cd.frames)
    st["states"] = states
    os.unlink(f.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(dest, "wb"))


def run_jobs(jobs, workers=5):
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(replay, jobs, chunksize=1))


def load(tag, det, seq):
    return pickle.load(open(OUT / "runs" / tag / det / f"{seq}.pkl", "rb"))


def vtag(target, value):
    return f"s1_{target}_{value}"


# ------------------------------------------------------------------ S1
def stage_s1():
    jobs = []
    for t, (field, values, _) in TARGETS.items():
        for v in values:
            jobs += [(vtag(t, v), dict(v4_overrides(), **{field: v}), d, s)
                     for d in DETS for s in DEV]
    run_jobs(jobs)
    res = {}
    for t, (_, values, _) in TARGETS.items():
        res[t] = {}
        for d in DETS:
            err = np.array([[cost(load(vtag(t, v), d, s)["frame_events"]).sum()
                             for s in DEV] for v in values])     # values x seq
            gt = sum(load(vtag(t, values[0]), d, s)["frame_events"][:, 0].sum()
                     for s in DEV)
            glob = err.sum(1).min()
            res[t][d] = dict(per_sequence=100 * (glob - err.min(0).sum()) / gt)
        res[t]["candidate"] = all(res[t][d]["per_sequence"] >= HEADROOM_MIN
                                  for d in DETS)
        print(f"S1 {t:<13} per-sequence headroom yolo "
              f"{res[t]['yolov8']['per_sequence']:.2f} rtdetr "
              f"{res[t]['rtdetr']['per_sequence']:.2f} -> "
              f"{'CANDIDATE' if res[t]['candidate'] else 'global'}")
    json.dump(res, open(OUT / "s1_headroom.json", "w"), indent=1, default=float)


def cost(ev):
    """(FP+FN) + (IDFP+IDFN) per frame (Amendment 5b)."""
    return ev[:, 2] + ev[:, 3] + ev[:, 6] + ev[:, 7]


# ------------------------------------------------------------------ windows
def windows(target, seqs):
    """rows: (det, seq, cue_vector, regret_vector, gt) per 30-frame window;
    features = state at the window's first frame in the V4 reference run."""
    _, values, v4v = TARGETS[target]
    rows = []
    for d, s in itertools.product(DETS, seqs):
        ev = [load(vtag(target, v), d, s)["frame_events"] for v in values]
        ref = load(vtag(target, v4v), d, s)["states"]
        n = len(ev[0])
        for w in range(int(np.ceil(n / WINDOW))):
            sl = slice(w * WINDOW, (w + 1) * WINDOW)
            err = np.array([cost(e[sl]).sum() for e in ev], dtype=float)
            rows.append((d, s, np.array([ref[w * WINDOW].get(c, np.nan)
                                         for c in ALL_CUES], float),
                         err - err.min(), ev[0][sl, 0].sum()))
    return values, rows


def arrays(rows):
    return (np.array([r[2] for r in rows]), np.array([r[3] for r in rows]),
            np.array([r[1] for r in rows]), np.array([r[0] for r in rows]),
            np.array([r[4] for r in rows]))


def fit_stump(x, R, min_leaf=0):
    ok = np.isfinite(x)
    base = R.sum(0)
    best = (np.inf, int(np.argmin(base)), int(np.argmin(base)), base.min())
    if ok.sum() < 10:
        return best
    for th in np.unique(np.quantile(x[ok], np.linspace(0.05, 0.95, 19))):
        le = (~ok) | (x <= th)
        if min(le.sum(), (~le).sum()) < min_leaf:
            continue
        cl, cg = R[le].sum(0), R[~le].sum(0)
        c = cl.min() + cg.min()
        if c < best[3]:
            best = (th, int(np.argmin(cl)), int(np.argmin(cg)), c)
    return best


def cv_gain(X, R, S, D, G, j, fold_lists, rng=None):
    x = X[:, j] if rng is None else rng.permutation(X[:, j])
    gain = {d: 0.0 for d in DETS}
    gts = {d: 0.0 for d in DETS}
    for fold in fold_lists:
        te = np.isin(S, fold)
        tr = ~te
        th, a, b, _ = fit_stump(x[tr], R[tr])
        glob = int(np.argmin(R[tr].sum(0)))
        pred = np.where(np.isfinite(x[te]) & (x[te] > th), b, a)
        idx = np.where(te)[0]
        for d in DETS:
            m = D[idx] == d
            gain[d] += R[idx[m], glob].sum() - R[idx[m], pred[m]].sum()
            gts[d] += G[idx[m]].sum()
    per = {d: 100 * gain[d] / max(gts[d], 1) for d in DETS}
    return 100 * sum(gain.values()) / max(sum(gts.values()), 1), per


def eligible_cues(target, seqs, fold_lists, n_null, seed):
    values, rows = windows(target, seqs)
    X, R, S, D, G = arrays(rows)
    rng = np.random.default_rng(seed)
    out = []
    for j, c in enumerate(ALL_CUES):
        g, per = cv_gain(X, R, S, D, G, j, fold_lists)
        if g <= 0 or min(per.values()) <= 0:
            continue
        null = [cv_gain(X, R, S, D, G, j, fold_lists, rng)[0]
                for _ in range(n_null)]
        if g > np.percentile(null, 95):
            out.append((c, g, per))
    return sorted(out, key=lambda x: -x[1]), values, (X, R, S, D, G)


# ------------------------------------------------------------------ S2
def stage_s2():
    s1 = json.load(open(OUT / "s1_headroom.json"))
    cand = [t for t in TARGETS if s1[t]["candidate"]]
    folds = folds5(DEV)
    rep = {}
    for t in cand:
        el, _, _ = eligible_cues(t, DEV, folds, N_NULL_S2, seed=0)
        rep[t] = [dict(cue=c, gain=g, per_detector=p) for c, g, p in el]
        print(f"S2 {t:<13} eligible: " + (", ".join(
            f"{c} {g:+.2f} (y {p['yolov8']:+.2f}, r {p['rtdetr']:+.2f})"
            for c, g, p in el) or "none"))
    json.dump(dict(candidates=cand, eligible=rep),
              open(OUT / "s2_cues.json", "w"), indent=1, default=float)


# ------------------------------------------------------------------ S3
def fit_controller(family, target, seqs, inner_folds, seed):
    """Returns (node or None, log)."""
    el, values, (X, R, S, D, G) = eligible_cues(target, seqs, inner_folds,
                                               N_NULL_S3, seed)
    if not el:
        return None, None
    names = [c for c, _, _ in el]
    if family == "C3":
        node = fit_c3(target, names, values, X, R,
                      study_name=f"v5_C3_{target}_{seed}")
        return node, dict(cues=names, root=names[0])
    j = ALL_CUES.index(names[0])
    th, a, b, _ = fit_stump(X[:, j], R)
    node = {"feature": names[0], "threshold": float(th),
            "le": values[a], "gt": values[b]}
    if family == "C2":
        min_leaf = int(MIN_LEAF * len(R))
        x = X[:, j]
        le = ~np.isfinite(x) | (x <= th)
        for side, mask in (("le", le), ("gt", ~le)):
            best = None
            for c in names:
                jj = ALL_CUES.index(c)
                st = fit_stump(X[mask, jj], R[mask], min_leaf)
                if best is None or st[3] < best[1][3]:
                    best = (c, st)
            c, (t2, a2, b2, cst) = best
            if cst < R[mask].sum(0).min() - 1e-9:
                node[side] = {"feature": c, "threshold": float(t2),
                              "le": values[a2], "gt": values[b2]}
    return node, dict(cues=names, root=names[0])


OPTUNA_DB = "sqlite:///outputs/v5/optuna_v5.db"
C3_TRIALS = 150


def fit_c3(target, names, values, X, R, study_name):
    """C3 (Amendment 5d/5e): linear scene score over the eligible cues.
    Search space (logged in the study's user attrs): per cue weight in
    [0, 1] and orientation {+, -}; threshold in [0, 1]; low/high value
    indices over the target's grid. Cues are normalised by 21 quantile
    knots of the TRAINING windows. Objective: total training-window regret
    (same cost as C1/C2). TPE sampler, seed 0, SQLite storage."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Path("outputs/v5").mkdir(parents=True, exist_ok=True)
    cols = [ALL_CUES.index(c) for c in names]
    knots = {c: np.nanquantile(X[:, j], np.linspace(0, 1, 21)).tolist()
             for c, j in zip(names, cols)}
    U = np.column_stack([np.interp(X[:, j], knots[c], np.linspace(0, 1, 21))
                         for c, j in zip(names, cols)])
    valid = np.isfinite(X[:, cols])

    def score(p):
        w = np.array([p[f"w_{c}"] for c in names])
        inv = np.array([p[f"inv_{c}"] for c in names], bool)
        V = np.where(inv, 1 - U, U)
        W = np.where(valid, w, 0.0)
        den = W.sum(1)
        z = np.where(den > 0, (np.where(valid, V, 0) * W).sum(1) /
                     np.maximum(den, 1e-12), -1.0)
        pred = np.where(z > p["threshold"], p["hi"], p["lo"])
        return float(R[np.arange(len(R)), pred].sum())

    def objective(trial):
        p = {f"w_{c}": trial.suggest_float(f"w_{c}", 0.0, 1.0) for c in names}
        p.update({f"inv_{c}": trial.suggest_categorical(f"inv_{c}",
                                                        [False, True])
                  for c in names})
        p["threshold"] = trial.suggest_float("threshold", 0.0, 1.0)
        p["lo"] = trial.suggest_int("lo", 0, len(values) - 1)
        p["hi"] = trial.suggest_int("hi", 0, len(values) - 1)
        return score(p)

    study = optuna.create_study(
        study_name=study_name, storage=OPTUNA_DB, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0), load_if_exists=True)
    study.set_user_attr("search_space", dict(
        cues=names, weights="[0,1]", orientation="{False,True}",
        threshold="[0,1]", lo_hi=f"value index 0..{len(values) - 1}",
        values=values, objective="training-window regret (FP+FN)+(IDFP+IDFN)"))
    remaining = C3_TRIALS - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    b = study.best_params
    return {"type": "linear", "threshold": b["threshold"],
            "le": values[b["lo"]], "gt": values[b["hi"]],
            "terms": [{"feature": c, "weight": b[f"w_{c}"],
                       "invert": b[f"inv_{c}"], "knots": knots[c]}
                      for c in names]}


def spec_ov(spec):
    ov = v4_overrides()
    ov["controller_spec"] = json.dumps({"targets": spec})
    return ov


def stage_s3():
    s2 = json.load(open(OUT / "s2_cues.json"))
    cand = s2["candidates"]
    outer = folds5(DEV)
    families = ["C1", "C2"]
    try:
        import optuna  # noqa: F401  (C3 only if installed before S3)
        families.append("C3")
    except ImportError:
        pass
    record = {f: [] for f in families}
    jobs = [("v4_ref", v4_overrides(), d, s) for d in DETS for s in DEV]
    for k, fold in enumerate(outer):
        train = [s for s in DEV if s not in fold]
        inner = folds5(train, seed=100 + k)[:4] if len(train) >= 20 else \
            [[s] for s in train]
        for fam in families:
            spec, logs = {}, {}
            for t in cand:
                node, lg = fit_controller(fam, t, train, inner, seed=k)
                if node is not None:
                    spec[TARGETS[t][0]] = node
                logs[t] = lg
            record[fam].append(dict(fold=k, held_out=fold, spec=spec,
                                    log=logs))
            jobs += [(f"s3_{fam}_fold{k}", spec_ov(spec), d, s)
                     for d in DETS for s in fold]
    run_jobs(jobs)
    json.dump(record, open(OUT / "s3_record.json", "w"), indent=1,
              default=float)
    from tools.seqstats import combine
    ref = {d: combine([load("v4_ref", d, s) for s in DEV]) for d in DETS}
    for fam in families:
        per = {d: combine([load(f"s3_{fam}_fold{r['fold']}", d, s)
                           for r in record[fam] for s in r["held_out"]])
               for d in DETS}
        ncat = sum(combine([load(f"s3_{fam}_fold{r['fold']}", d, s)])["MOTA"]
                   < 0 for r in record[fam] for s in r["held_out"]
                   for d in DETS)
        ncat4 = sum(combine([load("v4_ref", d, s)])["MOTA"] < 0
                    for s in DEV for d in DETS)
        q = lambda m: 0.5 * (m["HOTA"] + m["IDF1"])
        j4 = min((q(per[d]) - q(ref[d])) / q(ref[d]) for d in DETS)
        print(f"S3 {fam}: ncat {ncat} (V4 {ncat4})  worst-det rel gain "
              f"{100 * j4:+.2f}%  | " + " | ".join(
                  f"{d} ΔHOTA {per[d]['HOTA'] - ref[d]['HOTA']:+.2f} ΔIDF1 "
                  f"{per[d]['IDF1'] - ref[d]['IDF1']:+.2f} ΔMOTA "
                  f"{per[d]['MOTA'] - ref[d]['MOTA']:+.2f}" for d in DETS))
        for t in cand:
            print(f"     {t:<13} root cue per outer fold: " + " ".join(
                (r["log"][t]["root"] if r["log"].get(t) else "-")
                for r in record[fam]))


def choose_family():
    """Lexicographic choice from the S3 outer-fold results (Amendment 5d):
    (1) fewest catastrophic cells, (2) worst-detector relative gain of
    ½(HOTA+IDF1) over V4, (3) lower complexity (C1 before C2)."""
    from tools.seqstats import combine
    record = json.load(open(OUT / "s3_record.json"))
    q = lambda m: 0.5 * (m["HOTA"] + m["IDF1"])
    ref = {d: combine([load("v4_ref", d, s) for s in DEV]) for d in DETS}
    keys = {}
    for i, fam in enumerate([f for f in ("C1", "C2", "C3") if f in record]):
        cells = [(d, s, f"s3_{fam}_fold{r['fold']}") for r in record[fam]
                 for s in r["held_out"] for d in DETS]
        ncat = sum(combine([load(t, d, s)])["MOTA"] < 0 for d, s, t in cells)
        per = {d: combine([load(t, dd, s) for dd, s, t in cells if dd == d])
               for d in DETS}
        j4 = min((q(per[d]) - q(ref[d])) / q(ref[d]) for d in DETS)
        keys[fam] = (-ncat, round(j4, 6), -i)
    return max(keys, key=keys.get), keys, record


def stage_final():
    fam, keys, record = choose_family()
    cand = json.load(open(OUT / "s2_cues.json"))["candidates"]
    spec, logs = {}, {}
    for t in cand:
        node, lg = fit_controller(fam, t, DEV, folds5(DEV), seed=999)
        # stability rule: adapt only if the root cue agrees in the majority
        roots = [r["log"][t]["root"] if r["log"].get(t) else None
                 for r in record[fam]]
        if node is not None and lg and roots.count(lg["root"]) * 2 > len(roots):
            spec[TARGETS[t][0]] = node
        logs[t] = dict(fit=lg, outer_roots=roots,
                       adapted=TARGETS[t][0] in spec)
    ov = v4_overrides()
    ov.update(name="UniversalACMOT_v5", controller_spec=json.dumps(
        {"targets": spec}))
    ov.pop("scene_state", None)
    cfg = dict(name="UniversalACMOT_v5_frozen", base="V1",
               selected_by=f"Amendment 5d; family {fam}; S3 keys {keys}",
               overrides=ov, density_kwargs={}, targets_log=logs)
    Path("configs/universal_acmot_policy_v5.json").write_text(
        json.dumps(cfg, indent=1, default=float))
    print("family", fam, keys)
    print(json.dumps(spec, indent=1, default=float))


def stage_confirm():
    """ONCE, after the V5 freeze commit/tag."""
    from tools.seqstats import combine
    v5 = json.load(open("configs/universal_acmot_policy_v5.json"))["overrides"]
    v5 = dict(v5, scene_state=True)
    jobs = [("confirm_v4", v4_overrides(), d, s) for d in DETS for s in CONF]
    jobs += [("confirm_v5", v5, d, s) for d in DETS for s in CONF]
    run_jobs(jobs)
    q = lambda m: 0.5 * (m["HOTA"] + m["IDF1"])
    ok_cat, ok_q, out = True, True, {}
    rng = np.random.default_rng(42)
    for d in DETS:
        a = [load("confirm_v5", d, s) for s in CONF]
        b = [load("confirm_v4", d, s) for s in CONF]
        ma, mb = combine(a), combine(b)
        ca = sum(combine([x])["MOTA"] < 0 for x in a)
        cb = sum(combine([x])["MOTA"] < 0 for x in b)
        diffs = []
        for _ in range(10000):
            idx = rng.integers(0, len(CONF), len(CONF))
            diffs.append(q(combine([a[i] for i in idx]))
                         - q(combine([b[i] for i in idx])))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        ok_cat &= ca <= cb
        ok_q &= q(ma) >= q(mb)
        out[d] = dict(v5=ma, v4=mb, ncat_v5=ca, ncat_v4=cb,
                      dq=q(ma) - q(mb), ci=[lo, hi],
                      p_le0=float(np.mean(np.asarray(diffs) <= 0)))
        print(f"{d}: V4 MOTA {mb['MOTA']:.2f} HOTA {mb['HOTA']:.2f} IDF1 "
              f"{mb['IDF1']:.2f} ncat {cb} | V5 MOTA {ma['MOTA']:.2f} HOTA "
              f"{ma['HOTA']:.2f} IDF1 {ma['IDF1']:.2f} ncat {ca} | Δq "
              f"{q(ma) - q(mb):+.2f} CI [{lo:+.2f}, {hi:+.2f}]")
    out["adopted"] = bool(ok_cat and ok_q)
    print("V5 ADOPTED" if out["adopted"] else "V5 NOT ADOPTED")
    json.dump(out, open(OUT / "confirmation_result.json", "w"), indent=1,
              default=float)


if __name__ == "__main__":
    {"s1": stage_s1, "s2": stage_s2, "s3": stage_s3, "final": stage_final,
     "confirm": stage_confirm}[sys.argv[1]]()
