"""
V5 stage S3 — nested sequence-level selection of the scene-adaptive
controller (Amendment 5a). VisDrone val only; YOLOv8n + RT-DETR-L jointly.

For every outer held-out sequence h:
  * windows of the 6 training sequences (both detectors), per target;
  * per cue: inner-LOSO cv_gain + permutation null (100); eligible iff
    cv_gain > null95 and gain > 0 for both detectors; best eligible cue
    per target, cost-sensitive stump fitted on the 6 sequences;
  * end-to-end replay of that controller on h (both detectors).
Variants restrict the allowed cue families (ablations).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scene_state import ALL_CUES, FAMILIES
from tools.v5_s1_headroom import DETS, TARGETS
from tools.v5_s2_cues import apply, build, fit_stump

DATASET = "/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val"
OUT = Path(os.environ.get("V5_S3_OUT", "outputs/v5/s3"))
ADAPT = os.environ.get("V5_ADAPT", "sensitivity,gate_tau,assoc_offset").split(",")
FIELD = {"sensitivity": "fixed_sensitivity", "gate_tau": "gate_tau",
         "assoc_offset": "assoc_offset", "resolution": "resolution"}
V4_VALUE = {"sensitivity": 0.4, "gate_tau": 0.75, "assoc_offset": 0.10,
            "resolution": 736}
VARIANTS = {
    "full": ALL_CUES,
    "image_only": FAMILIES["image"],
    "detector_only": FAMILIES["detector"],
    "tracker_only": FAMILIES["tracker"],
    "minus_image": FAMILIES["detector"] + FAMILIES["tracker"],
    "minus_detector": FAMILIES["image"] + FAMILIES["tracker"],
    "minus_tracker": FAMILIES["image"] + FAMILIES["detector"],
}
N_PERM = 100


def cv_on(rows, seqs, j, rng=None):
    X = np.array([r[2][j] for r in rows])
    if rng is not None:
        X = rng.permutation(X)
    R = np.array([r[3] for r in rows])
    S = np.array([r[1] for r in rows])
    D = np.array([r[0] for r in rows])
    G = np.array([r[4] for r in rows])
    gain = {d: 0.0 for d in DETS}
    gts = {d: 0.0 for d in DETS}
    for held in seqs:
        tr, te = S != held, S == held
        th, a, b, _ = fit_stump(X[tr], R[tr])
        glob = int(np.argmin(R[tr].sum(0)))
        pred = apply(th, a, b, X[te])
        idx = np.where(te)[0]
        for d in DETS:
            m = D[idx] == d
            gain[d] += R[idx[m], glob].sum() - R[idx[m], pred[m]].sum()
            gts[d] += G[idx[m]].sum()
    per = {d: 100 * gain[d] / max(gts[d], 1) for d in DETS}
    return 100 * sum(gain.values()) / max(sum(gts.values()), 1), per


def select(target_rows, train_seqs, allowed, seed):
    """Inner selection on train_seqs. Returns spec targets + log."""
    rng = np.random.default_rng(seed)
    spec, log = {}, {}
    for target in ADAPT:
        values, rows = target_rows[target]
        rows = [r for r in rows if r[1] in train_seqs]
        best = None
        for cue in allowed:
            j = ALL_CUES.index(cue)
            g, per = cv_on(rows, train_seqs, j)
            if g <= 0 or min(per.values()) <= 0:
                continue
            null = [cv_on(rows, train_seqs, j, rng)[0] for _ in range(N_PERM)]
            if g > np.percentile(null, 95) and (best is None or g > best[1]):
                best = (cue, g, per)
        if best is None:
            spec[FIELD[target]] = V4_VALUE[target]
            log[target] = None
            continue
        cue = best[0]
        j = ALL_CUES.index(cue)
        X = np.array([r[2][j] for r in rows])
        R = np.array([r[3] for r in rows])
        th, a, b, _ = fit_stump(X, R)
        spec[FIELD[target]] = {"feature": cue, "threshold": float(th),
                               "le": values[a], "gt": values[b]}
        log[target] = dict(cue=cue, inner_gain=best[1], per_detector=best[2],
                           threshold=float(th), le=values[a], gt=values[b])
    spec["resolution"] = 736
    return spec, log


def replay(args):
    spec, det, seq, dest = args
    if Path(dest).exists():
        return pickle.load(open(dest, "rb"))
    from run_universal_acmot import build_config
    from tools.run_policy_validation import CachedDetector, make_tracker
    from tools.seqstats import sequence_stats
    from universal_policy_pipeline import (POLICIES, UniversalPolicyPipeline,
                                           replace)
    o = json.load(open("configs/universal_acmot_policy_v4.json"))["overrides"]
    o.update(scene_state=True, controller_spec=json.dumps({"targets": spec}))
    pol = replace(POLICIES["V1"], **o)
    cd = CachedDetector(f"outputs/det_cache/{det}/{seq}.npz")
    cfg = build_config()
    pipe = UniversalPolicyPipeline(cfg, cd, make_tracker(cfg, pol), pol)
    img = np.empty(cd.shape + (0,), np.uint8)
    lines, decisions = [], []
    for i in range(1, cd.frames + 1):
        cd.frame = i
        r = pipe.process(i, img, cd.visual_dict(i))
        decisions.append({k[2:]: v for k, v in r["audit"].items()
                          if k.startswith("d_")})
        for t in r["tracks"]:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},"
                         f"{t.x2 - t.x1:.3f},{t.y2 - t.y1:.3f},"
                         f"{t.confidence:.6f},{t.class_id},-1,-1\n")
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.writelines(lines)
    f.close()
    st = sequence_stats(DATASET, seq, f.name)
    os.unlink(f.name)
    st["decisions"] = decisions
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(dest, "wb"))
    return st


def main(variants):
    seqs = sorted(p.name for p in Path(DATASET, "sequences").iterdir()
                  if p.is_dir())
    target_rows = {t: build(t) for t in ADAPT}
    for vname in variants:
        allowed = VARIANTS[vname]
        folds, jobs = [], []
        for k, h in enumerate(seqs):
            train = [s for s in seqs if s != h]
            spec, log = select(target_rows, train, allowed, seed=k)
            folds.append(dict(held_out=h, spec=spec, log=log))
            jobs += [(spec, d, h, str(OUT / vname / "outer" / d / f"{h}.pkl"))
                     for d in DETS]
        final_spec, final_log = select(target_rows, seqs, allowed, seed=99)
        jobs += [(final_spec, d, s, str(OUT / vname / "final" / d / f"{s}.pkl"))
                 for d in DETS for s in seqs]
        with ProcessPoolExecutor(5) as ex:
            list(ex.map(replay, jobs, chunksize=1))
        json.dump(dict(folds=folds, final_spec=final_spec,
                       final_log=final_log),
                  open(OUT / vname / "selection.json", "w"), indent=1,
                  default=float)
        print(f"== {vname}: cue per target per outer fold")
        for t in ADAPT:
            print(f"   {t:<13} " + " ".join(
                (f["log"][t]["cue"] if f["log"][t] else "-") for f in folds)
                + f"  | final: {final_log[t]['cue'] if final_log[t] else '-'}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARIANTS))
