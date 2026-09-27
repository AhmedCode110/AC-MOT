"""
V5-TF family validation on the 40 development sequences (Amendments 6-7).
Nothing is fitted: each family is a fixed training-free rule; the
development data only VALIDATE and compare them (plus references).

  python tools/v5tf_dev.py run          # replays (both detectors, budget 736)
  python tools/v5tf_dev.py report       # table + declared choice among F3/F5
  python tools/v5tf_dev.py sens         # Amendment-7 §6 constant audit replays
  python tools/v5tf_dev.py sens-report  # E28 criterion per constant value

References at matched compute (736): V4 (VisDrone-tuned, 0.45 caches),
static default (ByteTrack defaults, raw scores), shared-static (raw 0.5).
Legacy SCI (V3) needs multi-resolution caches → evaluated on val only.
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

TRAIN = ("/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@"
         "gmail.com/My Drive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-train")
SPLIT = json.load(open("research/TRAIN_SPLIT_V5.json"))
DEV = SPLIT["development"]
DETS = ["yolov8", "rtdetr"]
OUT = Path("outputs/v5tf_dev")
TF_BASE = dict(feedback="accepted", normalizer="ecdf", scene_controller=False,
               fixed_resolution=736, nms_request=None, tracker_defaults="native",
               policy_raw_floor=0.0, gate_tau=0.0, scene_state=False)
FAMILIES = {
    "F1": dict(TF_BASE, candidate_mode="otsu3_window"),
    "F2": dict(TF_BASE, candidate_mode="otsu3_frame"),
    "F3": dict(TF_BASE, candidate_mode="otsu3_window", assoc_motion=True),
    "F5": dict(TF_BASE, candidate_mode="otsu3_window", assoc_motion=True,
               res_policy="size_tertile"),
    "F5R": dict(TF_BASE, candidate_mode="otsu3_window", assoc_motion=True,
                res_policy="random3"),
}
FAMILIES["E41"] = dict(TF_BASE, candidate_mode="exact3_frame",
                        assoc_motion=True)
BUDGET = 736
SENS = {"otsu_bins": [32, 128], "gate_window": [5, 20],
        "tf_history": [50, 200], "tf_warmup": [3, 10]}   # defaults 64/10/100/5
STATIC = {"static_default": dict(high=0.25, low=0.1, new=0.25),
          "shared_static": dict(high=0.5, low=0.1, new=0.5)}
ORDER = ["F1", "F2", "F3", "F5", "F5R", "E41"]
SELECTABLE = ["F3", "F5"]                  # Amendment 7 §5 (scene-state control)


def family_overrides(system):
    """Family name, or sensitivity variant 'S:<family>:<field>=<value>'."""
    if system.startswith("S:"):
        _, fam, kv = system.split(":")
        k, v = kv.split("=")
        return dict(FAMILIES[fam], **{k: int(v)})
    return FAMILIES.get(system)


def v4_overrides():
    return json.load(open("configs/universal_acmot_policy_v4.json"))["overrides"]


def run_one(job):
    system, det, seq = job
    dest = OUT / system / det / f"{seq}.pkl"
    if dest.exists():
        return
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from run_universal_acmot import build_config
    from tools.run_policy_validation import CachedDetector, make_tracker
    from tools.seqstats import sequence_stats
    from universal_policy_pipeline import (POLICIES, UniversalPolicyPipeline,
                                           replace)
    ov_tf = family_overrides(system)
    native = ov_tf is not None
    root = ("outputs/det_cache_train_res" if native and ov_tf.get("res_policy")
            else "outputs/det_cache_train_native" if native
            else "outputs/det_cache_train")
    cd = CachedDetector(f"{root}/{det}/{seq}.npz")
    lines, audit = [], []

    def emit(i, tracks):
        for t in tracks:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},"
                         f"{t.x2 - t.x1:.3f},{t.y2 - t.y1:.3f},"
                         f"{t.confidence:.6f},{t.class_id},-1,-1\n")
    if system in STATIC:
        c = STATIC[system]
        tr = ByteTrackAdapter(high=c["high"], low=c["low"], new=c["new"],
                              buffer=30, match=0.8)
        for i in range(1, cd.frames + 1):
            cd.frame = i
            emit(i, tr.update(cd.detect(None, 0.01, 0.45, 736), cd.shape))
    else:
        ov = ov_tf if native else v4_overrides()
        pol = replace(POLICIES["V1"], **ov)
        cfg = build_config()
        pipe = UniversalPolicyPipeline(cfg, cd, make_tracker(cfg, pol), pol)
        img = np.empty(cd.shape + (0,), np.uint8)
        for i in range(1, cd.frames + 1):
            cd.frame = i
            r = pipe.process(i, img, cd.visual_dict(i))
            emit(i, r["tracks"])
            audit.append({k: v for k, v in r["audit"].items()
                          if k.startswith(("tf_", "s_"))})
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.writelines(lines)
    f.close()
    st = sequence_stats(TRAIN, seq, f.name)
    os.unlink(f.name)
    st["audit"] = audit
    dest.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(dest, "wb"))


def load(system, det, seq):
    return pickle.load(open(OUT / system / det / f"{seq}.pkl", "rb"))


def pixel_cost(system, det):
    """Mean per-frame pixel cost relative to BUDGET**2 (1.0 = fixed 736)."""
    lv = [a.get("tf_level", BUDGET) for s in DEV for a in load(system, det, s)["audit"]]
    return float(np.mean(np.square(lv)) / BUDGET ** 2) if lv else 1.0


def report():
    from tools.seqstats import combine
    q = lambda m: 0.5 * (m["HOTA"] + m["IDF1"])
    systems = ["static_default", "shared_static", "V4"] + ORDER
    ref = {d: combine([load("V4", d, s) for s in DEV]) for d in DETS}
    keys, eligible, costs = {}, {}, {}
    for sy in systems:
        per = {d: combine([load(sy, d, s) for s in DEV]) for d in DETS}
        ncat = sum(combine([load(sy, d, s)])["MOTA"] < 0 for d in DETS for s in DEV)
        rg = min((q(per[d]) - q(ref[d])) / q(ref[d]) for d in DETS)
        cost = {d: pixel_cost(sy, d) for d in DETS} if sy in ORDER else \
            {d: 1.0 for d in DETS}
        if sy in SELECTABLE:
            keys[sy] = (-ncat, round(rg, 6), -SELECTABLE.index(sy))
            eligible[sy] = bool(max(cost.values()) <= 1.01)   # Amendment 7 §5
            costs[sy] = cost
        print(f"{sy:<15} ncat {ncat:2d} worst-det rel.gain vs V4 {100 * rg:+6.2f}% "
              f"cost {'/'.join(f'{cost[d]:.3f}' for d in DETS)} | "
              + " | ".join(f"{d} MOTA {per[d]['MOTA']:6.2f} HOTA {per[d]['HOTA']:5.2f}"
                           f" IDF1 {per[d]['IDF1']:5.2f} IDS {per[d]['IDS']:5d} "
                           f"R {per[d]['Recall']:4.1f} P {per[d]['Precision']:4.1f}"
                           for d in DETS))
    choice = max((k for k in keys if eligible[k]), key=keys.get)
    print("DECLARED-RULE FAMILY CHOICE (selectable F3/F5):", choice, keys)
    json.dump(dict(choice=choice, keys=keys, eligible=eligible, pixel_cost=costs),
              open(OUT / "family_choice.json",
                                                   "w"), indent=1)


def sens_systems():
    fam = json.load(open(OUT / "family_choice.json"))["choice"]
    return fam, [f"S:{fam}:{k}={v}" for k, vals in SENS.items() for v in vals]


def sens_report():
    """Amendment 7 §6 / E28 criterion: pooled |dHOTA| <= 0.4 per detector and
    no additional catastrophic cell -> structural (A); else the declared
    default is kept and reported as sensitive (E). Never re-selected."""
    from tools.seqstats import combine
    fam, variants = sens_systems()
    base = {d: combine([load(fam, d, s) for s in DEV]) for d in DETS}
    ncat0 = sum(combine([load(fam, d, s)])["MOTA"] < 0 for d in DETS for s in DEV)
    verdict = {}
    for sy in variants:
        k = sy.split(":")[2].split("=")[0]
        per = {d: combine([load(sy, d, s) for s in DEV]) for d in DETS}
        ncat = sum(combine([load(sy, d, s)])["MOTA"] < 0 for d in DETS for s in DEV)
        dh = {d: per[d]["HOTA"] - base[d]["HOTA"] for d in DETS}
        ok = all(abs(v) <= 0.4 for v in dh.values()) and ncat <= ncat0
        verdict[k] = verdict.get(k, True) and ok
        print(f"{sy:<32} dHOTA " + " ".join(f"{d} {dh[d]:+.2f}" for d in DETS)
              + f" ncat {ncat} (base {ncat0}) {'OK' if ok else 'SENSITIVE'}")
    out = {k: ("A (insensitive)" if v else "E (sensitive; default kept)")
           for k, v in verdict.items()}
    print("CONSTANT AUDIT:", out)
    json.dump(dict(family=fam, verdict=out), open(OUT / "constant_audit.json", "w"),
              indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("run", "sens"):
        systems = (sens_systems()[1] if cmd == "sens" else
                   ["V4", "static_default", "shared_static"] + ORDER)
        jobs = [(sy, d, s) for sy in systems for d in DETS for s in DEV]
        with ProcessPoolExecutor(5) as ex:
            list(ex.map(run_one, jobs, chunksize=1))
    if cmd in ("sens", "sens-report"):
        sens_report()
    else:
        report()
