"""
V5-TF family validation on the 40 development sequences (Amendment 6).
Nothing is fitted: each family is a fixed training-free rule; the
development data only VALIDATE and compare them (plus references).

  python tools/v5tf_dev.py run      # replays (both detectors, 736)
  python tools/v5tf_dev.py report   # table + declared lexicographic choice

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
               policy_raw_floor=0.0, gate_tau=0.0, scene_state=True)
FAMILIES = {
    "F1": dict(TF_BASE, candidate_mode="otsu3_window"),
    "F2": dict(TF_BASE, candidate_mode="otsu3_frame"),
    "F3": dict(TF_BASE, candidate_mode="otsu3_window", assoc_motion=True),
}
STATIC = {"static_default": dict(high=0.25, low=0.1, new=0.25),
          "shared_static": dict(high=0.5, low=0.1, new=0.5)}
ORDER = ["F1", "F2", "F3"]          # simplicity order (declared)


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
    native = system in FAMILIES
    root = "outputs/det_cache_train_native" if native else \
        "outputs/det_cache_train"
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
        ov = FAMILIES[system] if native else v4_overrides()
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


def report():
    from tools.seqstats import combine
    q = lambda m: 0.5 * (m["HOTA"] + m["IDF1"])
    systems = ["static_default", "shared_static", "V4"] + ORDER
    ref = {d: combine([load("V4", d, s) for s in DEV]) for d in DETS}
    keys = {}
    for sy in systems:
        per = {d: combine([load(sy, d, s) for s in DEV]) for d in DETS}
        ncat = sum(combine([load(sy, d, s)])["MOTA"] < 0 for d in DETS for s in DEV)
        rg = min((q(per[d]) - q(ref[d])) / q(ref[d]) for d in DETS)
        if sy in ORDER:
            keys[sy] = (-ncat, round(rg, 6), -ORDER.index(sy))
        print(f"{sy:<15} ncat {ncat:2d} worst-det rel.gain vs V4 {100 * rg:+6.2f}% | "
              + " | ".join(f"{d} MOTA {per[d]['MOTA']:6.2f} HOTA {per[d]['HOTA']:5.2f}"
                           f" IDF1 {per[d]['IDF1']:5.2f} IDS {per[d]['IDS']:5d} "
                           f"R {per[d]['Recall']:4.1f} P {per[d]['Precision']:4.1f}"
                           for d in DETS))
    choice = max(keys, key=keys.get)
    print("DECLARED-RULE FAMILY CHOICE:", choice, keys)
    json.dump(dict(choice=choice, keys=keys), open(OUT / "family_choice.json",
                                                   "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        jobs = [(sy, d, s) for sy in ["V4", "static_default", "shared_static"]
                + ORDER for d in DETS for s in DEV]
        with ProcessPoolExecutor(5) as ex:
            list(ex.map(run_one, jobs, chunksize=1))
    report()
