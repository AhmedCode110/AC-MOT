"""
Held-out evaluation of the systems listed in a lock file
(research/TESTDEV_LOCK.json). Runs every (system, tracker, detector,
sequence) from the verified detection cache, saves tracks and sufficient
statistics, then reports pooled metrics and paired sequence bootstraps.

Refuses to run if the code files no longer match the lock's hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

STATIC = {
    "default": dict(res=640, high=0.25, low=0.1, new=0.25),
    "shared_static_832": dict(res=832, high=0.5, low=0.1, new=0.5),
    "shared_static_736": dict(res=736, high=0.5, low=0.1, new=0.5),
}
ORACLE = {"yolov8": dict(res=832, high=0.4, low=0.1, new=0.4),
          "rtdetr": dict(res=832, high=0.5, low=0.1, new=0.5)}


def policy_for(system):
    from universal_acmot import load_policy
    from universal_policy_pipeline import POLICIES, replace
    ecdf = dict(feedback="accepted", normalizer="ecdf", policy_raw_floor=0.0)
    if system == "V3":
        from universal_acmot import V3_POLICY_FILE
        return load_policy(V3_POLICY_FILE)
    if system.startswith("V4"):
        pol, dk = load_policy()          # frozen V4 (default policy file)
        parts = system.split("_")
        pol = replace(pol, fixed_resolution=int(parts[1]))
        if "nogate" in parts:
            pol = replace(pol, gate_tau=0.0)
        return pol, dk
    table = {
        "V1": POLICIES["V1"],
        "V2b": POLICIES["V2b"],
        "V2cA": POLICIES["V2cA"],
        "universal_v1_superseded": replace(POLICIES["V1"], name="uv1",
                                           leader_rho=0.6,
                                           feedback="accepted"),
        "V3_off": replace(POLICIES["V1"], name="V3_off", **ecdf),
        "V3_ecdf_ratio06": replace(POLICIES["V1"], name="V3_r06",
                                   leader_rho=0.6, **ecdf),
    }
    return table[system], {}


def jobs_from_lock(lock, dets, seqs):
    jobs = []
    for tr, systems in lock["tracker_systems"].items():
        for s in systems:
            jobs += [(s, tr, d, q) for d in dets for q in seqs]
    return jobs


def run_one(args):
    (system, tracker_name, det, seq), dataset, cache_root, out = args
    target = Path(out) / "stats" / tracker_name / system / det / f"{seq}.pkl"
    if target.exists():
        return
    from adapters.trackers.botsort import BoTSORTAdapter
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from run_universal_acmot import build_config
    from tools.run_policy_validation import CachedDetector
    from tools.seqstats import sequence_stats
    from universal_policy_pipeline import UniversalPolicyPipeline

    cls = {"botsort": BoTSORTAdapter, "bytetrack": ByteTrackAdapter}[
        tracker_name]
    cd = CachedDetector(Path(cache_root) / det / f"{seq}.npz")
    need_img = cls.needs_image
    frames = sorted((Path(dataset) / "sequences" / seq).glob("*.jpg"))
    static = STATIC.get(system) or (ORACLE[det] if system == "oracle"
                                    else None)
    if static is None:
        cfg = build_config()
        policy, dk = policy_for(system)
        tr = cls(high=cfg.high, low=cfg.low, new=cfg.new, buffer=cfg.buffer,
                 match=cfg.match, fuse=cfg.fuse)
        pipe = UniversalPolicyPipeline(cfg, cd, tr, policy,
                                       density_kwargs=dk)
    else:
        tr = cls(high=static["high"], low=static["low"], new=static["new"],
                 buffer=30, match=0.8)
    lines = []
    blank = np.empty(cd.shape + (0,), np.uint8)
    for i in range(1, cd.frames + 1):
        cd.frame = i
        img = cv2.imread(str(frames[i - 1])) if need_img else blank
        if static is None:
            v = cd.visual[i - 1]
            tracks = pipe.process(i, img, dict(edges=v[0], brightness=v[1],
                                               blur=v[2]))["tracks"]
        else:
            tracks = tr.update(cd.detect(None, 0.01, 0.45, static["res"]),
                               cd.shape, image=img if need_img else None)
        for t in tracks:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},"
                         f"{max(0.0, t.x2 - t.x1):.3f},"
                         f"{max(0.0, t.y2 - t.y1):.3f},"
                         f"{t.confidence:.6f},{t.class_id},-1,-1\n")
    tdir = Path(out) / "tracks" / tracker_name / system / det
    tdir.mkdir(parents=True, exist_ok=True)
    tfile = tdir / f"{seq}.txt"
    tfile.write_text("".join(lines))
    st = sequence_stats(dataset, seq, tfile)
    target.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(target, "wb"))


def verify_lock(lock):
    for rel, h in lock["file_sha256"].items():
        got = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"LOCK VIOLATION: {rel} changed ({got})")


def bootstrap(stats_a, stats_b, metric, n=10000, seed=42):
    from tools.seqstats import combine
    rng = np.random.default_rng(seed)
    k = len(stats_a)
    base = combine(stats_a)[metric] - combine(stats_b)[metric]
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, k)
        diffs[i] = (combine([stats_a[j] for j in idx])[metric]
                    - combine([stats_b[j] for j in idx])[metric])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, lo, hi, float((diffs <= 0).mean())


def report(out, dets, seqs, lock, n_boot):
    from tools.seqstats import combine
    S = lambda tr, sy, d: [pickle.load(open(
        Path(out) / "stats" / tr / sy / d / f"{q}.pkl", "rb")) for q in seqs]
    rows = []
    for tr, systems in lock["tracker_systems"].items():
        for sy in systems:
            for d in dets:
                st = S(tr, sy, d)
                m = combine(st)
                ncat = sum(combine([x])["MOTA"] < 0 for x in st)
                rows.append(dict(tracker=tr, system=sy, detector=d,
                                 n_catastrophic=ncat, **m))
                print(f"{tr:<9}{sy:<24}{d:<7} MOTA {m['MOTA']:7.2f} "
                      f"HOTA {m['HOTA']:6.2f} IDF1 {m['IDF1']:6.2f} "
                      f"IDS {m['IDS']:5d} FP {m['FP']:6d} FN {m['FN']:6d} "
                      f"P {m['Precision']:5.1f} R {m['Recall']:5.1f} "
                      f"ncat {ncat}", flush=True)
    json.dump(rows, open(Path(out) / "pooled_metrics.json", "w"), indent=1)
    boots = []
    print("\nPaired sequence bootstrap: system minus baseline "
          f"({n_boot} resamples, seed 42, 95% CI)")
    for tr, sysname, base in lock["bootstrap_pairs"]:
        for d in dets:
            for metric in ("HOTA", "IDF1", "MOTA"):
                diff, lo, hi, p_le0 = bootstrap(S(tr, sysname, d),
                                                S(tr, base, d), metric,
                                                n=n_boot)
                boots.append(dict(tracker=tr, system=sysname, baseline=base,
                                  detector=d, metric=metric, diff=diff,
                                  ci_lo=lo, ci_hi=hi, p_le0=p_le0))
                print(f"  {tr:<9} {d:<7} {sysname:<10} vs {base:<18} "
                      f"{metric:<5} Δ {diff:7.2f}  CI [{lo:7.2f}, {hi:7.2f}]"
                      f"  P(Δ≤0) {p_le0:.3f}", flush=True)
    json.dump(boots, open(Path(out) / "bootstrap.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    lock = json.load(open(a.lock))
    verify_lock(lock)
    dets = lock.get("detectors_run", ["yolov8", "rtdetr"])
    seqs = sorted(p.name for p in Path(a.dataset, "sequences").iterdir()
                  if p.is_dir())
    if not a.report_only:
        jobs = jobs_from_lock(lock, dets, seqs)
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(run_one, [(j, a.dataset, a.cache_root, a.out)
                                  for j in jobs], chunksize=1))
    report(a.out, dets, seqs, lock, a.n_boot)


if __name__ == "__main__":
    main()
