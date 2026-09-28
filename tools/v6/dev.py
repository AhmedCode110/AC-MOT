"""
V6 development runner (Amendment 9): val-7 is the iterative development
sandbox; confirmation-16, test-dev, UAVDT, Faster R-CNN and BoT-SORT are
refused until the freeze tag exists.

  python tools/v6/dev.py run  <system> [<system> ...]   # replays, val-7
  python tools/v6/dev.py report <system> [...]          # pooled + catastrophic
  python tools/v6/dev.py seq <system> [...]             # per-sequence table

Systems are named in SYSTEMS (fixed overrides of one shared PolicySpec) or
the references V4 / shared_static / static_default. No detector name reaches
the policy: the only per-detector inputs are the cached detections.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLITS = {
    "val7": dict(data="/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val",
                 native="outputs/det_cache_val_native", v4="outputs/det_cache",
                 protected=False),
    "dev40": dict(data="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@"
                       "gmail.com/My Drive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-train",
                  native="outputs/det_cache_train_native", v4="outputs/det_cache_train",
                  protected=False),
    "conf16": dict(data="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@"
                        "gmail.com/My Drive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-train",
                   native="outputs/det_cache_train_native", v4="outputs/det_cache_train",
                   protected=True),
    "testdev": dict(data="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@"
                         "gmail.com/.shortcut-targets-by-id/1IvH3DmlX4Ce5k2cZWDvu0ixfbvxZ-67m/"
                         "visdrone goda1/VisDrone_Zips/VisDrone2019-MOT-test-dev/"
                         "VisDrone2019-MOT-test-dev",
                    native="outputs/det_cache_testdev_native", v4="outputs/det_cache_testdev",
                    protected=True),
    "uavdt": dict(data=str(ROOT / "outputs/uavdt_view"),
                  native="outputs/det_cache_uavdt_native", v4="outputs/det_cache_uavdt",
                  protected=True),
}
DETS = ["yolov8", "rtdetr"]
FREEZE_TAG = "universal-acmot-v6-freeze"


def split_sequences(split):
    if split in ("dev40", "conf16"):
        s = json.load(open(ROOT / "research/TRAIN_SPLIT_V5.json"))
        return s["development" if split == "dev40" else "confirmation"]
    return sorted(p.stem for p in (ROOT / SPLITS[split]["native"] / "yolov8").glob("*.npz")
                  if (ROOT / SPLITS[split]["native"] / "visual_cues" / p.name).exists()
                  or split == "val7")


def frozen():
    tags = subprocess.run(["git", "tag"], cwd=ROOT, capture_output=True,
                          text=True).stdout.split()
    return FREEZE_TAG in tags


def guard(split, dets):
    if (SPLITS[split]["protected"] or any(d not in DETS for d in dets)) and not frozen():
        raise SystemExit(f"PROTECTED: {split}/{dets} before {FREEZE_TAG}")


STATIC = {"static_default": dict(high=0.25, low=0.1, new=0.25),
          "shared_static": dict(high=0.5, low=0.1, new=0.5)}
TF_BASE = dict(feedback="accepted", normalizer="ecdf", scene_controller=False,
               fixed_resolution=736, nms_request=None, tracker_defaults="native",
               policy_raw_floor=0.0, gate_tau=0.0, scene_state=False)
SYSTEMS = {
    "E41": dict(TF_BASE, candidate_mode="exact3_frame", assoc_motion=True),
    "F3": dict(TF_BASE, candidate_mode="otsu3_window", assoc_motion=True),
    # X1: E41 + IoU-0.5 duplicate suppression (hypothesis H3)
    "X1": dict(TF_BASE, candidate_mode="exact3_frame", assoc_motion=True,
               dedup_iou=0.5),
    # X2: X1 without the extension-only band (hypothesis H4)
    "X2": dict(TF_BASE, candidate_mode="exact3_frame", assoc_motion=True,
               dedup_iou=0.5, tf_secondary="none"),
    # X3 / X3b: causal thresholds (exact Otsu on frames t-10..t-1) for X1 / X2
    "X3": dict(TF_BASE, candidate_mode="exact3_window", assoc_motion=True,
               dedup_iou=0.5),
    "X3b": dict(TF_BASE, candidate_mode="exact3_window", assoc_motion=True,
                dedup_iou=0.5, tf_secondary="none"),
    # X4: X3 with the jitter-calibrated extension band (hypothesis H5)
    "X4": dict(TF_BASE, candidate_mode="exact3_window", assoc_motion=True,
               dedup_iou=0.5, tf_secondary="jitter"),
    # X5*: nested (hierarchical) Otsu bands (hypothesis H6), band variants
    "X5b": dict(TF_BASE, candidate_mode="nested_window", assoc_motion=True,
                dedup_iou=0.5, tf_secondary="none"),
    "X5": dict(TF_BASE, candidate_mode="nested_window", assoc_motion=True,
               dedup_iou=0.5, tf_secondary="otsu"),
    "X5j": dict(TF_BASE, candidate_mode="nested_window", assoc_motion=True,
                dedup_iou=0.5, tf_secondary="jitter"),
}


def resolve(system):
    """'<base>[@mod...]': mod 't:<transform>' replays a monotone score
    recalibration (stress test); 'k=v' overrides one PolicySpec field."""
    base, *mods = system.split("@")
    ov = SYSTEMS.get(base)
    ov = dict(ov) if ov is not None else (
        None if base in STATIC else dict(v4_overrides()) if base == "V4" else None)
    transform = None
    for m in mods:
        if m.startswith("t:") or m.startswith("trk:"):
            if m.startswith("t:"):
                transform = m[2:]
        else:
            from dataclasses import fields
            from universal_policy_pipeline import PolicySpec
            k, v = m.split("=")
            ty = {f.name: f.type for f in fields(PolicySpec)}[k]
            ov[k] = ((lambda x: x in ("1", "True", "true")) if "bool" in str(ty)
                     else int if "int" in str(ty) else float if "float" in str(ty)
                     else str)(v)
    return ov, transform


# The frozen final policy, read from its config file (post-freeze runs).
SYSTEMS["V6TF"] = json.load(open(ROOT / "configs/universal_acmot_policy_v6tf.json"))["overrides"]


def v4_overrides():
    return json.load(open(ROOT / "configs/universal_acmot_policy_v4.json"))["overrides"]


def out_dir(split):
    return ROOT / "outputs/v6" / split


def run_one(job):
    split, system, det, seq = job
    dest = out_dir(split) / system / det / f"{seq}.pkl"
    if dest.exists():
        return
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from adapters.trackers.bytetrack import ByteTrackAdapter
    botsort = "@trk:botsort" in system
    if botsort:
        from adapters.trackers.botsort import BoTSORTAdapter as ByteTrackAdapter  # noqa: F811
    frames = sorted((Path(SPLITS[split]["data"]) / "sequences" / seq).glob("*.jpg"))
    from run_universal_acmot import build_config
    from tools.run_policy_validation import CachedDetector, make_tracker
    from tools.seqstats import sequence_stats
    from universal_policy_pipeline import POLICIES, UniversalPolicyPipeline, replace
    sp = SPLITS[split]
    ov, transform = resolve(system)
    base = system.split("@")[0]
    root = sp["v4"] if base == "V4" or base in STATIC else sp["native"]
    cd = CachedDetector(f"{root}/{det}/{seq}.npz", transform=transform)
    lines, audit = [], []

    def emit(i, tracks):
        for t in tracks:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},"
                         f"{t.x2 - t.x1:.3f},{t.y2 - t.y1:.3f},"
                         f"{t.confidence:.6f},{t.class_id},-1,-1\n")
    if system.split("@")[0] in STATIC:
        c = STATIC[system.split("@")[0]]
        tr = ByteTrackAdapter(high=c["high"], low=c["low"], new=c["new"],
                              buffer=30, match=0.8)
        for i in range(1, cd.frames + 1):
            cd.frame = i
            im = cv2.imread(str(frames[i - 1])) if botsort else None
            emit(i, tr.update(cd.detect(None, 0.01, 0.45, 736), cd.shape, image=im))
    else:
        pol = replace(POLICIES["V1"], **ov)
        cfg = build_config()
        pipe = UniversalPolicyPipeline(cfg, cd, make_tracker(cfg, pol, cls=ByteTrackAdapter), pol)
        img = np.empty(cd.shape + (0,), np.uint8)
        for i in range(1, cd.frames + 1):
            cd.frame = i
            if botsort:
                img = cv2.imread(str(frames[i - 1]))
            r = pipe.process(i, img, cd.visual_dict(i))
            emit(i, r["tracks"])
            a = r["audit"]
            audit.append({k: v for k, v in a.items()
                          if k.startswith(("tf_", "v6_")) or k in (
                              "raw_count", "accepted_after_topk", "tracks",
                              "low_threshold", "high_threshold",
                              "new_track_threshold", "demoted")})
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.writelines(lines)
    f.close()
    st = sequence_stats(sp["data"], seq, f.name)
    os.unlink(f.name)
    st["audit"] = audit
    st["tracks_txt"] = "".join(lines)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(dest, "wb"))


def load(split, system, det, seq):
    return pickle.load(open(out_dir(split) / system / det / f"{seq}.pkl", "rb"))


def verify_lock():
    """Post-freeze runs: every locked file must be byte-identical."""
    import hashlib
    lock = json.load(open(ROOT / "research/V6TF_POLICY_LOCK.json"))
    for rel, h in lock["file_sha256"].items():
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"LOCK VIOLATION: {rel} changed since the freeze")


def run(split, systems, dets=DETS, workers=6):
    guard(split, dets)
    if SPLITS[split]["protected"] or any(d not in DETS for d in dets):
        verify_lock()
        man = out_dir(split) / "RUN_MANIFEST.jsonl"
        man.parent.mkdir(parents=True, exist_ok=True)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
        import datetime
        with open(man, "a") as f:
            f.write(json.dumps(dict(utc=datetime.datetime.utcnow().isoformat(),
                                    head=head, systems=systems, dets=dets)) + "\n")
    seqs = split_sequences(split)
    jobs = [(split, sy, d, s) for sy in systems for d in dets for s in seqs]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(run_one, jobs, chunksize=1))


def summary(split, system, dets=DETS):
    from tools.seqstats import combine
    seqs = split_sequences(split)
    res = {}
    for d in dets:
        st = [load(split, system, d, s) for s in seqs]
        m = combine(st)
        per = {s: combine([x]) for s, x in zip(seqs, st)}
        gt = sum(x["counts"]["num_objects"] for x in st)
        nfr = sum(x["tracks_per_frame"] for x in st)  # not used for ratio
        trk = sum(x["counts"]["num_objects"] - x["counts"]["num_misses"]
                  + x["counts"]["num_false_positives"] for x in st)
        m["cat"] = [s for s in seqs if per[s]["MOTA"] < 0]
        m["trk_gt"] = trk / max(gt, 1)
        m["per"] = per
        res[d] = m
    return res


def report(split, systems, dets=DETS):
    q = lambda m: 0.5 * (m["HOTA"] + m["IDF1"])
    ref = summary(split, "V4", dets) if (out_dir(split) / "V4").exists() else None
    print(f"{'system':<18}{'cat':>4} | " + " | ".join(
        f"{d:>6}: MOTA  HOTA  IDF1   IDS     FP     FN   P    R  box/GT rel" for d in dets))
    for sy in systems:
        r = summary(split, sy, dets)
        ncat = sum(len(r[d]["cat"]) for d in dets)
        cells = []
        for d in dets:
            m = r[d]
            rg = 100 * (q(m) - q(ref[d])) / q(ref[d]) if ref else float("nan")
            cells.append(f"{'':>6} {m['MOTA']:5.1f} {m['HOTA']:5.1f} {m['IDF1']:5.1f} "
                         f"{m['IDS']:5d} {m['FP']:6d} {m['FN']:6d} {m['Precision']:4.1f} "
                         f"{m['Recall']:4.1f} {m['trk_gt']:5.2f} {rg:+5.1f}")
        print(f"{sy:<18}{ncat:>4} | " + " | ".join(cells))


def seq_table(split, systems, dets=DETS):
    seqs = split_sequences(split)
    res = {sy: summary(split, sy, dets) for sy in systems}
    for d in dets:
        print(f"== {d}  (MOTA / HOTA / IDF1 / P / box-per-GT)")
        print(f"{'seq':<20}" + "".join(f"{sy:>30}" for sy in systems))
        for s in seqs:
            row = ""
            for sy in systems:
                m = res[sy][d]["per"][s]
                x = load(split, sy, d, s)["counts"]
                bg = (x["num_objects"] - x["num_misses"] + x["num_false_positives"]) / max(1, x["num_objects"])
                row += f"{m['MOTA']:8.1f}{m['HOTA']:6.1f}{m['IDF1']:6.1f}{m['Precision']:5.0f}{bg:5.2f}"
            print(f"{s:<20}{row}")


def _official_one(job):
    split, system, det, seq = job
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    dest = out_dir(split) / system / det / f"{seq}.official.pkl"
    if dest.exists():
        return
    import io as _io
    from tools.v6.eval_official import official_sequence_stats
    txt = load(split, system, det, seq)["tracks_txt"]
    tr = np.loadtxt(_io.StringIO(txt), delimiter=",", ndmin=2) if txt else np.zeros((0, 10))
    pickle.dump(official_sequence_stats(SPLITS[split]["data"], seq, tr), open(dest, "wb"))


def official_summary(split, system, dets=DETS):
    from tools.v6.eval_official import combine_official
    seqs = split_sequences(split)
    res = {}
    for d in dets:
        st = [pickle.load(open(out_dir(split) / system / d / f"{s}.official.pkl", "rb"))
              for s in seqs]
        m = combine_official(st)
        m["per"] = {s: combine_official([x]) for s, x in zip(seqs, st)}
        m["cat"] = [s for s in seqs if m["per"][s]["MOTA"] < 0]
        res[d] = m
    return res


def official_report(split, systems, dets=DETS, workers=6):
    guard(split, dets)
    seqs = split_sequences(split)
    jobs = [(split, sy, d, s) for sy in systems for d in dets for s in seqs]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_official_one, jobs, chunksize=1))
    print("OFFICIAL-COMPATIBLE VisDrone-MOT (class-aware, ignored regions dropped)")
    print(f"{'system':<18}{'cat':>4} | " + " | ".join(
        f"{d:>6}: MOTA  HOTA  IDF1   IDS     FP     FN   P    R" for d in dets))
    for sy in systems:
        r = official_summary(split, sy, dets)
        print(f"{sy:<18}{sum(len(r[d]['cat']) for d in dets):>4} | " + " | ".join(
            f"{'':>6} {r[d]['MOTA']:5.1f} {r[d]['HOTA']:5.1f} {r[d]['IDF1']:5.1f} "
            f"{r[d]['IDS']:5d} {r[d]['FP']:6d} {r[d]['FN']:6d} {r[d]['Precision']:4.1f} "
            f"{r[d]['Recall']:4.1f}" for d in dets))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    split = os.environ.get("V6_SPLIT", "val7")
    dets = os.environ.get("V6_DETS", ",".join(DETS)).split(",")
    if cmd == "run":
        run(split, args, dets)
    elif cmd == "report":
        report(split, args, dets)
    elif cmd == "seq":
        seq_table(split, args, dets)
    elif cmd == "official":
        official_report(split, args, dets)
