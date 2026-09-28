"""
V7 development runner for the VisDrone/UAVDT systems (ByteTrack / BoT-SORT
hosts on cached detector outputs). Evaluation code is shared (read-only)
with the V6 runner so every number is comparable to the V6 record.

  python tools/v7/dev.py run  <system> [<system> ...]
  python tools/v7/dev.py report <system> [...]     # pooled, internal protocol
  python tools/v7/dev.py official <system> [...]   # official-compatible
  python tools/v7/dev.py seq <system> [...]
  python tools/v7/dev.py track <system> [...]      # label-free: tracks + audit only

A system is '<base>[@mod...]':
  base  a key of SYSTEMS (fixed V7Spec overrides) or 'V6:<name>' to read a
        V6 run from outputs/v6/<split>/<name> (reference only)
  mods  'k=v'        override one V7Spec field
        't:<name>'   monotone score transform (stress test)
        'floor=<f>'  emission floor applied after the transform (stress test)
        'trk:botsort' BoT-SORT host instead of ByteTrack
Env: V7_SPLIT (val7 | dev40 | conf16 | testdev | uavdt), V7_DETS.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.v6.dev import SPLITS as _V6_SPLITS, split_sequences as _v6_split_sequences  # noqa: E402  (read-only reuse)


def split_sequences(split):
    ex = set(filter(None, os.environ.get("V7_EXCLUDE", "").split(",")))   # declared host-failure subsets only
    return [s for s in _split_sequences(split) if s not in ex]


def _split_sequences(split):
    if split in _V6_SPLITS:
        return _v6_split_sequences(split)
    nat = ROOT / SPLITS[split]["native"]
    return sorted(p.stem for p in (nat / "yolov8").glob("*.npz")
                  if (nat / "visual_cues" / p.name).exists())

# Dataset roots may be relocated (cloud / Codex) through environment variables;
# defaults are the original Mac paths recorded in the V6 runner.
SPLITS = {k: dict(v) for k, v in _V6_SPLITS.items()}
for _k, _env in (("val7", "ACMOT_VISDRONE_VAL"), ("dev40", "ACMOT_VISDRONE_TRAIN"),
                 ("conf16", "ACMOT_VISDRONE_TRAIN"), ("testdev", "ACMOT_VISDRONE_TESTDEV"),
                 ("uavdt", "ACMOT_UAVDT_VIEW")):
    if os.environ.get(_env):
        SPLITS[_k]["data"] = os.environ[_env]
# KITTI tracking TRAINING split (cloud fallback development data, 21 sequences):
# a frame view (sequences/<seq>/<frame>.jpg -> the official PNG frames) and
# native caches built with the original recipe. Evaluation: tools/v7/kitti/kitti_eval.py.
SPLITS["kitti"] = dict(data=os.environ.get("ACMOT_KITTI_VIEW",
                                           str(Path.home() / "acmot_work/kitti/view")),
                       native="outputs/det_cache_kitti_native", protected=False)
V7_FREEZE_TAG = "universal-acmot-v7-freeze"


def _code_sha():
    import hashlib
    return hashlib.sha256((ROOT / "acmot_v7.py").read_bytes()).hexdigest()


def guard(split):
    """confirmation-16 is reserved for ONE post-freeze V7 check."""
    if split == "conf16":
        import subprocess
        tags = subprocess.run(["git", "tag"], cwd=ROOT, capture_output=True, text=True).stdout.split()
        if V7_FREEZE_TAG not in tags:
            raise SystemExit("PROTECTED: conf16 is reserved until " + V7_FREEZE_TAG)

DETS = ["yolov8", "rtdetr"]
from tools.v7.systems import HOST_BYTETRACK, SYSTEMS, parse  # noqa: E402,F401


def out_dir(split):
    return ROOT / "outputs/v7" / split


def track_sequence(split, system, det, seq):
    """Layer + host tracker on one cached sequence (no labels). Returns the
    MOT-format track lines and the per-frame audit (layer log + host
    output size)."""
    os.chdir(ROOT)
    import cv2
    from acmot_v7 import HostContract, V7Layer, spec_from_dict
    from adapters.types import Detection
    from tools.run_policy_validation import CachedDetector
    base, ov, tf, floor, botsort = parse(system)
    ocsort = "@trk:ocsort" in system
    if botsort:
        from adapters.trackers.botsort import BoTSORTAdapter as Trk
    else:
        from adapters.trackers.bytetrack import ByteTrackAdapter as Trk
    sp = SPLITS[split]
    cd = CachedDetector(f"{sp['native']}/{det}/{seq}.npz", transform=tf)
    frames = sorted((Path(sp["data"]) / "sequences" / seq).glob("*.jpg"))
    if botsort and len(frames) != cd.frames:
        raise SystemExit(f"{seq}: BoT-SORT needs the real frames ({len(frames)} *.jpg in "
                         f"{sp['data']}, cache has {cd.frames})")
    h = HOST_BYTETRACK
    if ocsort:
        h = HOST_OCSORT
        tr = _ocsort_host(h)
    else:
        tr = Trk(high=h["assoc"], low=h["low"], new=h["birth"], buffer=30, match=h["match"], fuse=True)
    layer = V7Layer(spec_from_dict(dict(ov, name=base)), HostContract(**h))
    lines, audit = [], []
    for i in range(1, cd.frames + 1):
        cd.frame = i
        raw = cd.detect(None, 0.0, None, 736)
        if floor is not None:
            raw = [d for d in raw if d.confidence >= floor]
        b = np.array([[d.x1, d.y1, d.x2, d.y2] for d in raw]).reshape(-1, 4)
        s = np.array([d.confidence for d in raw])
        vis = cd.visual_dict(i)
        dec = layer.step(b, s, vis.get("motion"), classes=[d.class_id for d in raw])
        dets = [Detection(x1=raw[k].x1, y1=raw[k].y1, x2=raw[k].x2, y2=raw[k].y2,
                          confidence=float(v), class_id=raw[k].class_id)
                for k, v in zip(dec.keep, dec.scores)]
        tr.set_association_tolerance(dec.match)
        if ocsort:
            tracks = tr.update([[x.x1, x.y1, x.x2, x.y2, x.confidence] for x in dets],
                               cd.shape, min(dec.assoc, dec.birth))
            _ocsort_classes(tracks, dets, tr)
        else:
            img = cv2.imread(str(frames[i - 1])) if botsort else np.empty(cd.shape + (0,), np.uint8)
            tracks = tr.update(dets, cd.shape, association_threshold=dec.assoc,
                               birth_threshold=dec.birth, image=img if botsort else None)
        layer.observe([[t.x1, t.y1, t.x2, t.y2] for t in tracks], [t.track_id for t in tracks])
        for t in tracks:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},{t.x2 - t.x1:.3f},"
                         f"{t.y2 - t.y1:.3f},{t.confidence:.6f},{t.class_id},-1,-1\n")
        audit.append(dict(dec.log, tracks=len(tracks)))
    return lines, audit, cd.frames


# OC-SORT (CVPR 2023, noahcao/OC_SORT @ 8462e7e) official arguments: single
# threshold 0.6 (association = birth; no low stage -> low = 0.6), IoU 0.3.
HOST_OCSORT = dict(assoc=0.6, birth=0.6, low=0.6, match=0.7)


def _ocsort_host(h):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mot17_host_v7", ROOT / "tools/v7/external/mot17_bytetrack_v7.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    t = m.OCSortHost(h)
    t.cls = {}
    return t


def _ocsort_classes(tracks, dets, tr):
    """OC-SORT outputs no class: a track takes the class of the passed
    detection it reproduces (IoU >= 0.5), else keeps its last class."""
    from acmot_v7 import iou_matrix
    if not tracks:
        return
    M = iou_matrix([[t.x1, t.y1, t.x2, t.y2] for t in tracks],
                   [[d.x1, d.y1, d.x2, d.y2] for d in dets]) if dets else None
    for i, t in enumerate(tracks):
        if M is not None and M.shape[1] and M[i].max() >= 0.5:
            tr.cls[t.track_id] = dets[int(M[i].argmax())].class_id
        t.class_id = tr.cls.get(t.track_id, -1)


def run_one(job):
    split, system, det, seq = job
    dest = out_dir(split) / system / det / f"{seq}.pkl"
    base, ov, tf, floor, botsort = parse(system)
    stamp = dict(code_sha=_code_sha(), spec=dict(ov, name=base), transform=tf, floor=floor,
                 botsort=botsort)
    if dest.exists():
        try:
            old = pickle.load(open(dest, "rb")).get("v7_stamp")
        except Exception:
            old = None
        if old == stamp:
            return                      # identical code + spec: reuse
    sp = SPLITS[split]
    frames = sorted((Path(sp["data"]) / "sequences" / seq).glob("*.jpg"))
    lines, audit, n = track_sequence(split, system, det, seq)
    if len(frames) != n:
        raise SystemExit(f"{seq}: {len(frames)} *.jpg files in {sp['data']} but the cache has "
                         f"{n} frames (the evaluator counts *.jpg files; see "
                         "scripts/setup_research_assets.sh for placeholder frames)")
    from tools.seqstats import sequence_stats
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.writelines(lines)
    f.close()
    st = sequence_stats(sp["data"], seq, f.name)
    os.unlink(f.name)
    st["audit"] = audit
    st["tracks_txt"] = "".join(lines)
    st["v7_stamp"] = stamp
    dest.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(dest, "wb"))


def track_one(job):
    """Label-free run (no evaluation): outputs/v7/<split>/<system>/<det>/<seq>.trk.pkl"""
    split, system, det, seq = job
    dest = out_dir(split) / system / det / f"{seq}.trk.pkl"
    base, ov, tf, floor, botsort = parse(system)
    stamp = dict(code_sha=_code_sha(), spec=dict(ov, name=base), transform=tf, floor=floor,
                 botsort=botsort)
    if dest.exists() and pickle.load(open(dest, "rb")).get("v7_stamp") == stamp:
        return
    lines, audit, _ = track_sequence(split, system, det, seq)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(audit=audit, tracks_txt="".join(lines), v7_stamp=stamp), open(dest, "wb"))


def track(split, systems, dets=DETS, workers=int(os.environ.get("V7_WORKERS", 6))):
    guard(split)
    seqs = split_sequences(split)
    jobs = [(split, sy, d, s) for sy in systems for d in dets for s in seqs]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(track_one, jobs, chunksize=1))


def _path(split, system, det, seq, suffix=".pkl"):
    if system.startswith("V6:"):
        return ROOT / "outputs/v6" / split / system[3:] / det / f"{seq}{suffix}"
    return out_dir(split) / system / det / f"{seq}{suffix}"


def load(split, system, det, seq):
    return pickle.load(open(_path(split, system, det, seq), "rb"))


def run(split, systems, dets=DETS, workers=int(os.environ.get("V7_WORKERS", 6))):
    guard(split)
    seqs = split_sequences(split)
    jobs = [(split, sy, d, s) for sy in systems if not sy.startswith("V6:")
            for d in dets for s in seqs]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(run_one, jobs, chunksize=1))


def summary(split, system, dets=DETS):
    from tools.seqstats import combine
    seqs = split_sequences(split)
    res = {}
    for d in dets:
        st = [load(split, system, d, s) for s in seqs]
        m = combine(st)
        m["per"] = {s: combine([x]) for s, x in zip(seqs, st)}
        m["cat"] = [s for s in seqs if m["per"][s]["MOTA"] < 0]
        aud = [a for x in st for a in x.get("audit", [])]
        m["clean_frac"] = (np.mean([a.get("regime") == "clean" for a in aud])
                           if aud and "regime" in aud[0] else float("nan"))
        res[d] = m
    return res


def report(split, systems, dets=DETS):
    print(f"{'system':<34}{'cat':>4} | " + " | ".join(
        f"{d:>10}: MOTA  HOTA  IDF1   IDS     FP     FN    P    R clean" for d in dets))
    for sy in systems:
        r = summary(split, sy, dets)
        print(f"{sy:<34}{sum(len(r[d]['cat']) for d in dets):>4} | " + " | ".join(
            f"{'':>10} {r[d]['MOTA']:5.1f} {r[d]['HOTA']:5.1f} {r[d]['IDF1']:5.1f} {r[d]['IDS']:5d} "
            f"{r[d]['FP']:6d} {r[d]['FN']:6d} {r[d]['Precision']:4.1f} {r[d]['Recall']:4.1f} "
            f"{r[d]['clean_frac']:5.2f}" for d in dets))


def _official_one(job):
    split, system, det, seq = job
    os.chdir(ROOT)
    dest = _path(split, system, det, seq, ".official.pkl")
    if dest.exists():
        return
    from tools.v6.eval_official import official_sequence_stats
    txt = load(split, system, det, seq)["tracks_txt"]
    tr = np.loadtxt(io.StringIO(txt), delimiter=",", ndmin=2) if txt else np.zeros((0, 10))
    pickle.dump(official_sequence_stats(SPLITS[split]["data"], seq, tr), open(dest, "wb"))


def official_summary(split, system, dets=DETS):
    from tools.v6.eval_official import combine_official
    seqs = split_sequences(split)
    res = {}
    for d in dets:
        st = [pickle.load(open(_path(split, system, d, s, ".official.pkl"), "rb")) for s in seqs]
        m = combine_official(st)
        m["per"] = {s: combine_official([x]) for s, x in zip(seqs, st)}
        m["cat"] = [s for s in seqs if m["per"][s]["MOTA"] < 0]
        res[d] = m
    return res


def official_report(split, systems, dets=DETS, workers=int(os.environ.get("V7_WORKERS", 6))):
    guard(split)
    seqs = split_sequences(split)
    jobs = [(split, sy, d, s) for sy in systems for d in dets for s in seqs]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_official_one, jobs, chunksize=1))
    print("OFFICIAL-COMPATIBLE VisDrone-MOT (class-aware, ignored regions dropped)")
    print(f"{'system':<34}{'cat':>4} | " + " | ".join(
        f"{d:>10}: MOTA  HOTA  IDF1   IDS     FP     FN    P    R" for d in dets))
    for sy in systems:
        r = official_summary(split, sy, dets)
        print(f"{sy:<34}{sum(len(r[d]['cat']) for d in dets):>4} | " + " | ".join(
            f"{'':>10} {r[d]['MOTA']:5.1f} {r[d]['HOTA']:5.1f} {r[d]['IDF1']:5.1f} {r[d]['IDS']:5d} "
            f"{r[d]['FP']:6d} {r[d]['FN']:6d} {r[d]['Precision']:4.1f} {r[d]['Recall']:4.1f}" for d in dets))


def seq_table(split, systems, dets=DETS):
    seqs = split_sequences(split)
    res = {sy: summary(split, sy, dets) for sy in systems}
    for d in dets:
        print(f"== {d}  (MOTA / HOTA / IDF1 per system)")
        print(f"{'seq':<20}" + "".join(f"{sy[:22]:>24}" for sy in systems))
        for s in seqs:
            print(f"{s:<20}" + "".join(
                f"{res[sy][d]['per'][s]['MOTA']:8.1f}{res[sy][d]['per'][s]['HOTA']:8.1f}"
                f"{res[sy][d]['per'][s]['IDF1']:8.1f}" for sy in systems))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    split = os.environ.get("V7_SPLIT", "val7")
    dets = os.environ.get("V7_DETS", ",".join(DETS)).split(",")
    {"run": run, "track": track, "report": report, "official": official_report, "seq": seq_table}[cmd](split, args, dets)
