"""
SCI + V7f development runner on VisDrone2019-MOT-val (val-7) ONLY.

Protected splits (confirmation-16, test-dev, train, UAVDT, any other) are
refused before any file is opened: see guard().

  python tools/sci_v7/dev.py run     <system> [...]
  python tools/sci_v7/dev.py report  <system> [...]   internal protocol
  python tools/sci_v7/dev.py official <system> [...]  official-compatible
  python tools/sci_v7/dev.py seq     <system> [...]
  python tools/sci_v7/dev.py ops     <system> [...]   compute levels, V7f regime, interventions
  python tools/sci_v7/dev.py boot    <A> <B> [--protocol internal|official] [--json f]

A system is '<layer>+<levels>':
  layer   NATIVE (host pass-through) or V7f (frozen record, tools/v7/systems.py)
  levels  SCI            scene layer (acmot_sci.SceneLayer)
          LOW|MEDIUM|HIGH one fixed level
          ORACLE|ORACLEM GT-derived headroom diagnostics (tools/sci_v7/oracle.py),
                         never a controller
          PERM<k>        budget-matched scene-blind control: the level
                         schedule of '<layer>+SCI' on the same sequence and
                         detector, cut into 30-frame segments whose order is
                         shuffled with seed k (identical level counts, so
                         identical compute; scene dependence removed)
Env: V7_DETS (default yolov8,rtdetr), V7_WORKERS, ACMOT_VISDRONE_VAL.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ALLOWED_SPLITS = ("val7",)
SPLIT = "val7"
DETS = ["yolov8", "rtdetr"]
SEGMENT = 30
CODE = ["acmot_sci.py", "acmot_sci_v7.py", "acmot_v7.py", "adapters/detectors/compute_profile.py",
        "configs/sci_v7_profiles.json", "tools/v7/systems.py", "tools/sci_v7/dev.py",
        "tools/sci_v7/sweep_cache.py", "tools/sci_v7/hosts.py"]
# Detection source and level profile (defaults = the E-SCI-1 setting):
#   SCI_CACHE=native  V7-record cache (640/736/832)
#   SCI_CACHE=sweep   resolution-sweep cache under SCI_SWEEP_ROOT
#   SCI_PROFILE=<key> level profile key in configs/sci_v7_profiles.json
CACHE = os.environ.get("SCI_CACHE", "native")
PROFILE = os.environ.get("SCI_PROFILE", "resolution")


def guard(split):
    """Development split only. Raises before any data path is touched."""
    if split not in ALLOWED_SPLITS:
        raise SystemExit(f"PROTECTED: split '{split}' is not allowed in the SCI + V7f cycle "
                         f"(allowed: {ALLOWED_SPLITS}); confirmation-16 and test-dev need the "
                         "owner's explicit authorization")


guard(os.environ.get("V7_SPLIT", SPLIT))


def _splits():
    from tools.v7.dev import SPLITS, split_sequences
    return SPLITS, split_sequences


def code_sha():
    h = hashlib.sha256()
    for f in CODE:
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()


def parse(system):
    """'<layer>+<levels>[@<host>]' -> (layer, V7Spec overrides, levels); host via host_of()."""
    layer, levels = system.split("@")[0].split("+")
    from tools.v7.systems import SYSTEMS
    if layer not in ("NATIVE", "V7f"):
        raise ValueError(layer)
    if not (levels in ("SCI", "LOW", "MEDIUM", "HIGH", "ORACLE", "ORACLEM") or
            (levels.startswith("ORACLEB") and levels[7:].isdigit()) or
            (levels.startswith("PERM") and levels[4:].isdigit()) or
            (levels.startswith("R") and levels[1:].isdigit())):
        raise ValueError(levels)
    return layer, dict(SYSTEMS[layer]), levels


def host_of(system):
    from tools.sci_v7.hosts import HOSTS
    h = system.split("@")[1] if "@" in system else "bytetrack"
    if h not in HOSTS:
        raise ValueError(h)
    return h


def out_dir():
    if CACHE == "native" and PROFILE == "resolution":
        return ROOT / "outputs/sci_v7" / SPLIT
    return ROOT / "outputs/sci_v7" / f"{SPLIT}__{CACHE}__{PROFILE}"


def open_cache(det, seq):
    SPLITS, _ = _splits()
    nat = SPLITS[SPLIT]["native"]
    if CACHE == "sweep":
        from tools.sci_v7.sweep_cache import SweepCache
        return SweepCache(os.environ["SCI_SWEEP_ROOT"], det, seq, f"{nat}/visual_cues/{seq}.npz")
    from tools.run_policy_validation import CachedDetector
    return CachedDetector(f"{nat}/{det}/{seq}.npz")


def _path(system, det, seq, suffix=".pkl"):
    return out_dir() / system / det / f"{seq}{suffix}"


def load(system, det, seq):
    return pickle.load(open(_path(system, det, seq), "rb"))


def perm_schedule(layer, seed, det, seq, host=""):
    src = [a["level"] for a in load(f"{layer}+SCI" + host, det, seq)["audit"]]
    segs = [src[i:i + SEGMENT] for i in range(0, len(src), SEGMENT)]
    rng = np.random.default_rng([seed, zlib.crc32(seq.encode()), zlib.crc32(det.encode())])
    out = [lv for k in rng.permutation(len(segs)) for lv in segs[k]]
    assert sorted(out) == sorted(src)
    return out


class _ResolutionSchedule:
    """Diagnostic adapter: the schedule already holds native resolutions."""
    def __init__(self, supported):
        self.supported = supported

    def resolution(self, level):
        if int(level) not in self.supported:
            raise ValueError(level)
        return int(level)


def track_sequence(system, det, seq, trace=False):
    """Scene layer + adapter + frozen V7f + tracker host on one cached sequence."""
    os.chdir(ROOT)
    from acmot_sci import SceneLayer
    from acmot_sci_v7 import LevelSource, SciV7Pipeline
    from acmot_v7 import HostContract, V7Layer, spec_from_dict
    from adapters.detectors.compute_profile import ComputeProfileAdapter
    from adapters.types import Detection
    from tools.sci_v7.hosts import make_host
    layer, spec, levels = parse(system)
    host = host_of(system)
    tracker, h, needs_image = make_host(host)
    sfx = "" if host == "bytetrack" else "@" + host
    cd = open_cache(det, seq)
    if levels.startswith("R") and levels[1:].isdigit():      # one fixed native resolution
        r = int(levels[1:])
        adapter = ComputeProfileAdapter({"LOW": r, "MEDIUM": r, "HIGH": r}, supported=set(cd.by_res))
        levels = "MEDIUM"
    else:
        adapter = ComputeProfileAdapter.from_config(det, supported=set(cd.by_res), key=PROFILE)
    if levels == "SCI":
        src = LevelSource(scene=SceneLayer())
    elif levels.startswith("PERM"):
        src = LevelSource(schedule=perm_schedule(layer, int(levels[4:]), det, seq, sfx))
    elif levels.startswith("ORACLE"):     # GT-derived diagnostic schedule (tools/sci_v7/oracle.py)
        sched = json.loads((out_dir() / "oracle" / layer / det / f"{seq}.json").read_text())[levels]
        if levels.startswith("ORACLEB"):  # schedule of native resolutions
            adapter = _ResolutionSchedule(set(cd.by_res))
        src = LevelSource(schedule=sched)
    else:
        src = LevelSource(fixed=levels)
    pipe = SciV7Pipeline(
        levels=src, adapter=adapter,
        layer=V7Layer(spec_from_dict(dict(spec, name=layer)), HostContract(**h)),
        tracker=tracker,
        make_detection=lambda d, v: Detection(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                                              confidence=v, class_id=d.class_id),
        record_trace=trace)
    lines, audit = [], []
    if needs_image:                       # a host that reads pixels (BoT-SORT camera-motion compensation)
        import cv2
        SPLITS, _ = _splits()
        frames = sorted((Path(SPLITS[SPLIT]["data"]) / "sequences" / seq).glob("*.jpg"))
        if len(frames) != cd.frames or frames[0].stat().st_size == 0:
            raise SystemExit(f"{seq}: host '{host}' needs the real frames")
    for i in range(1, cd.frames + 1):
        cd.frame = i
        vis = cd.visual_dict(i)
        tracks, a = pipe.step(dict(edges=vis["edges"], brightness=vis["brightness"], blur=vis["blur"]),
                              lambda r: cd.detect(None, 0.0, None, r), cd.shape,
                              motion=vis.get("motion"),
                              image=cv2.imread(str(frames[i - 1])) if needs_image else None)
        for t in tracks:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},{t.x2 - t.x1:.3f},"
                         f"{t.y2 - t.y1:.3f},{t.confidence:.6f},{t.class_id},-1,-1\n")
        audit.append(a)
    return lines, audit, cd.frames, pipe


def run_one(job):
    system, det, seq = job
    dest = _path(system, det, seq)
    stamp = dict(code_sha=code_sha(), system=system)
    if dest.exists():
        try:
            if pickle.load(open(dest, "rb")).get("stamp") == stamp:
                return
        except Exception:
            pass
    SPLITS, _ = _splits()
    data = SPLITS[SPLIT]["data"]
    lines, audit, n, _ = track_sequence(system, det, seq)
    if len(list((Path(data) / "sequences" / seq).glob("*.jpg"))) != n:
        raise SystemExit(f"{seq}: frame count of {data} differs from the cache ({n})")
    from tools.seqstats import sequence_stats
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.writelines(lines)
    f.close()
    st = sequence_stats(data, seq, f.name)
    os.unlink(f.name)
    st.update(audit=audit, tracks_txt="".join(lines), stamp=stamp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(dest, "wb"))


def _pool(fn, jobs):
    with ProcessPoolExecutor(int(os.environ.get("V7_WORKERS", 4))) as ex:
        list(ex.map(fn, jobs, chunksize=1))


def run(systems, dets):
    _, seqs_of = _splits()
    seqs = seqs_of(SPLIT)
    first = [s for s in systems if not parse(s)[2].startswith(("PERM", "ORACLE"))]
    perm = [s for s in systems if parse(s)[2].startswith(("PERM", "ORACLE"))]
    for group in (first, perm):          # PERM reads the SCI schedule
        _pool(run_one, [(sy, d, s) for sy in group for d in dets for s in seqs])


def summary(system, dets):
    from tools.seqstats import combine
    _, seqs_of = _splits()
    seqs = seqs_of(SPLIT)
    res = {}
    for d in dets:
        st = [load(system, d, s) for s in seqs]
        m = combine(st)
        m["per"] = {s: combine([x]) for s, x in zip(seqs, st)}
        m["cat"] = [s for s in seqs if m["per"][s]["MOTA"] < 0]
        m["ops"] = operating_stats([x["audit"] for x in st], d)
        m["per_ops"] = {s: operating_stats([x["audit"]], d) for s, x in zip(seqs, st)}
        res[d] = m
    return res


def operating_stats(audits, det):
    """Compute level use, relative compute (r^2 / 736^2), switches, V7f
    regime shares and intervention rate (frames where the score layer passed
    the host anything other than its raw candidates at its own thresholds)."""
    from tools.v7.systems import HOST_BYTETRACK as h
    ref = json.loads((ROOT / "configs/sci_v7_profiles.json").read_text())["reference_resolution"]
    a = [x for au in audits for x in au]
    res = np.array([x["setting"] for x in a], float)
    lv = [x["level"] for x in a]
    sw = sum(sum(au[i]["level"] != au[i - 1]["level"] for i in range(1, len(au))) for au in audits)
    reg = [x.get("regime") for x in a]
    inter = [x["n_pass"] != x["n_in"] or abs(x["assoc"] - h["assoc"]) > 1e-9 or
             abs(x["birth"] - h["birth"]) > 1e-9 or x.get("n_rescued", 0) > 0 or
             abs(x["match"] - h["match"]) > 1e-9 for x in a]
    return dict(frames=len(a), mean_resolution=float(res.mean()),
                rel_compute=float(np.mean(res ** 2) / ref ** 2),
                frac={k: float(np.mean([v == k for v in lv])) for k in ("LOW", "MEDIUM", "HIGH")},
                switches=int(sw),
                regime={k: float(np.mean([r == k for r in reg])) for k in ("cold", "clean", "noisy")},
                intervention=float(np.mean(inter)))


def report(systems, dets):
    print(f"{'system':<16}{'cat':>4} | " + " | ".join(
        f"{d:>7}: MOTA  HOTA  IDF1   IDS     FP     FN    P    R  comp  L/M/H" for d in dets))
    for sy in systems:
        r = summary(sy, dets)
        print(f"{sy:<16}{sum(len(r[d]['cat']) for d in dets):>4} | " + " | ".join(
            f"{'':>7} {r[d]['MOTA']:5.1f} {r[d]['HOTA']:5.1f} {r[d]['IDF1']:5.1f} {r[d]['IDS']:5d} "
            f"{r[d]['FP']:6d} {r[d]['FN']:6d} {r[d]['Precision']:4.1f} {r[d]['Recall']:4.1f} "
            f"{r[d]['ops']['rel_compute']:5.3f} "
            + "/".join(f"{r[d]['ops']['frac'][k]:.2f}" for k in ("LOW", "MEDIUM", "HIGH")) for d in dets))


def _official_one(job):
    system, det, seq = job
    os.chdir(ROOT)
    dest = _path(system, det, seq, ".official.pkl")
    st = load(system, det, seq)
    if dest.exists():
        try:
            if pickle.load(open(dest, "rb")).get("stamp") == st["stamp"]:
                return
        except Exception:
            pass
    from tools.v6.eval_official import official_sequence_stats
    SPLITS, _ = _splits()
    txt = st["tracks_txt"]
    tr = np.loadtxt(io.StringIO(txt), delimiter=",", ndmin=2) if txt else np.zeros((0, 10))
    o = official_sequence_stats(SPLITS[SPLIT]["data"], seq, tr)
    o["stamp"] = st["stamp"]
    pickle.dump(o, open(dest, "wb"))


def official_summary(system, dets):
    from tools.v6.eval_official import combine_official
    _, seqs_of = _splits()
    seqs = seqs_of(SPLIT)
    res = {}
    for d in dets:
        st = [pickle.load(open(_path(system, d, s, ".official.pkl"), "rb")) for s in seqs]
        m = combine_official(st)
        m["per"] = {s: combine_official([x]) for s, x in zip(seqs, st)}
        m["cat"] = [s for s in seqs if m["per"][s]["MOTA"] < 0]
        res[d] = m
    return res


def official(systems, dets):
    _, seqs_of = _splits()
    _pool(_official_one, [(sy, d, s) for sy in systems for d in dets for s in seqs_of(SPLIT)])
    print("OFFICIAL-COMPATIBLE VisDrone-MOT (class-aware, ignored regions dropped)")
    print(f"{'system':<16}{'cat':>4} | " + " | ".join(
        f"{d:>7}: MOTA  HOTA  IDF1   IDS     FP     FN    P    R" for d in dets))
    for sy in systems:
        r = official_summary(sy, dets)
        print(f"{sy:<16}{sum(len(r[d]['cat']) for d in dets):>4} | " + " | ".join(
            f"{'':>7} {r[d]['MOTA']:5.1f} {r[d]['HOTA']:5.1f} {r[d]['IDF1']:5.1f} {r[d]['IDS']:5d} "
            f"{r[d]['FP']:6d} {r[d]['FN']:6d} {r[d]['Precision']:4.1f} {r[d]['Recall']:4.1f}" for d in dets))


def seq_table(systems, dets):
    _, seqs_of = _splits()
    seqs = seqs_of(SPLIT)
    res = {sy: summary(sy, dets) for sy in systems}
    for d in dets:
        print(f"== {d}  (MOTA / HOTA / IDF1 / relative compute per system)")
        print(f"{'seq':<20}" + "".join(f"{sy:>30}" for sy in systems))
        for s in seqs:
            print(f"{s:<20}" + "".join(
                f"{res[sy][d]['per'][s]['MOTA']:7.1f}{res[sy][d]['per'][s]['HOTA']:7.1f}"
                f"{res[sy][d]['per'][s]['IDF1']:7.1f}{res[sy][d]['per_ops'][s]['rel_compute']:9.3f}"
                for sy in systems))


def ops(systems, dets):
    for sy in systems:
        r = summary(sy, dets)
        for d in dets:
            o = r[d]["ops"]
            print(f"{sy:<16} {d:<7} mean res {o['mean_resolution']:6.1f}  rel compute {o['rel_compute']:.3f}  "
                  f"L/M/H {o['frac']['LOW']:.3f}/{o['frac']['MEDIUM']:.3f}/{o['frac']['HIGH']:.3f}  "
                  f"switches {o['switches']:4d}  cold/clean/noisy {o['regime']['cold']:.3f}/"
                  f"{o['regime']['clean']:.3f}/{o['regime']['noisy']:.3f}  intervention {o['intervention']:.3f}")


def boot(a, b, dets, protocol="internal", n=10000, seed=42, out=None):
    """Paired sequence bootstrap, same method as tools/v7/bootstrap.py."""
    from tools.v7.bootstrap import combiner, fmt, paired
    _, seqs_of = _splits()
    seqs = seqs_of(SPLIT)
    keys = ["MOTA", "HOTA", "IDF1", "IDS", "FP", "FN"]
    if protocol == "official":
        official([a, b], dets)
        get = lambda sy, d, s: pickle.load(open(_path(sy, d, s, ".official.pkl"), "rb"))
    else:
        get = load
    comb = combiner(protocol)
    r = dict(split=SPLIT, A=a, B=b, protocol=protocol, n=n, seed=seed, per_det={})
    allA, allB = [], []
    for d in dets:
        A, B = [get(a, d, s) for s in seqs], [get(b, d, s) for s in seqs]
        r["per_det"][d] = paired(A, B, comb, n, seed, keys)
        allA += A
        allB += B
    if len(dets) > 1:
        r["pooled_cells"] = paired(allA, allB, comb, n, seed, keys)
    print(fmt(r))
    if out:
        p = Path(out)
        old = json.load(open(p)) if p.exists() else []
        old.append(r)
        json.dump(old, open(p, "w"), indent=1)
    return r


if __name__ == "__main__":
    dets = os.environ.get("V7_DETS", ",".join(DETS)).split(",")
    cmd, *args = sys.argv[1:]
    if cmd == "boot":
        ap = argparse.ArgumentParser()
        ap.add_argument("A")
        ap.add_argument("B")
        ap.add_argument("--protocol", default="internal", choices=["internal", "official"])
        ap.add_argument("--json", default=None)
        x = ap.parse_args(args)
        boot(x.A, x.B, dets, x.protocol, out=x.json)
    else:
        {"run": run, "report": report, "official": official, "seq": seq_table,
         "ops": ops}[cmd](args, dets)
