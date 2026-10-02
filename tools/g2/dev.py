"""
G2 development runner, VisDrone2019-MOT-val (val-7) ONLY; protected splits
are refused before any data path is touched (tools/sci_v7/dev.guard).

  python tools/g2/dev.py run|report|official|seq|ops <system> [...]
  python tools/g2/dev.py boot <A> <B> [--protocol internal|official] [--json f]

A system is '<layer>+<compute>[@host]':
  layer    NATIVE (host alone) or V7f (frozen)
  compute  R<px>K<k>      static: detector at <px> every k-th frame, the last
                          canonical detections reused in between
           SCHED:<name>   per-frame (profile, period) schedule from
                          outputs/g2/val7/schedules/<layer>/<det>/<seq>.json
                          (oracle diagnostics, shuffled controls)
           CTRL:<name>    a controller from tools/g2/controllers.py
Detections: the resolution-sweep cache (SCI_SWEEP_ROOT); image cues: the
V7-record visual-cue cache. Env: V7_DETS, V7_WORKERS, ACMOT_VISDRONE_VAL.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sci_v7.dev import guard  # noqa: E402  (val-7 only; refuses protected splits)

SPLIT = "val7"
guard(os.environ.get("V7_SPLIT", SPLIT))
DETS = ["yolov8", "rtdetr"]
CODE = ["acmot_g2.py", "acmot_v7.py", "tools/g2/dev.py", "tools/g2/controllers.py", "configs/g2_compute_cost.json",
        "tools/sci_v7/sweep_cache.py", "tools/sci_v7/hosts.py", "tools/v7/systems.py"]
STATIC = re.compile(r"^R(\d+)K(\d+)$")


def code_sha():
    h = hashlib.sha256()
    for f in CODE:
        p = ROOT / f
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()


def costs(det):
    c = json.loads((ROOT / "configs/g2_compute_cost.json").read_text())["normalized_cost"][det]
    return {f"R{k}": float(v) for k, v in c.items()}


def parse(system):
    from tools.v7.systems import SYSTEMS
    base, host = (system.split("@") + ["bytetrack"])[:2]
    layer, comp = base.split("+", 1)
    if layer not in ("NATIVE", "V7f"):
        raise ValueError(layer)
    if not (STATIC.match(comp) or comp.startswith(("SCHED:", "CTRL:"))):
        raise ValueError(comp)
    return layer, dict(SYSTEMS[layer]), comp, host


def out_dir():
    return ROOT / "outputs/g2" / SPLIT


def _path(system, det, seq, suffix=".pkl"):
    return out_dir() / system.replace(":", "_") / det / f"{seq}{suffix}"


def load(system, det, seq):
    return pickle.load(open(_path(system, det, seq), "rb"))


def _splits():
    from tools.v7.dev import SPLITS, split_sequences
    return SPLITS, split_sequences


def open_cache(det, seq):
    from tools.sci_v7.sweep_cache import SweepCache
    SPLITS, _ = _splits()
    return SweepCache(os.environ["SCI_SWEEP_ROOT"], det, seq,
                      f"{SPLITS[SPLIT]['native']}/visual_cues/{seq}.npz")


def make_controller(comp, layer, det, seq, n_frames, cost):
    from acmot_g2 import Schedule, StaticSchedule
    m = STATIC.match(comp)
    if m:
        return StaticSchedule(f"R{m.group(1)}", int(m.group(2)))
    if comp.startswith("SCHED:"):
        f = out_dir() / "schedules" / layer / det / f"{seq}.json"
        return Schedule(json.loads(f.read_text())[comp[6:]])
    from tools.g2 import controllers
    return controllers.make(comp[5:], cost=cost, n_frames=n_frames)


def track_sequence(system, det, seq, trace=False):
    os.chdir(ROOT)
    from acmot_g2 import G2Pipeline
    from acmot_v7 import HostContract, V7Layer, spec_from_dict
    from adapters.types import Detection
    from tools.sci_v7.hosts import make_host
    layer, spec, comp, host = parse(system)
    tracker, h, needs_image = make_host(host)
    cd = open_cache(det, seq)
    cost = {p: c for p, c in costs(det).items() if int(p[1:]) in cd.by_res}
    ctrl = make_controller(comp, layer, det, seq, cd.frames, cost)
    pipe = G2Pipeline(controller=ctrl, detector=lambda p: cd.detect(None, 0.0, None, int(p[1:])), cost=cost,
                      layer=V7Layer(spec_from_dict(dict(spec, name=layer)), HostContract(**h)), tracker=tracker,
                      make_detection=lambda d, v: Detection(d.x1, d.y1, d.x2, d.y2, v, d.class_id),
                      record_trace=trace)
    frames = None
    if needs_image:
        import cv2
        SPLITS, _ = _splits()
        frames = sorted((Path(SPLITS[SPLIT]["data"]) / "sequences" / seq).glob("*.jpg"))
        if len(frames) != cd.frames or frames[0].stat().st_size == 0:
            raise SystemExit(f"{seq}: host '{host}' needs the real frames")
    lines, audit = [], []
    for i in range(1, cd.frames + 1):
        cd.frame = i
        vis = cd.visual_dict(i)
        obs = dict(edges=vis["edges"], brightness=vis["brightness"], blur=vis["blur"], motion=vis.get("motion"),
                   motion_resp=vis.get("motion_resp"))
        tracks, a = pipe.step(obs, cd.shape, motion=vis.get("motion"),
                              image=cv2.imread(str(frames[i - 1])) if needs_image else None)
        for t in tracks:
            lines.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},{t.x2 - t.x1:.3f},"
                         f"{t.y2 - t.y1:.3f},{t.confidence:.6f},{t.class_id},-1,-1\n")
        audit.append(a)
    return lines, audit, cd.frames, pipe


def run_one(job):
    system, det, seq = job
    dest = _path(system, det, seq)
    stamp = dict(code_sha=code_sha(), system=system, sweep=os.environ.get("SCI_SWEEP_ROOT"))
    if system.split("+", 1)[1].startswith("SCHED:"):
        layer = system.split("+")[0]
        stamp["schedule_sha"] = hashlib.sha256((out_dir() / "schedules" / layer / det / f"{seq}.json")
                                               .read_bytes()).hexdigest()
    if dest.exists():
        try:
            if pickle.load(open(dest, "rb")).get("stamp") == stamp:
                return
        except Exception:
            pass
    SPLITS, _ = _splits()
    data = SPLITS[SPLIT]["data"]
    lines, audit, n, _ = track_sequence(system, det, seq)
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
    _pool(run_one, [(sy, d, s) for sy in systems for d in dets for s in seqs_of(SPLIT)])


def combine(stats_list):
    """tools/seqstats.combine plus DetA / AssA of the same combined HOTA."""
    import io as _io
    from contextlib import redirect_stdout
    from tools.eval_local import HOTA
    from tools.seqstats import combine as base
    m = base(stats_list)
    with redirect_stdout(_io.StringIO()):
        h = HOTA().combine_sequences({str(i): s["hota"] for i, s in enumerate(stats_list)})
    m["DetA"] = 100 * float(np.mean(h["DetA"]))
    m["AssA"] = 100 * float(np.mean(h["AssA"]))
    return m


def operating_stats(audits):
    a = [x for au in audits for x in au]
    from tools.v7.systems import HOST_BYTETRACK as h
    prof = [x["profile"] for x in a]
    inter = [x["n_pass"] != x["n_in"] or abs(x["assoc"] - h["assoc"]) > 1e-9 or abs(x["birth"] - h["birth"]) > 1e-9
             or x.get("n_rescued", 0) > 0 or abs(x["match"] - h["match"]) > 1e-9 for x in a]
    return dict(frames=len(a), mean_cost=float(np.mean([x["cost"] for x in a])),
                call_rate=float(np.mean([x["called"] for x in a])),
                profiles={p: float(np.mean([q == p for q in prof])) for p in sorted(set(prof))},
                switches=int(sum(sum(au[i]["profile"] != au[i - 1]["profile"] for i in range(1, len(au))) for au in audits)),
                clean=float(np.mean([x.get("regime") == "clean" for x in a])),
                noisy=float(np.mean([x.get("regime") == "noisy" for x in a])),
                intervention=float(np.mean(inter)))


def summary(system, dets):
    _, seqs_of = _splits()
    seqs = seqs_of(SPLIT)
    res = {}
    for d in dets:
        st = [load(system, d, s) for s in seqs]
        m = combine(st)
        m["per"] = {s: combine([x]) for s, x in zip(seqs, st)}
        m["cat"] = [s for s in seqs if m["per"][s]["MOTA"] < 0]
        m["ops"] = operating_stats([x["audit"] for x in st])
        m["per_ops"] = {s: operating_stats([x["audit"]]) for s, x in zip(seqs, st)}
        res[d] = m
    return res


def report(systems, dets):
    print(f"{'system':<26}{'cat':>4} | " + " | ".join(
        f"{d:>7}: HOTA  DetA  AssA  MOTA  IDF1   IDS     FP     FN    P    R  cost  call" for d in dets))
    for sy in systems:
        r = summary(sy, dets)
        print(f"{sy:<26}{sum(len(r[d]['cat']) for d in dets):>4} | " + " | ".join(
            f"{'':>7} {r[d]['HOTA']:5.2f} {r[d].get('DetA', float('nan')):5.1f} {r[d].get('AssA', float('nan')):5.1f} "
            f"{r[d]['MOTA']:5.1f} {r[d]['IDF1']:5.1f} {r[d]['IDS']:5d} {r[d]['FP']:6d} {r[d]['FN']:6d} "
            f"{r[d]['Precision']:4.1f} {r[d]['Recall']:4.1f} {r[d]['ops']['mean_cost']:5.3f} {r[d]['ops']['call_rate']:5.2f}"
            for d in dets))


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
    for sy in systems:
        r = official_summary(sy, dets)
        print(f"{sy:<26}{sum(len(r[d]['cat']) for d in dets):>4} | " + " | ".join(
            f"{d}: HOTA {r[d]['HOTA']:5.2f} MOTA {r[d]['MOTA']:5.1f} IDF1 {r[d]['IDF1']:5.1f} IDS {r[d]['IDS']:5d} "
            f"FP {r[d]['FP']:6d} FN {r[d]['FN']:6d}" for d in dets))


def boot(a, b, dets, protocol="internal", n=10000, seed=42, out=None, quiet=False):
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
    if not quiet:
        print(fmt(r))
    if out:
        p = Path(out)
        old = json.loads(p.read_text()) if p.exists() else []
        old.append(r)
        p.write_text(json.dumps(old, indent=1))
    return r


def ops(systems, dets):
    for sy in systems:
        r = summary(sy, dets)
        for d in dets:
            o = r[d]["ops"]
            print(f"{sy:<26} {d:<7} cost {o['mean_cost']:.3f} call {o['call_rate']:.3f} switches {o['switches']:4d} "
                  f"clean/noisy {o['clean']:.3f}/{o['noisy']:.3f} intervention {o['intervention']:.3f} "
                  f"profiles {json.dumps({k: round(v, 3) for k, v in o['profiles'].items()})}")


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
        {"run": run, "report": report, "official": official, "ops": ops}[cmd](args, dets)
