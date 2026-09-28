"""
KITTI tracking (TRAINING split) evaluation for V7 development runs, with the
official KITTI HOTA implementation (TrackEval Kitti2DBox, pinned 12c8791):
classes car and pedestrian, KITTI distractors (van / person sitting),
DontCare regions and occlusion/truncation rules exactly as TrackEval
implements them.

Tracker files come from `tools/v7/dev.py track` with V7_SPLIT=kitti
(outputs/v7/kitti/<system>/<det>/<seq>.trk.pkl). Conversion to the KITTI
format: frame index 0-based (the runner is 1-based), detector task classes
COCO person -> Pedestrian, COCO car -> Car (declared in
outputs/det_cache_kitti_native/<det>/task.json before any run).

  python tools/v7/kitti/kitti_eval.py <det> <system> [<system> ...]
  python tools/v7/kitti/kitti_eval.py --boot <det> <A> <B>
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

for _n, _v in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _n):
        setattr(np, _n, _v)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
TE = os.environ.get("ACMOT_TRACKEVAL", str(Path.home() / "acmot_work/CUE_SELECTION/cue_ablation_tools/TrackEval"))
sys.path.insert(0, TE)
import trackeval  # noqa: E402

KITTI = Path(os.environ.get("ACMOT_KITTI", Path.home() / "acmot_work/kitti"))
GT = KITTI / "gt"
TYPES = {0: "Pedestrian", 2: "Car"}
CLASSES = ["car", "pedestrian"]


def seqs():
    return [l.split()[0] for l in open(GT / "evaluate_tracking.seqmap.training") if l.strip()]


def write_tracker(det, system):
    """Convert the runner output to KITTI files; returns the trackers root."""
    root = ROOT / "outputs/v7/kitti_trackeval" / det
    d = root / system.replace("@", "__") / "data"
    d.mkdir(parents=True, exist_ok=True)
    for s in seqs():
        run = pickle.load(open(ROOT / "outputs/v7/kitti" / system / det / f"{s}.trk.pkl", "rb"))
        a = np.loadtxt(io.StringIO(run["tracks_txt"]), delimiter=",", ndmin=2) if run["tracks_txt"] else np.zeros((0, 10))
        lines = []
        for r in a:
            t = TYPES.get(int(r[7]))
            if t is None:
                continue
            x1, y1, w, h = r[2:6]
            lines.append(f"{int(r[0]) - 1} {int(r[1])} {t} 0 0 -10 {x1:.2f} {y1:.2f} {x1 + w:.2f} "
                         f"{y1 + h:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {r[6]:.4f}\n")
        open(d / f"{s}.txt", "w").writelines(lines)
    return root


def evaluate(det, system):
    """Per-sequence, per-class TrackEval results (cached)."""
    tname = system.replace("@", "__")
    cache = ROOT / "outputs/v7/kitti_trackeval" / det / tname / "per_seq.pkl"
    stamp = [pickle.load(open(ROOT / "outputs/v7/kitti" / system / det / f"{s}.trk.pkl", "rb"))["v7_stamp"]
             for s in seqs()]
    if cache.exists():
        c = pickle.load(open(cache, "rb"))
        if c.get("stamp") == stamp:
            return c["per"]
    root = write_tracker(det, system)
    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update(dict(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False,
                   OUTPUT_DETAILED=False, PLOT_CURVES=False, TIME_PROGRESS=False,
                   DISPLAY_LESS_PROGRESS=True))
    dc = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
    dc.update(dict(GT_FOLDER=str(GT), TRACKERS_FOLDER=str(root), TRACKERS_TO_EVAL=[tname],
                   CLASSES_TO_EVAL=CLASSES, SPLIT_TO_EVAL="training", PRINT_CONFIG=False))
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    with redirect_stdout(io.StringIO()):
        res, _ = trackeval.Evaluator(ec).evaluate([trackeval.datasets.Kitti2DBox(dc)], metrics)
    r = res["Kitti2DBox"][tname]
    per = [{c: r[s][c] for c in CLASSES} for s in seqs()]
    pickle.dump(dict(stamp=stamp, per=per), open(cache, "wb"))
    return per


def combine(cells):
    """Pooled metrics over sequences, per class and class-averaged."""
    out = {}
    H, C, I = trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()
    for c in CLASSES:
        d = {str(i): x[c] for i, x in enumerate(cells)}
        with redirect_stdout(io.StringIO()):
            h = H.combine_sequences({k: v["HOTA"] for k, v in d.items()})
            cl = C.combine_sequences({k: v["CLEAR"] for k, v in d.items()})
            i = I.combine_sequences({k: v["Identity"] for k, v in d.items()})
        k = c[:3]
        out.update({f"HOTA_{k}": 100 * float(np.mean(h["HOTA"])), f"MOTA_{k}": 100 * float(cl["MOTA"]),
                    f"IDF1_{k}": 100 * float(i["IDF1"]), f"IDS_{k}": int(cl["IDSW"]),
                    f"FP_{k}": int(cl["CLR_FP"]), f"FN_{k}": int(cl["CLR_FN"])})
    for m in ("HOTA", "MOTA", "IDF1"):
        out[f"{m}_avg"] = (out[f"{m}_car"] + out[f"{m}_ped"]) / 2
    out["IDS_sum"] = out["IDS_car"] + out["IDS_ped"]
    return out


KEYS = ["HOTA_avg", "MOTA_avg", "IDF1_avg", "IDS_sum", "HOTA_car", "MOTA_car", "IDF1_car", "IDS_car",
        "HOTA_ped", "MOTA_ped", "IDF1_ped", "IDS_ped"]


if __name__ == "__main__":
    if sys.argv[1] == "--boot":
        from tools.v7.bootstrap import paired
        det, a, b = sys.argv[2:5]
        r = paired(evaluate(det, a), evaluate(det, b), combine, keys=KEYS)
        print(json.dumps({k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}
                          for k, v in r.items()}, indent=1))
    else:
        det, *systems = sys.argv[1:]
        for sy in systems:
            m = combine(evaluate(det, sy))
            print(f"{det:8} {sy:<28} " + " ".join(
                f"{k} {m[k]:.3f}" if isinstance(m[k], float) else f"{k} {m[k]}" for k in KEYS[:4]) +
                  f" | car {m['HOTA_car']:.2f}/{m['MOTA_car']:.2f}/{m['IDF1_car']:.2f} ped "
                  f"{m['HOTA_ped']:.2f}/{m['MOTA_ped']:.2f}/{m['IDF1_ped']:.2f}")
