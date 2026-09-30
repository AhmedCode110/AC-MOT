#!/usr/bin/env python3
"""Re-score archived tracker outputs of the U2MOT and SparseTrack experiments.

Nothing is tracked or tuned here: the script reads result files that already
exist and evaluates them.

  u2mot        VisDrone2019-MOT-test-dev, U2MOT baseline vs frozen controller.
               GT/prediction filtering is the U2MOT repository protocol
               (tools/utils/eval_visdrone.py, commit 7411211): classes
               {1,4,5,6,9} merged, ignore regions of classes 0 and 11 removed
               by box centre, IoU 0.5. MOTA/IDF1 with motmetrics exactly as
               that script, plus HOTA/CLEAR/Identity from TrackEval metric
               classes on the same filtered boxes.
  sparsetrack  MOT17 val_half (7 FRCNN sequences), static NMS runs and the
               frozen Adaptive Edge V1 run. MOTA/IDF1 with the motmetrics
               protocol of the frozen runner (GT min_confidence=1), plus
               official TrackEval MOT17 HOTA/CLEAR/Identity (pedestrian,
               standard preprocessing) on gt_val_half.txt.

Every archived number found next to the inputs is recomputed and compared
(reproduction check). Paired comparisons use a bootstrap over sequences
(10 000 resamples, seed 0), win/tie/loss per sequence and leave-one-out.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

# NumPy >= 1.24 / 2.x removed aliases used by motmetrics 1.4.0 and TrackEval.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
for _name, _typ in (("float", float), ("int", int), ("bool", bool)):
    if _name not in np.__dict__:
        setattr(np, _name, _typ)

import pandas as pd  # noqa: E402
import motmetrics as mm  # noqa: E402

B_DEFAULT = 10000
SEED_DEFAULT = 0

U2MOT_TESTDEV_SEQS = [
    "uav0000009_03358_v", "uav0000073_00600_v", "uav0000073_04464_v",
    "uav0000077_00720_v", "uav0000088_00290_v", "uav0000119_02301_v",
    "uav0000120_04775_v", "uav0000161_00000_v", "uav0000188_00000_v",
    "uav0000201_00000_v", "uav0000249_00001_v", "uav0000249_02688_v",
    "uav0000297_00000_v", "uav0000297_02761_v", "uav0000306_00230_v",
    "uav0000355_00001_v", "uav0000370_00001_v",
]

# Archived values (FINAL_METRICS.json of FINAL_U2MOT_ACMOT_FREEZE_2026-09-14).
U2MOT_ARCHIVED = {
    "baseline": {"FP": 41385, "FN": 63241, "IDS": 1239, "MOTA_1dec": 53.9, "IDF1_1dec": 69.8},
    "controller": {"FP": 40155, "FN": 64801, "IDS": 1152,
                   "MOTA": 53.76678605352365, "IDF1": 69.84938968519636},
}

MOT17_VAL_HALF_FRAMES = OrderedDict([
    ("MOT17-02-FRCNN", 299), ("MOT17-04-FRCNN", 524), ("MOT17-05-FRCNN", 418),
    ("MOT17-09-FRCNN", 262), ("MOT17-10-FRCNN", 326), ("MOT17-11-FRCNN", 449),
    ("MOT17-13-FRCNN", 374),
])


# ---------------------------------------------------------------------------
# U2MOT protocol (verbatim logic of clean_gt_and_ts in eval_visdrone.py)
# ---------------------------------------------------------------------------

def u2mot_clean_gt_and_ts(gt, ts):
    valid_cls = np.array([1, 4, 5, 6, 9]).reshape(1, -1)  # pedestrian,car,van,truck,bus

    def filter_cls(df):
        valid = (df['ClassId'].values.reshape(-1, 1) == valid_cls).any(axis=1)
        valid &= df['Confidence'] > 0.
        return df[valid]

    def filter_ignore(df, ignore_regions):
        ignore = df['ClassId'] == -1
        frame_ids = np.array([v[0] for v in df.index.values])
        for frame_id, regions in ignore_regions.items():
            x1, y1, x2, y2 = np.split(np.array(regions).reshape(-1, 4).T, 4, axis=0)
            f_ind = frame_ids == frame_id
            xc, yc = (df['X'][f_ind] + df['Width'][f_ind] / 2.).values.reshape(-1, 1), \
                     (df['Y'][f_ind] + df['Height'][f_ind] / 2.).values.reshape(-1, 1)
            ignore[f_ind] = ((xc > x1) & (xc < x2) & (yc > y1) & (yc < y2)).any(axis=1)
        return df[~ignore]

    _gt, _ts = OrderedDict(), OrderedDict()
    for k, label in gt.items():
        ignore_indices = (label['ClassId'] == 0) | (label['ClassId'] == 11)
        ig_x1, ig_y1, ig_w, ig_h = label['X'][ignore_indices].values, \
                                   label['Y'][ignore_indices].values, \
                                   label['Width'][ignore_indices].values, \
                                   label['Height'][ignore_indices].values
        ig_x2, ig_y2 = ig_x1 + ig_w, ig_y1 + ig_h
        ignore_regions = {}
        frame_ids = np.array([v[0] for v in label.index.values])
        for i, frame_id in enumerate(frame_ids[ignore_indices]):
            if frame_id not in ignore_regions:
                ignore_regions[frame_id] = []
            ignore_regions[frame_id].append([ig_x1[i], ig_y1[i], ig_x2[i], ig_y2[i]])

        label = filter_cls(label)
        label = filter_ignore(label, ignore_regions)
        pred = filter_cls(ts[k])
        pred = filter_ignore(pred, ignore_regions)

        _gt[k] = label
        _ts[k] = pred
    return _gt, _ts


# ---------------------------------------------------------------------------
# motmetrics
# ---------------------------------------------------------------------------

MM_METRICS = ["num_objects", "num_false_positives", "num_misses", "num_switches",
              "idtp", "idfp", "idfn", "mota", "idf1", "precision", "recall"]


def motmetrics_summary(gt, ts, names):
    accs = [mm.utils.compare_to_groundtruth(gt[k], ts[k], 'iou', distth=0.5) for k in names]
    mh = mm.metrics.create()
    return mh.compute_many(accs, names=names, metrics=MM_METRICS, generate_overall=True)


# ---------------------------------------------------------------------------
# TrackEval
# ---------------------------------------------------------------------------

def import_trackeval(trackeval_dir):
    if trackeval_dir:
        sys.path.insert(0, str(trackeval_dir))
    import trackeval  # noqa: F401
    return trackeval


def trackeval_data_from_frames(trackeval, gt_df, ts_df, num_timesteps):
    """Builds the TrackEval metric input from already-filtered motmetrics frames."""
    def per_frame(df):
        out = {}
        if len(df) == 0:
            return out
        frames = np.array([v[0] for v in df.index.values], dtype=int)
        ids = np.array([v[1] for v in df.index.values], dtype=int)
        boxes = df[['X', 'Y', 'Width', 'Height']].values.astype(float)
        for f in np.unique(frames):
            m = frames == f
            out[int(f)] = (ids[m], boxes[m])
        return out

    g, t = per_frame(gt_df), per_frame(ts_df)
    gt_uids = sorted({int(i) for v in g.values() for i in v[0]})
    tr_uids = sorted({int(i) for v in t.values() for i in v[0]})
    gmap = {u: n for n, u in enumerate(gt_uids)}
    tmap = {u: n for n, u in enumerate(tr_uids)}
    data = {"gt_ids": [], "tracker_ids": [], "similarity_scores": [],
            "num_timesteps": num_timesteps, "num_gt_ids": len(gt_uids),
            "num_tracker_ids": len(tr_uids), "num_gt_dets": 0, "num_tracker_dets": 0}
    empty_ids, empty_boxes = np.zeros(0, dtype=int), np.zeros((0, 4))
    for f in range(1, num_timesteps + 1):
        gi, gb = g.get(f, (empty_ids, empty_boxes))
        ti, tb = t.get(f, (empty_ids, empty_boxes))
        data["gt_ids"].append(np.array([gmap[int(i)] for i in gi], dtype=int))
        data["tracker_ids"].append(np.array([tmap[int(i)] for i in ti], dtype=int))
        data["similarity_scores"].append(
            trackeval.datasets._base_dataset._BaseDataset._calculate_box_ious(gb, tb, box_format='xywh'))
        data["num_gt_dets"] += len(gi)
        data["num_tracker_dets"] += len(ti)
    return data


def te_metric_objects(trackeval):
    return OrderedDict([
        ("HOTA", trackeval.metrics.HOTA()),
        ("CLEAR", trackeval.metrics.CLEAR({"PRINT_CONFIG": False, "THRESHOLD": 0.5})),
        ("Identity", trackeval.metrics.Identity({"PRINT_CONFIG": False, "THRESHOLD": 0.5})),
    ])


def te_eval_sequence(metrics, data):
    return {name: m.eval_sequence(data) for name, m in metrics.items()}


def te_combine(metrics, per_seq, keys):
    out = {}
    for name, m in metrics.items():
        out[name] = m.combine_sequences({f"{i}_{k}": per_seq[k][name] for i, k in enumerate(keys)})
    return out


def te_headline(res):
    h, c, i = res["HOTA"], res["CLEAR"], res["Identity"]
    return {
        "HOTA": 100 * float(np.mean(h["HOTA"])), "DetA": 100 * float(np.mean(h["DetA"])),
        "AssA": 100 * float(np.mean(h["AssA"])), "LocA": 100 * float(np.mean(h["LocA"])),
        "MOTA": 100 * float(c["MOTA"]), "IDF1": 100 * float(i["IDF1"]),
        "IDSW": int(c["IDSW"]), "CLR_FP": int(c["CLR_FP"]), "CLR_FN": int(c["CLR_FN"]),
        "Frag": int(c["Frag"]),
    }


# ---------------------------------------------------------------------------
# Paired statistics over sequences
# ---------------------------------------------------------------------------

def paired_stats(metrics, per_seq_a, per_seq_b, seqs, b, seed, keys=("HOTA", "MOTA", "IDF1", "AssA", "DetA")):
    rng = np.random.default_rng(seed)
    n = len(seqs)
    full_a = te_headline(te_combine(metrics, per_seq_a, seqs))
    full_b = te_headline(te_combine(metrics, per_seq_b, seqs))
    out = {"n_sequences": n, "bootstrap_resamples": b, "seed": seed, "metrics": {}}
    idx = rng.integers(0, n, size=(b, n))
    boot = {k: np.empty(b) for k in keys}
    for r in range(b):
        sample = [seqs[j] for j in idx[r]]
        ha = te_headline(te_combine(metrics, per_seq_a, sample))
        hb = te_headline(te_combine(metrics, per_seq_b, sample))
        for k in keys:
            boot[k][r] = ha[k] - hb[k]
    for k in keys:
        per = [te_headline(te_combine(metrics, per_seq_a, [s]))[k]
               - te_headline(te_combine(metrics, per_seq_b, [s]))[k] for s in seqs]
        loo = []
        for s in seqs:
            rest = [x for x in seqs if x != s]
            loo.append(te_headline(te_combine(metrics, per_seq_a, rest))[k]
                       - te_headline(te_combine(metrics, per_seq_b, rest))[k])
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        out["metrics"][k] = {
            "a": full_a[k], "b": full_b[k], "delta": full_a[k] - full_b[k],
            "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "wins": int(sum(d > 1e-12 for d in per)), "ties": int(sum(abs(d) <= 1e-12 for d in per)),
            "losses": int(sum(d < -1e-12 for d in per)),
            "loo_min": float(min(loo)), "loo_max": float(max(loo)),
            "per_sequence_delta": dict(zip(seqs, [float(d) for d in per])),
        }
    return out


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def write_per_sequence_csv(path, per_seq_te, mm_summary, seqs, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sequence", "HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "IDSW", "CLR_FP", "CLR_FN", "Frag",
              "mm_MOTA", "mm_IDF1", "mm_IDS", "mm_FP", "mm_FN", "mm_GT"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in seqs + ["OVERALL"]:
            if s == "OVERALL":
                h = te_headline(te_combine(metrics, per_seq_te, seqs))
            else:
                h = te_headline(per_seq_te[s])
            row = {"sequence": s, **h}
            r = mm_summary.loc[s]
            row.update({"mm_MOTA": 100 * float(r["mota"]), "mm_IDF1": 100 * float(r["idf1"]),
                        "mm_IDS": int(r["num_switches"]), "mm_FP": int(r["num_false_positives"]),
                        "mm_FN": int(r["num_misses"]), "mm_GT": int(r["num_objects"])})
            w.writerow(row)


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_manifest(folder, seqs):
    return {s: sha256(Path(folder) / f"{s}.txt") for s in seqs}


# ---------------------------------------------------------------------------
# U2MOT
# ---------------------------------------------------------------------------

def run_u2mot(args, trackeval):
    gt_dir = Path(args.gt_dir)
    systems = OrderedDict([("baseline", Path(args.baseline_dir)), ("controller", Path(args.controller_dir))])
    seqs = list(U2MOT_TESTDEV_SEQS)
    for name, d in [("gt", gt_dir)] + list(systems.items()):
        missing = [s for s in seqs if not (d / f"{s}.txt").is_file()]
        if missing:
            raise SystemExit(f"{name}: missing {len(missing)} sequence files in {d}: {missing[:3]}")

    gt_raw = OrderedDict((s, mm.io.load_motchallenge(str(gt_dir / f"{s}.txt"), min_confidence=-1.0)) for s in seqs)
    out_dir = Path(args.out_dir)
    metrics = te_metric_objects(trackeval)
    report = {"dataset": "VisDrone2019-MOT-test-dev", "protocol": "U2MOT eval_visdrone.py filtering, IoU 0.5",
              "inputs": {"gt": input_manifest(gt_dir, seqs)}, "systems": {}, "reproduction": {}}
    per_seq = {}
    for name, d in systems.items():
        ts_raw = OrderedDict((s, mm.io.load_motchallenge(str(d / f"{s}.txt"), min_confidence=-1.0)) for s in seqs)
        gt_f, ts_f = u2mot_clean_gt_and_ts(gt_raw, ts_raw)
        summ = motmetrics_summary(gt_f, ts_f, seqs)
        per_seq[name] = {}
        for s in seqs:
            frames = [v[0] for v in gt_raw[s].index.values] + [v[0] for v in ts_raw[s].index.values]
            data = trackeval_data_from_frames(trackeval, gt_f[s], ts_f[s], int(max(frames)))
            per_seq[name][s] = te_eval_sequence(metrics, data)
        ov = summ.loc["OVERALL"]
        report["inputs"][name] = input_manifest(d, seqs)
        report["systems"][name] = {
            "motmetrics_overall": {"MOTA": 100 * float(ov["mota"]), "IDF1": 100 * float(ov["idf1"]),
                                   "IDS": int(ov["num_switches"]), "FP": int(ov["num_false_positives"]),
                                   "FN": int(ov["num_misses"]), "GT": int(ov["num_objects"])},
            "trackeval_overall": te_headline(te_combine(metrics, per_seq[name], seqs)),
        }
        write_per_sequence_csv(out_dir / f"u2mot_{name}_per_sequence.csv", per_seq[name], summ, seqs, metrics)

    for name in systems:
        got = report["systems"][name]["motmetrics_overall"]
        arch = U2MOT_ARCHIVED[name]
        checks = {k: (got[k], arch[k], got[k] == arch[k]) for k in ("FP", "FN", "IDS")}
        if "MOTA" in arch:
            checks["MOTA"] = (got["MOTA"], arch["MOTA"], abs(got["MOTA"] - arch["MOTA"]) < 1e-9)
            checks["IDF1"] = (got["IDF1"], arch["IDF1"], abs(got["IDF1"] - arch["IDF1"]) < 1e-9)
        else:
            checks["MOTA_1dec"] = (round(got["MOTA"], 1), arch["MOTA_1dec"], round(got["MOTA"], 1) == arch["MOTA_1dec"])
            checks["IDF1_1dec"] = (round(got["IDF1"], 1), arch["IDF1_1dec"], round(got["IDF1"], 1) == arch["IDF1_1dec"])
        report["reproduction"][name] = {k: {"recomputed": v[0], "archived": v[1], "match": bool(v[2])}
                                        for k, v in checks.items()}
    report["reproduction_all_match"] = all(c["match"] for r in report["reproduction"].values() for c in r.values())
    report["paired_controller_vs_baseline"] = paired_stats(
        metrics, per_seq["controller"], per_seq["baseline"], seqs, args.bootstrap, args.seed)
    write_json(out_dir / "u2mot_testdev_rescore.json", report)
    return report


# ---------------------------------------------------------------------------
# SparseTrack
# ---------------------------------------------------------------------------

def sparsetrack_mm(gt_root, results_dir, seqs):
    gt, ts = OrderedDict(), OrderedDict()
    for s in seqs:
        gt[s] = mm.io.loadtxt(str(Path(gt_root) / s / "gt" / "gt_val_half.txt"), fmt="mot15-2D", min_confidence=1)
        ts[s] = mm.io.loadtxt(str(Path(results_dir) / f"{s}.txt"), fmt="mot15-2D", min_confidence=-1)
    return motmetrics_summary(gt, ts, seqs)


def sparsetrack_trackeval(trackeval, gt_root, runs, seqs, workdir):
    trackers_root = Path(workdir) / "trackers"
    for name, d in runs.items():
        (trackers_root / name).mkdir(parents=True, exist_ok=True)
        link = trackers_root / name / "data"
        if not link.exists():
            os.symlink(os.path.abspath(d), link)
    cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    cfg.update({
        "GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(trackers_root), "OUTPUT_FOLDER": str(Path(workdir) / "te_out"),
        "TRACKERS_TO_EVAL": list(runs.keys()), "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train", "SKIP_SPLIT_FOL": True, "TRACKER_SUB_FOLDER": "data",
        "SEQ_INFO": {s: MOT17_VAL_HALF_FRAMES[s] for s in seqs},
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt_val_half.txt", "PRINT_CONFIG": False,
    })
    dataset = trackeval.datasets.MotChallenge2DBox(cfg)
    metrics = te_metric_objects(trackeval)
    per_seq = {}
    for name in runs:
        per_seq[name] = {}
        for s in seqs:
            raw = dataset.get_raw_seq_data(name, s)
            data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
            per_seq[name][s] = te_eval_sequence(metrics, data)
    return metrics, per_seq


def read_archived_metrics_csv(run_dir):
    p = Path(run_dir) / "metrics.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p).set_index("sequence")


def run_sparsetrack(args, trackeval):
    seqs = list(MOT17_VAL_HALF_FRAMES.keys())
    runs = OrderedDict()
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = Path(path)
    for name, d in runs.items():
        missing = [s for s in seqs if not (d / f"{s}.txt").is_file()]
        if missing:
            raise SystemExit(f"{name}: missing {len(missing)} sequence files in {d}: {missing[:3]}")
    out_dir = Path(args.out_dir)
    workdir = Path(tempfile.mkdtemp(prefix="rescore_st_"))
    try:
        metrics, per_seq = sparsetrack_trackeval(trackeval, args.gt_root, runs, seqs, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    report = {"dataset": "MOT17 val_half (7 FRCNN sequences)",
              "protocol": {"motmetrics": "frozen runner compute_metrics (GT min_confidence=1, IoU 0.5)",
                           "trackeval": "MotChallenge2DBox, pedestrian, standard preprocessing"},
              "inputs": {"gt": {s: sha256(Path(args.gt_root) / s / "gt" / "gt_val_half.txt") for s in seqs}},
              "systems": {}, "reproduction": {}, "pairs": {}}
    for name, d in runs.items():
        summ = sparsetrack_mm(args.gt_root, d, seqs)
        report["inputs"][name] = input_manifest(d, seqs)
        ov = summ.loc["OVERALL"]
        report["systems"][name] = {
            "results_dir": str(d),
            "motmetrics_overall": {"MOTA": 100 * float(ov["mota"]), "IDF1": 100 * float(ov["idf1"]),
                                   "IDS": int(ov["num_switches"]), "FP": int(ov["num_false_positives"]),
                                   "FN": int(ov["num_misses"]), "GT": int(ov["num_objects"])},
            "trackeval_overall": te_headline(te_combine(metrics, per_seq[name], seqs)),
        }
        write_per_sequence_csv(out_dir / f"sparsetrack_{name}_per_sequence.csv", per_seq[name], summ, seqs, metrics)
        arch = read_archived_metrics_csv(d.parent)
        if arch is not None:
            rows = {}
            for s in seqs + ["OVERALL"]:
                a, r = arch.loc[s], summ.loc[s]
                rows[s] = {
                    "mota": [float(r["mota"]), float(a["mota"]), abs(float(r["mota"]) - float(a["mota"])) < 1e-9],
                    "idf1": [float(r["idf1"]), float(a["idf1"]), abs(float(r["idf1"]) - float(a["idf1"])) < 1e-9],
                    "ids": [int(r["num_switches"]), int(a["num_switches"]), int(r["num_switches"]) == int(a["num_switches"])],
                    "fp": [int(r["num_false_positives"]), int(a["num_false_positives"]),
                           int(r["num_false_positives"]) == int(a["num_false_positives"])],
                    "fn": [int(r["num_misses"]), int(a["num_misses"]), int(r["num_misses"]) == int(a["num_misses"])],
                }
            report["reproduction"][name] = {
                "archived_file": str(d.parent / "metrics.csv"),
                "all_match": all(v[2] for row in rows.values() for v in row.values()),
                "rows": rows,
            }
        else:
            report["reproduction"][name] = {"archived_file": None, "all_match": None}
    for spec in args.pair:
        a, b = spec.split(":", 1)
        report["pairs"][f"{a}_vs_{b}"] = paired_stats(metrics, per_seq[a], per_seq[b], seqs, args.bootstrap, args.seed)
    write_json(out_dir / "sparsetrack_val_half_rescore.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trackeval-dir", default=None, help="TrackEval checkout (commit 12c8791)")
    p.add_argument("--bootstrap", type=int, default=B_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    sub = p.add_subparsers(dest="mode", required=True)
    u = sub.add_parser("u2mot")
    u.add_argument("--gt-dir", required=True, help="VisDrone2019-MOT-test-dev/annotations")
    u.add_argument("--baseline-dir", required=True, help="01_A0_BASELINE/track_res")
    u.add_argument("--controller-dir", required=True, help="06_FINAL_TESTDEV_17SEQ/track_res")
    u.add_argument("--out-dir", required=True)
    s = sub.add_parser("sparsetrack")
    s.add_argument("--gt-root", required=True, help="MOT17/train (contains <seq>/gt/gt_val_half.txt)")
    s.add_argument("--run", action="append", required=True, help="name=path/to/track_results (repeatable)")
    s.add_argument("--pair", action="append", default=[], help="a:b paired comparison a minus b (repeatable)")
    s.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    trackeval = import_trackeval(args.trackeval_dir)
    if args.mode == "u2mot":
        rep = run_u2mot(args, trackeval)
        print(json.dumps({"reproduction_all_match": rep["reproduction_all_match"],
                          "systems": rep["systems"],
                          "paired": {k: {m: v[m] for m in ("delta", "ci95", "wins", "losses")}
                                     for k, v in rep["paired_controller_vs_baseline"]["metrics"].items()}},
                         indent=2))
    else:
        rep = run_sparsetrack(args, trackeval)
        print(json.dumps({"reproduction": {k: v["all_match"] for k, v in rep["reproduction"].items()},
                          "systems": {k: v["trackeval_overall"] for k, v in rep["systems"].items()},
                          "pairs": {k: {m: {x: v["metrics"][m][x] for x in ("delta", "ci95", "wins", "losses")}
                                        for m in v["metrics"]} for k, v in rep["pairs"].items()}},
                         indent=2))


if __name__ == "__main__":
    main()
