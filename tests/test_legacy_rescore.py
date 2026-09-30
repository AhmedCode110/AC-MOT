"""Tests for tools/legacy_rescore/rescore_legacy_pipelines.py on synthetic data.

TrackEval (commit 12c8791) must be importable; set TRACKEVAL_DIR to a checkout
if it is not installed.
"""

import importlib.util
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rescore", ROOT / "tools" / "legacy_rescore" / "rescore_legacy_pipelines.py")
rescore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rescore)

try:
    trackeval = rescore.import_trackeval(os.environ.get("TRACKEVAL_DIR"))
except ImportError:  # pragma: no cover
    trackeval = None

pytestmark = pytest.mark.skipif(trackeval is None, reason="TrackEval not importable")

mm = rescore.mm


def _objects(n_obj, n_frames, rng, cls_choices=(1,)):
    rows = []
    for oid in range(1, n_obj + 1):
        x, y = rng.uniform(0, 800), rng.uniform(0, 500)
        vx, vy = rng.uniform(-4, 4), rng.uniform(-4, 4)
        w, h = rng.uniform(20, 60), rng.uniform(40, 100)
        cls = int(rng.choice(cls_choices))
        for f in range(1, n_frames + 1):
            rows.append([f, oid, x + vx * f, y + vy * f, w, h, cls])
    return rows


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")


def _visdrone_seq(tmp, seq, rng, n_frames=15, perturb=True):
    """VisDrone-style GT (frame,id,x,y,w,h,score,cat,trunc,occ) and a tracker file."""
    objs = _objects(12, n_frames, rng, cls_choices=(1, 4, 5, 2))
    gt = [[f, i, x, y, w, h, 1, c, 0, 0] for f, i, x, y, w, h, c in objs]
    gt.append([1, 999, 900, 900, 50, 50, 0, 0, 0, 0])  # ignore region (class 0)
    trk = []
    for f, i, x, y, w, h, c in objs:
        if perturb and rng.random() < 0.1:
            continue
        tid = i + (100 if (perturb and f > n_frames // 2 and i % 5 == 0) else 0)
        jx = rng.normal(0, 2) if perturb else 0.0
        trk.append([f, tid, x + jx, y, w, h, 0.9, c, -1, -1])
    if perturb:
        for f in range(1, n_frames + 1):
            trk.append([f, 500 + f, rng.uniform(0, 800), rng.uniform(0, 500), 30, 30, 0.9, 1, -1, -1])
    _write(tmp / "gt" / f"{seq}.txt", gt)
    return trk


def test_u2mot_perfect_and_agreement(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    seqs = ["seqA", "seqB", "seqC"]
    for s in seqs:
        trk = _visdrone_seq(tmp_path, s, rng)
        _write(tmp_path / "noisy" / f"{s}.txt", trk)
        gt = pd.read_csv(tmp_path / "gt" / f"{s}.txt", header=None).values.tolist()
        perfect = [[r[0], r[1], r[2], r[3], r[4], r[5], 0.9, r[7], -1, -1] for r in gt if r[7] in (1, 4, 5, 6, 9)]
        _write(tmp_path / "perfect" / f"{s}.txt", perfect)

    monkeypatch.setattr(rescore, "U2MOT_TESTDEV_SEQS", seqs)
    args = type("A", (), {})()
    args.gt_dir, args.baseline_dir, args.controller_dir = tmp_path / "gt", tmp_path / "noisy", tmp_path / "perfect"
    args.out_dir, args.bootstrap, args.seed = tmp_path / "out", 200, 0
    rep = rescore.run_u2mot(args, trackeval)

    perf = rep["systems"]["controller"]
    assert perf["motmetrics_overall"]["MOTA"] == pytest.approx(100.0)
    assert perf["trackeval_overall"]["HOTA"] == pytest.approx(100.0)
    assert perf["trackeval_overall"]["MOTA"] == pytest.approx(100.0)

    noisy = rep["systems"]["baseline"]
    # motmetrics and TrackEval CLEAR agree on the same filtered boxes
    assert noisy["trackeval_overall"]["MOTA"] == pytest.approx(noisy["motmetrics_overall"]["MOTA"], abs=1e-9)
    assert noisy["trackeval_overall"]["CLR_FP"] == noisy["motmetrics_overall"]["FP"]
    assert noisy["trackeval_overall"]["CLR_FN"] == noisy["motmetrics_overall"]["FN"]
    assert noisy["trackeval_overall"]["IDF1"] == pytest.approx(noisy["motmetrics_overall"]["IDF1"], abs=1e-9)
    assert noisy["motmetrics_overall"]["GT"] == perf["motmetrics_overall"]["GT"]
    # class 2 GT and predictions are removed by the U2MOT class filter
    assert rep["paired_controller_vs_baseline"]["metrics"]["HOTA"]["delta"] > 0
    assert rep["reproduction_all_match"] is False  # synthetic numbers differ from the archived ones
    assert (tmp_path / "out" / "u2mot_testdev_rescore.json").is_file()


def test_u2mot_ignore_region_removes_boxes():
    idx = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (1, 3)], names=["FrameId", "Id"])
    cols = ["X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility", "unused"]
    gt = pd.DataFrame([[0, 0, 100, 100, 0, 0, 0, 0],       # ignore region (class 0)
                       [10, 10, 10, 10, 1, 1, 0, 0],       # centre inside -> removed
                       [300, 300, 10, 10, 1, 4, 0, 0]],    # kept
                      index=idx, columns=cols)
    ts = pd.DataFrame([[20, 20, 10, 10, 0.9, 1, -1, -1],
                       [300, 300, 10, 10, 0.9, 7, -1, -1],  # class 7 -> removed
                       [300, 300, 10, 10, 0.9, 4, -1, -1]], index=idx, columns=cols)
    g, t = rescore.u2mot_clean_gt_and_ts(OrderedDict(s=gt), OrderedDict(s=ts))
    assert [v[1] for v in g["s"].index.values] == [3]
    assert [v[1] for v in t["s"].index.values] == [3]


def _runner_compute_metrics(gt_root, results_dir, seqs, save_csv):
    """Copy of compute_metrics in the frozen SparseTrack runner."""
    accs, names = [], []
    for seq in seqs:
        g = mm.io.loadtxt(str(Path(gt_root) / seq / "gt" / "gt_val_half.txt"), fmt="mot15-2D", min_confidence=1)
        t = mm.io.loadtxt(str(Path(results_dir) / f"{seq}.txt"), fmt="mot15-2D", min_confidence=-1)
        accs.append(mm.utils.compare_to_groundtruth(g, t, "iou", distth=0.5))
        names.append(seq)
    metrics = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses", "precision", "recall",
               "num_objects"]
    summary = mm.metrics.create().compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    summary.reset_index().rename(columns={"index": "sequence"}).to_csv(save_csv, index=False)


def test_sparsetrack_reproduction_and_pairs(tmp_path):
    rng = np.random.default_rng(2)
    seqs = list(rescore.MOT17_VAL_HALF_FRAMES)
    gt_root = tmp_path / "MOT17" / "train"
    runs = {"static075": tmp_path / "S075" / "NMS_075" / "track_results",
            "adaptive": tmp_path / "AD" / "NMS_070" / "track_results"}
    for s in seqs:
        objs = _objects(8, 20, rng)
        gt = [[f, i, x, y, w, h, 1, 1, 1.0] for f, i, x, y, w, h, _ in objs]
        gt.append([3, 77, 50, 50, 30, 60, 0, 8, 1.0])  # distractor, not counted
        _write(gt_root / s / "gt" / "gt_val_half.txt", gt)
        for k, (name, d) in enumerate(runs.items()):
            trk = [[f, i, x + rng.normal(0, 1.5), y, w, h, 0.9, -1, -1, -1]
                   for f, i, x, y, w, h, _ in objs if rng.random() > 0.05 * (k + 1)]
            _write(d / f"{s}.txt", trk)
    for d in runs.values():
        _runner_compute_metrics(gt_root, d, seqs, d.parent / "metrics.csv")

    args = type("A", (), {})()
    args.gt_root = gt_root
    args.run = [f"{k}={v}" for k, v in runs.items()]
    args.pair = ["adaptive:static075", "static075:static075"]
    args.out_dir, args.bootstrap, args.seed = tmp_path / "out", 300, 0
    rep = rescore.run_sparsetrack(args, trackeval)

    assert rep["reproduction"]["static075"]["all_match"] is True
    assert rep["reproduction"]["adaptive"]["all_match"] is True
    same = rep["pairs"]["static075_vs_static075"]["metrics"]
    for m in same.values():
        assert m["delta"] == 0 and m["ci95"] == [0.0, 0.0] and m["ties"] == len(seqs)
    diff = rep["pairs"]["adaptive_vs_static075"]["metrics"]["MOTA"]
    assert diff["ci95"][0] <= diff["delta"] <= diff["ci95"][1]
    assert diff["wins"] + diff["ties"] + diff["losses"] == len(seqs)
    for name in runs:
        assert 0 < rep["systems"][name]["trackeval_overall"]["HOTA"] <= 100


def test_bootstrap_matches_manual_sum():
    """Combining per-sequence CLEAR results equals MOTA from summed counts."""
    metrics = rescore.te_metric_objects(trackeval)
    per_seq = {}
    rng = np.random.default_rng(3)
    for s in ["a", "b"]:
        n_t = 5
        data = {"num_timesteps": n_t, "num_gt_ids": 2, "num_tracker_ids": 2, "gt_ids": [], "tracker_ids": [],
                "similarity_scores": [], "num_gt_dets": 0, "num_tracker_dets": 0}
        for _ in range(n_t):
            data["gt_ids"].append(np.array([0, 1]))
            keep = rng.random(2) > 0.3
            tids = np.array([0, 1])[keep]
            data["tracker_ids"].append(tids)
            data["similarity_scores"].append(np.eye(2)[:, keep] * 0.9)
            data["num_gt_dets"] += 2
            data["num_tracker_dets"] += int(keep.sum())
        per_seq[s] = rescore.te_eval_sequence(metrics, data)
    comb = rescore.te_headline(rescore.te_combine(metrics, per_seq, ["a", "b"]))
    tp = sum(per_seq[s]["CLEAR"]["CLR_TP"] for s in per_seq)
    fn = sum(per_seq[s]["CLEAR"]["CLR_FN"] for s in per_seq)
    fp = sum(per_seq[s]["CLEAR"]["CLR_FP"] for s in per_seq)
    ids = sum(per_seq[s]["CLEAR"]["IDSW"] for s in per_seq)
    assert comb["MOTA"] == pytest.approx(100 * (1 - (fn + fp + ids) / (tp + fn)))
