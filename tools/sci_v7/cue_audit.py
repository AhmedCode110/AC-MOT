"""
General-SCI cue audit (development split val-7 only; uses GT for the
target, never inside a controller).

Target per 30-frame segment s of a sequence: the benefit of more detector
compute, b(s) = q_HIGH(s) - q_LOW(s), where q is the per-frame MOTA numerator
(TP - FP - IDS) of the static runs '<layer>+<LEVEL>' summed over s
(tools/sci_v7/oracle.py). Cues are computed causally for the first frame f0
of s from frames f0-10 .. f0-1 (and the image of f0 for image cues), all from
the reference static run '<layer>+MEDIUM' (so the cue never depends on the
level chosen inside s).

Candidate cues (no detector or tracker name, no absolute score threshold,
no absolute pixel-size threshold):
  image   img_edges, img_dark (-gray level), img_blur (-log Laplacian
          variance), img_motion (global motion magnitude), img_motion_unrel
          (1 - phase-correlation response)
  canonical detections (score bands of the score layer, frames < f0)
          det_density   candidates in the primary band per megapixel
          det_small     -median relative size sqrt(box area / image area)
                        of primary-band candidates
          det_ambig     share of passed candidates in the ambiguous band [t1, t2)
  probe (detector-conditioned, label-free; one extra detector pass at
          another level on frame f0-1)
          probe_up      primary-band candidates at the higher level without an
                        IoU >= 0.5 partner among the passed candidates at the
                        reference level, per primary candidate at the reference
          probe_down    the same, reference level vs the lower level
  tracks (tracker-conditioned)
          trk_density   reported tracks per megapixel
          trk_small     -median relative size of reported tracks
          trk_churn     share of track ids at f0-1 absent at f0-10

Statistics: within-sequence Spearman rho(cue, b) per (detector, sequence),
its mean, sign agreement across sequences and detectors, and the pooled
across-sequence Spearman.

  python tools/sci_v7/cue_audit.py <layer> [--levels LOW MEDIUM HIGH] [--json out]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sci_v7 import dev  # noqa: E402  (val-7 guard)
from tools.sci_v7.oracle import frame_quality  # noqa: E402

CUES = ["img_edges", "img_dark", "img_blur", "img_motion", "img_motion_unrel",
        "det_density", "det_small", "det_ambig", "trk_density", "trk_small", "trk_churn",
        "probe_up", "probe_down"]
SEG = dev.SEGMENT


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _novel(A, B, a):
    """Primary-band boxes of A (score layer bands of the previous frame)
    without an IoU >= 0.5 partner among B's passed boxes, per primary box of B."""
    from acmot_v7 import iou_matrix
    pa = A[_logit(A[:, 5]) >= a["t2"]] if len(A) else A
    pb = B[_logit(B[:, 5]) >= a["t1"]] if len(B) else B
    nb = int((_logit(B[:, 5]) >= a["t2"]).sum()) if len(B) else 0
    if not len(pa):
        return 0.0
    if not len(pb):
        return len(pa) / max(nb, 1)
    M = iou_matrix(pa[:, 1:5], pb[:, 1:5])
    return float((M.max(1) < 0.5).sum()) / max(nb, 1)


def segment_table(layer, det, seq, levels, ref="MEDIUM", run_of=None, probe_res=None):
    """Rows (cue values, benefit) for every segment with history."""
    from tools.run_policy_validation import CachedDetector
    run_of = run_of or (lambda lv: f"{layer}+{lv}")
    SPLITS, _ = dev._splits()
    cd = CachedDetector(f"{SPLITS[dev.SPLIT]['native']}/{det}/{seq}.npz")
    H, W = cd.shape
    mp = H * W / 1e6
    lo, hi = levels[0], levels[-1]
    qlo, qhi = frame_quality(run_of(lo), det, seq), frame_quality(run_of(hi), det, seq)
    st = dev.load(run_of(ref), det, seq)
    aud = st["audit"]
    setting = aud[0]["setting"]
    tr = np.loadtxt(io.StringIO(st["tracks_txt"]), delimiter=",", ndmin=2)
    n = len(aud)
    rows = []
    for f0 in range(SEG + 1, n + 1, SEG):
        b = float(qhi[f0 - 1:f0 - 1 + SEG].sum() - qlo[f0 - 1:f0 - 1 + SEG].sum())
        hist = range(f0 - 10, f0)
        v = cd.visual_cues[f0 - 1]
        c = dict(img_edges=v[0], img_dark=-v[1], img_blur=-np.log1p(v[2]), img_motion=v[3],
                 img_motion_unrel=1 - v[4])
        dens, small, amb = [], [], []
        for f in hist:
            a = aud[f - 1]
            dets = cd.by_res[setting].get(f, np.zeros((0, 7)))
            if "t2" not in a or not len(dets):
                continue
            L = _logit(dets[:, 5])
            prim = dets[L >= a["t2"]]
            dens.append(len(prim) / mp)
            if len(prim):
                small.append(-np.median(np.sqrt((prim[:, 3] - prim[:, 1]) * (prim[:, 4] - prim[:, 2]) / (H * W))))
            passed = L >= a["t1"]
            amb.append(float(((L >= a["t1"]) & (L < a["t2"])).sum() / max(passed.sum(), 1)))
        c["det_density"] = np.mean(dens) if dens else np.nan
        c["det_small"] = np.mean(small) if small else np.nan
        c["det_ambig"] = np.mean(amb) if amb else np.nan
        t_last = tr[tr[:, 0] == f0 - 1] if len(tr) else np.zeros((0, 10))
        t_old = tr[tr[:, 0] == f0 - 10] if len(tr) else np.zeros((0, 10))
        c["trk_density"] = np.mean([(tr[:, 0] == f).sum() for f in hist]) / mp if len(tr) else 0.0
        c["trk_small"] = (-np.median(np.sqrt(t_last[:, 4] * t_last[:, 5] / (H * W)))
                          if len(t_last) else np.nan)
        c["trk_churn"] = (float(np.mean(~np.isin(t_last[:, 1], t_old[:, 1]))) if len(t_last) else np.nan)
        a = aud[f0 - 2]
        if "t2" in a and probe_res is not None:
            lo_r, hi_r = probe_res
            ref = cd.by_res[setting].get(f0 - 1, np.zeros((0, 7)))
            c["probe_up"] = _novel(cd.by_res[hi_r].get(f0 - 1, np.zeros((0, 7))), ref, a)
            c["probe_down"] = _novel(ref, cd.by_res[lo_r].get(f0 - 1, np.zeros((0, 7))), a)
        else:
            c["probe_up"] = c["probe_down"] = np.nan
        rows.append(dict(seq=seq, det=det, f0=f0, benefit=b, **{k: float(c[k]) for k in CUES}))
    return rows


def spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def audit(layer, levels, dets):
    _, seqs_of = dev._splits()
    prof = json.loads((ROOT / "configs/sci_v7_profiles.json").read_text())["detectors"]
    rows = [r for d in dets for s in seqs_of(dev.SPLIT)
            for r in segment_table(layer, d, s, levels, probe_res=(prof[d]["resolution"]["LOW"],
                                                                  prof[d]["resolution"]["HIGH"]))]
    out = dict(layer=layer, levels=levels, segments=len(rows), cues={})
    for c in CUES:
        per = {}
        for d in dets:
            for s in seqs_of(dev.SPLIT):
                R = [r for r in rows if r["det"] == d and r["seq"] == s]
                per[f"{d}/{s}"] = spearman(np.array([r[c] for r in R]), np.array([r["benefit"] for r in R]))
        vals = np.array([v for v in per.values() if np.isfinite(v)])
        bydet = {d: float(np.nanmean([v for k, v in per.items() if k.startswith(d + "/")])) for d in dets}
        pooled = {d: spearman(np.array([r[c] for r in rows if r["det"] == d]),
                              np.array([r["benefit"] for r in rows if r["det"] == d])) for d in dets}
        out["cues"][c] = dict(mean_within=float(vals.mean()) if len(vals) else np.nan,
                              positive=int((vals > 0).sum()), negative=int((vals < 0).sum()),
                              by_detector=bydet, pooled_by_detector=pooled, per_cell=per)
    return out, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("layer")
    ap.add_argument("--levels", nargs=3, default=["LOW", "MEDIUM", "HIGH"])
    ap.add_argument("--json")
    a = ap.parse_args()
    dets = os.environ.get("V7_DETS", "yolov8,rtdetr").split(",")
    res, rows = audit(a.layer, a.levels, dets)
    print(f"{'cue':<18}{'mean rho':>9}{'+/-':>7}  " + "  ".join(f"{d}: within / pooled" for d in dets))
    for c, v in res["cues"].items():
        print(f"{c:<18}{v['mean_within']:+9.3f}{v['positive']:>4}/{v['negative']:<3} " + "  ".join(
            f"{v['by_detector'][d]:+.3f} / {v['pooled_by_detector'][d]:+.3f}" for d in dets))
    if a.json:
        Path(a.json).write_text(json.dumps(dict(result=res, rows=rows), indent=1, default=float))
