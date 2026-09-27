"""
Run a named universal policy over a dataset, from the detection cache
(default) or live on the detector (--live-weights), then evaluate.

Outputs (new directory per run; refuses to overwrite):
  tracks/<seq>.txt, density_audit.csv, summary.csv, metrics.csv, policy.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from adapters.trackers.bytetrack import ByteTrackAdapter
from adapters.types import Detection
from run_universal_acmot import build_config
from tools.eval_local import evaluate
from universal_policy_pipeline import (POLICIES, UniversalPolicyPipeline,
                                       describe, replace)


class CachedDetector:
    """Replays tools/cache_detections.py output for one sequence."""

    # Monotone score transforms emulating detectors with a different
    # confidence calibration (stress test only; default = identity).
    TRANSFORMS = {
        None: lambda s: s,
        "temp2": lambda s: 1 / (1 + np.exp(-np.log(s / (1 - s)) / 2.0)),
        "temp05": lambda s: 1 / (1 + np.exp(-np.log(s / (1 - s)) / 0.5)),
        "scale05": lambda s: 0.5 * s,
        "pow3": lambda s: s ** 3,
    }

    def __init__(self, npz_path, transform=None, cached_nms=0.45):
        self.CACHED_NMS = float(cached_nms)
        z = np.load(npz_path)
        # Task class set (dataset protocol, e.g. UAVDT vehicles) declared
        # in task.json next to the cache files; default = all cached.
        task = Path(npz_path).parent / "task.json"
        self.classes = (set(json.loads(task.read_text())["classes"])
                        if task.exists() else None)
        self.visual = z["visual"]
        vc = Path(npz_path).parent.parent / "visual_cues" / \
            f"{Path(npz_path).stem}.npz"
        self.visual_cues = np.load(vc)["cues"] if vc.exists() else None
        self.shape = tuple(int(v) for v in z["shape"])
        self.frames = int(z["frames"])
        self.by_res = {}
        for r in (640, 736, 832):
            if f"det_{r}" not in z:     # subset caches (transfer tests)
                continue
            a = z[f"det_{r}"].copy()
            if self.classes is not None:
                a = a[np.isin(a[:, 6].astype(int), list(self.classes))]
            if transform is not None:
                sc = np.clip(a[:, 5], 1e-6, 1 - 1e-6)
                a[:, 5] = self.TRANSFORMS[transform](sc)
            self.by_res[r] = {int(f): a[a[:, 0] == f]
                              for f in np.unique(a[:, 0])}
        self.frame = 0

    # The cache stores POST-NMS outputs at this IoU (YOLO) / no NMS
    # (RT-DETR ignores suppression). A policy that requests any other NMS
    # value (e.g. Config.adaptive_nms) cannot be replayed from this cache.
    CACHED_NMS = 0.45

    def visual_dict(self, i):
        """Image statistics for frame i (1-based); motion only if the
        visual-cue cache exists (otherwise NaN, never guessed)."""
        if self.visual_cues is not None:
            e, b, bl, m, r = self.visual_cues[i - 1]
            return dict(edges=e, brightness=b, blur=bl, motion=m,
                        motion_resp=r)
        v = self.visual[i - 1]
        return dict(edges=v[0], brightness=v[1], blur=v[2])

    def detect(self, image, confidence, suppression, resolution):
        if abs(float(suppression) - self.CACHED_NMS) > 1e-9:
            raise RuntimeError(
                f"Cache holds NMS={self.CACHED_NMS} outputs; policy asked "
                f"for NMS={suppression}. Run live instead.")
        rows = self.by_res[int(resolution)].get(self.frame, ())
        return [Detection(x1=float(r[1]), y1=float(r[2]), x2=float(r[3]),
                          y2=float(r[4]), confidence=float(r[5]),
                          class_id=int(r[6]))
                for r in rows if r[5] >= confidence]


def make_tracker(cfg, policy=None, cls=ByteTrackAdapter):
    native = policy is not None and \
        getattr(policy, "tracker_defaults", "ac") == "native"
    return cls(high=cfg.high, low=cfg.low, new=cfg.new,
               buffer=30 if native else cfg.buffer,
               match=0.8 if native else cfg.match, fuse=cfg.fuse)


def run(policy, dataset, cache_dir, output_dir, sequences=None,
        density_kwargs=None, live_weights=None, quiet=False):
    dataset = Path(dataset)
    out = Path(output_dir)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {out}")
    (out / "tracks").mkdir(parents=True)
    json.dump(describe(policy, density_kwargs),
              open(out / "policy.json", "w"), indent=2)

    seqs = sorted(p.name for p in (dataset / "sequences").iterdir()
                  if p.is_dir())
    if sequences:
        seqs = [s for s in seqs if s in sequences]

    live = None
    if live_weights:
        import cv2
        from adapters.detectors.factory import create_detector
        from run_universal_acmot import analyze_visual
        live = create_detector(live_weights, family="auto")

    audit_rows, summary = [], []
    cfg = build_config()
    for seq in seqs:
        t0 = time.perf_counter()
        if live is None:
            det = CachedDetector(Path(cache_dir) / f"{seq}.npz")
            frames = list(range(1, det.frames + 1))
        else:
            det = live
            frames = sorted((dataset / "sequences" / seq).glob("*.jpg"))
        pipe = UniversalPolicyPipeline(cfg, det, make_tracker(cfg, policy),
                                       policy,
                                       density_kwargs=density_kwargs)
        rows = []
        with open(out / "tracks" / f"{seq}.txt", "w") as f:
            for i, item in enumerate(frames, start=1):
                if live is None:
                    det.frame = i
                    visual = det.visual_dict(i)
                    image = np.empty(det.shape + (0,), dtype=np.uint8)
                else:
                    image = cv2.imread(str(item))
                    visual = analyze_visual(image)
                res = pipe.process(i, image, visual)
                for t in res["tracks"]:
                    f.write(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},"
                            f"{max(0.0, t.x2 - t.x1):.3f},"
                            f"{max(0.0, t.y2 - t.y1):.3f},"
                            f"{t.confidence:.6f},{t.class_id},-1,-1\n")
                rows.append(dict(sequence=seq, **res["audit"]))
        audit_rows += rows
        raw = np.array([r["raw_count"] for r in rows], float)
        acc = np.array([r["accepted_after_topk"] for r in rows], float)
        summary.append(dict(
            sequence=seq, frames=len(rows),
            mean_sci=float(np.mean([r["sci"] for r in rows])),
            mean_keep_pct=float(np.mean(np.where(raw > 0, 100 * acc /
                                                 np.maximum(raw, 1), 0))),
            mean_tracks=float(np.mean([r["tracks"] for r in rows])),
            mean_target=float(np.mean([r["target_candidates"]
                                       for r in rows])),
            mean_reliability=float(np.mean([r["reliability"]
                                            for r in rows])),
            resolutions="/".join(str(v) for v in sorted(
                {r["resolution"] for r in rows})),
            seconds=round(time.perf_counter() - t0, 1)))
        if not quiet:
            s = summary[-1]
            print(f"  {seq}: SCI={s['mean_sci']:.3f} "
                  f"keep={s['mean_keep_pct']:.1f}% "
                  f"tracks={s['mean_tracks']:.1f} "
                  f"target={s['mean_target']:.1f} "
                  f"rel={s['mean_reliability']:.3f} ({s['seconds']}s)")

    for name, rows in (("density_audit.csv", audit_rows),
                       ("summary.csv", summary)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    metrics = evaluate(dataset, out / "tracks", seqs)
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0]))
        w.writeheader()
        w.writerows(metrics)
    return summary, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--policy-overrides", default="{}",
                    help="JSON dict of PolicySpec field overrides")
    ap.add_argument("--density-kwargs", default="{}")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache-dir")
    ap.add_argument("--live-weights")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sequences", nargs="*")
    args = ap.parse_args()

    policy = replace(POLICIES[args.policy],
                     **json.loads(args.policy_overrides))
    summary, metrics = run(policy, args.dataset, args.cache_dir,
                           args.output_dir, args.sequences,
                           json.loads(args.density_kwargs),
                           args.live_weights)
    for r in metrics:
        print(f"{r['sequence']:<22} MOTA {r['MOTA']:8.3f} HOTA {r['HOTA']:6.3f}"
              f" IDF1 {r['IDF1']:6.3f} IDS {r['IDS']:5d} FP {r['FP']:6d}"
              f" FN {r['FN']:6d} P {r['Precision']:6.2f} R {r['Recall']:6.2f}")


if __name__ == "__main__":
    main()
