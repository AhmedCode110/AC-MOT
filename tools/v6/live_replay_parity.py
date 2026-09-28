#!/usr/bin/env python3
"""V6-TF live == replay parity (development data, no GT metric).

Live path: the real detector on the image (UniversalACMOT wrapper, image
statistics computed from the frames). Replay path: the detection cache and
the cached visual cues through UniversalPolicyPipeline. Same policy file.
Reported separately: (a) detection-level agreement of live vs cached raw
candidates, (b) exact equality of tracks and control decisions.

  python tools/v6/live_replay_parity.py [--frames 20] [--policy configs/...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VAL = Path("/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val")
CACHE = ROOT / "outputs/det_cache_val_native"
WEIGHTS = {"yolov8": "/Users/ahmedgouda/Desktop/CUE_SELECTION/yolov8n.pt",
           "rtdetr": str(ROOT / "rtdetr-l.pt")}
SEQS = ["uav0000086_00000_v", "uav0000268_05773_v"]
KEYS = ("raw_count", "accepted_after_topk", "tracks", "tf_otsu_t1", "tf_otsu_t2",
        "tf_band_lo", "tf_n_primary", "tf_n_secondary", "tf_motion_ratio")


def rows(tracks):
    return [[t.track_id, t.x1, t.y1, t.x2, t.y2, t.confidence, t.class_id] for t in tracks]


def same(a, b, atol):
    if len(a) != len(b):
        return False
    return not a or bool(np.allclose(np.asarray(a, float), np.asarray(b, float),
                                     atol=atol, rtol=0, equal_nan=True))


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from adapters.detectors.factory import create_detector
    from adapters.trackers.bytetrack import ByteTrackAdapter
    from run_universal_acmot import build_config
    from tools.run_policy_validation import CachedDetector, make_tracker
    from ultralytics.trackers.basetrack import BaseTrack
    from universal_acmot import UniversalACMOT, load_policy
    from universal_policy_pipeline import UniversalPolicyPipeline

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--policy", default="configs/universal_acmot_policy_v6tf.json")
    ap.add_argument("--out", default="outputs/v6/live_replay_parity.json")
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    policy, dk = load_policy(ROOT / a.policy)
    cfg = build_config()
    cases, ok_all = [], True
    for det in ("yolov8", "rtdetr"):
        live_det = create_detector(WEIGHTS[det], family="auto")
        for seq in SEQS:
            cache = CachedDetector(str(CACHE / det / f"{seq}.npz"))
            frames = sorted((VAL / "sequences" / seq).glob("*.jpg"))[:a.frames]
            BaseTrack.reset_id()
            live = UniversalACMOT(live_det, ByteTrackAdapter(), policy=policy,
                                  density_kwargs=dk)
            L = []
            for f in frames:
                img = cv2.imread(str(f))
                tr = live(img)
                L.append((rows(tr), dict(live.last["audit"]),
                          [[d.x1, d.y1, d.x2, d.y2, d.confidence]
                           for d in live.last["raw_detections"]]))
            BaseTrack.reset_id()
            rep = UniversalPolicyPipeline(cfg, cache, make_tracker(cfg, policy), policy,
                                          density_kwargs=dk)
            R = []
            for i, f in enumerate(frames, 1):
                img = cv2.imread(str(f))
                cache.frame = i
                r = rep.process(i, img, cache.visual_dict(i))
                R.append((rows(r["tracks"]), dict(r["audit"]),
                          [[d.x1, d.y1, d.x2, d.y2, d.confidence]
                           for d in r["raw_detections"]]))
            det_eq = [same(l[2], r[2], 1e-3) for l, r in zip(L, R)]
            trk_eq = [same(l[0], r[0], 1e-3) for l, r in zip(L, R)]
            aud = []
            for i, (l, r) in enumerate(zip(L, R), 1):
                bad = [k for k in KEYS if not np.isclose(
                    float(l[1].get(k, np.nan)), float(r[1].get(k, np.nan)),
                    atol=1e-6, rtol=0, equal_nan=True)]
                if bad:
                    aud.append({"frame": i, "keys": bad})
            passed = all(det_eq) and all(trk_eq) and not aud
            ok_all &= passed
            cases.append(dict(detector=det, sequence=seq, frames=len(frames),
                              raw_candidates_equal_frames=int(sum(det_eq)),
                              tracks_equal_frames=int(sum(trk_eq)),
                              audit_mismatches=aud, passed=passed))
            print(cases[-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        check="V6-TF live (real detector + image cues) vs cache replay",
        policy=a.policy, data_role="VisDrone2019-MOT-val development (no GT metric)",
        tolerance="boxes/scores 1e-3, decisions 1e-6", passed=bool(ok_all),
        cases=cases), indent=1) + "\n")
    print("PASSED" if ok_all else "FAILED")


if __name__ == "__main__":
    main()
