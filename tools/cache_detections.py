"""
Cache raw detector outputs (confidence floor 0.01, NMS 0.45) for every
frame at every AC resolution (640/736/832).

The detector is deterministic given (frame, resolution), and AC-MOT only
chooses among these three resolutions with fixed NMS, so a policy run
replayed from this cache is equivalent to a live run. Use
tools/run_policy_validation.py --live on a sequence to verify that.

Also stores analyze_visual() per frame so replays skip image decoding.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from adapters.detectors.factory import create_detector
from run_universal_acmot import analyze_visual

RESOLUTIONS = (640, 736, 832)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sequences", nargs="*")
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--resolutions", type=int, nargs="+",
                    default=list(RESOLUTIONS))
    args = ap.parse_args()

    detector = create_detector(args.weights, family="auto")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p for p in (Path(args.dataset) / "sequences").iterdir()
                  if p.is_dir())
    if args.sequences:
        seqs = [s for s in seqs if s.name in args.sequences]

    print("Adapter:", detector.__class__.__name__, "device:",
          getattr(detector, "device", "?"))
    for seq in seqs:
        target = out / f"{seq.name}.npz"
        if target.exists():
            print("skip", seq.name)
            continue
        frames = sorted(seq.glob("*.jpg"))
        visual = np.zeros((len(frames), 3), dtype=np.float64)
        rows = {r: [] for r in args.resolutions}
        shape = None
        t0 = time.perf_counter()
        for i, fp in enumerate(frames, start=1):
            image = cv2.imread(str(fp))
            shape = image.shape[:2]
            v = analyze_visual(image)
            visual[i - 1] = (v["edges"], v["brightness"], v["blur"])
            for r in args.resolutions:
                dets = detector.detect(image, confidence=args.floor,
                                       suppression=args.nms, resolution=r)
                for d in dets:
                    rows[r].append((i, d.x1, d.y1, d.x2, d.y2,
                                    d.confidence, d.class_id))
        payload = {f"det_{r}": np.asarray(rows[r], dtype=np.float64)
                   .reshape(-1, 7) for r in args.resolutions}
        np.savez_compressed(target, visual=visual,
                            shape=np.asarray(shape), frames=len(frames),
                            **payload)
        dt = time.perf_counter() - t0
        print(f"{seq.name}: {len(frames)} frames, {dt:.0f}s "
              f"({1000 * dt / len(frames) / len(args.resolutions):.0f} ms/det)")


if __name__ == "__main__":
    main()
