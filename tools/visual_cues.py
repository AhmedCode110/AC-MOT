"""Cache per-frame image cues (edges, brightness, blur, global motion via
phase correlation, motion response) exactly as scene_state.image_stats
computes them live. Written to <cache_root>/visual_cues/<seq>.npz."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def one(args):
    seqdir, out = args
    from scene_state import image_stats
    if Path(out).exists():
        return
    prev = None
    rows = []
    for fp in sorted(Path(seqdir).glob("*.jpg")):
        st, prev = image_stats(cv2.imread(str(fp)), prev)
        rows.append([st[k] for k in ("edges", "brightness", "blur", "motion",
                                     "motion_resp")])
    np.savez_compressed(out, cues=np.asarray(rows, dtype=np.float64))


if __name__ == "__main__":
    dataset, cache_root = Path(sys.argv[1]), Path(sys.argv[2])
    (cache_root / "visual_cues").mkdir(parents=True, exist_ok=True)
    jobs = [(p, cache_root / "visual_cues" / f"{p.name}.npz")
            for p in sorted((dataset / "sequences").iterdir()) if p.is_dir()]
    with ProcessPoolExecutor(4) as ex:
        list(ex.map(one, jobs))
    print("visual cues:", len(jobs))
