"""
Merge per-sequence detection caches that hold different resolution levels of
the SAME frozen detector/settings into one NPZ per sequence (det_<level> keys),
so replay can choose the level per frame (Amendment 7 R-res). Caching only.

  python tools/merge_cache_levels.py BASE_DIR EXTRA_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np


def main(base, extra, out):
    base, extra, out = Path(base), Path(extra), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for e in sorted(extra.glob("*.npz")):
        b, dest = base / e.name, out / e.name
        if dest.exists() or not b.exists():
            continue
        zb, ze = np.load(b), np.load(e)
        assert int(zb["frames"]) == int(ze["frames"]), e.name
        assert tuple(zb["shape"]) == tuple(ze["shape"]), e.name
        dets = {k: zb[k] for k in zb.files if k.startswith("det_")}
        dets.update({k: ze[k] for k in ze.files if k.startswith("det_")})
        np.savez_compressed(dest, visual=zb["visual"], shape=zb["shape"],
                            frames=zb["frames"], **dets)
        n += 1
    print(f"merged {n} sequences into {out} (total {len(list(out.glob('*.npz')))})")


if __name__ == "__main__":
    main(*sys.argv[1:4])
