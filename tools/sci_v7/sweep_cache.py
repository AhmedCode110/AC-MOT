"""Reader of the resolution-sweep cache (tools/sci_v7/build_sweep_cache.py)
with the interface of tools.run_policy_validation.CachedDetector. Image cues
(edges, brightness, blur, motion) come from the V7-record visual-cue cache:
they are computed from the frames, not from the detector."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from adapters.types import Detection


class SweepCache:
    def __init__(self, root, det, seq, visual_cues_npz):
        self.by_res = {}
        shape = frames = None
        for d in sorted(Path(root, det).iterdir()):
            f = d / f"{seq}.npz"
            if not (d.name.isdigit() and f.exists()):
                continue
            z = np.load(f)
            a = z["det"]
            self.by_res[int(d.name)] = {int(k): a[a[:, 0] == k] for k in np.unique(a[:, 0])}
            shape, frames = tuple(int(v) for v in z["shape"]), int(z["frames"])
        if not self.by_res:
            raise FileNotFoundError(f"no sweep cache for {det}/{seq} in {root}")
        self.shape, self.frames = shape, frames
        self.visual_cues = np.load(visual_cues_npz)["cues"]
        self.frame = 0

    def visual_dict(self, i):
        e, b, bl, m, r = self.visual_cues[i - 1]
        return dict(edges=e, brightness=b, blur=bl, motion=m, motion_resp=r)

    def detect(self, image, confidence, suppression, resolution):
        if suppression is not None:
            raise RuntimeError("the sweep cache holds detector-native suppression only")
        rows = self.by_res[int(resolution)].get(self.frame, ())
        return [Detection(x1=float(r[1]), y1=float(r[2]), x2=float(r[3]), y2=float(r[4]),
                          confidence=float(r[5]), class_id=int(r[6])) for r in rows if r[5] >= confidence]
