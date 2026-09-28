"""Runs the OFFICIAL SparseTrack track.py (mot17_ab_track_cfg.py) on Apple MPS
and records, WITHOUT changing behaviour:
  - the published detector output per frame (for the identical-input +V6 arm)
  - per-frame detector and tracker wall time (torch.mps.synchronize)."""
import pickle, sys, time
from pathlib import Path
import numpy as _np  # NUMPY2-COMPAT: removed aliases (behaviour-identical builtins)
for _n, _v in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(_np, _n):
        setattr(_np, _n, _v)
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tracker"))   # pbcvt shim location

REC, TIMES = {}, []
import models.yolox as Y
_orig_fwd = Y.YOLOX.forward
def _fwd(self, batch_data):
    if self.training:
        return _orig_fwd(self, batch_data)
    torch.mps.synchronize(); t0 = time.perf_counter()
    r = _orig_fwd(self, batch_data)
    torch.mps.synchronize(); dt = time.perf_counter() - t0
    inst = r[0]["instances"]
    key = "/".join(Path(batch_data[0]["file_name"]).parts[-3:])
    REC[key] = np.hstack([inst.pred_boxes.tensor.float().cpu().numpy(),
                          inst.scores.float().cpu().numpy()[:, None]]).astype(np.float32)
    TIMES.append(dict(key=key, t_det=dt))
    return r
Y.YOLOX.forward = _fwd

import tracker.sparse_tracker as STK
_orig_upd = STK.SparseTracker.update
def _upd(self, output_results, curr_img=None):
    t0 = time.perf_counter(); r = _orig_upd(self, output_results, curr_img)
    TIMES[-1]["t_trk"] = time.perf_counter() - t0
    return r
STK.SparseTracker.update = _upd

import track
from detectron2.config import LazyConfig as _LC
_orig_load = _LC.load
def _load(*a, **k):
    cfg = _orig_load(*a, **k)
    # same runtime patch as the pinned deterministic protocol (reproducibility doc):
    # the LazyConfig loader's own `opt` controls workers; 0 workers (no behaviour change)
    cfg.dataloader.test._target_.__globals__["opt"].DATALOADER.NUM_WORKERS = 0
    return cfg
_LC.load = staticmethod(_load)
track.LazyConfig.load = staticmethod(_load)
from detectron2.engine import default_argument_parser
args = default_argument_parser(epilog="SparseTrack Eval").parse_args()
try:
    track.main(args)
finally:
    od = [o.split("=", 1)[1] for o in args.opts if o.startswith("train.output_dir=")]
    out = Path(od[0]) if od else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    pickle.dump(REC, open(out / "published_detections.pkl", "wb"))
    pickle.dump(TIMES, open(out / "timing.pkl", "wb"))
    print("saved", len(REC), "frames of detections to", out)
