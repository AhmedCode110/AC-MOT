"""
Run the official TrackTrack feature extraction (`2. FastReID/ext_feats.py`,
kamkyu94/TrackTrack @ ee7f1c5) on CPU. Compatibility only: `.cuda()` no-op,
`.half()` keeps float32 (the authors ran CUDA float16), torch.load to CPU,
FastReID MODEL.DEVICE cpu. Arguments are passed through unchanged.

  python tracktrack_feats.py --data_path <split dir>/ --pickle_path <det.pickle> --output_path <det_feat.pickle> \
      --config_path configs/DanceTrack/sbs_S50.yml --weight_path weights/dance_sbs_S50.pth
"""
import os
import runpy
import sys
from pathlib import Path

import torch

FR = Path(os.environ.get("TRACKTRACK_ROOT", Path.home() / "acmot_work/acmot_external/TrackTrack")) / "2. FastReID"
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
torch.Tensor.half = lambda self, *a, **k: self
torch.nn.Module.half = lambda self, *a, **k: self
_load = torch.load


def _cpu_load(*a, **k):
    k.setdefault("map_location", "cpu")
    k.setdefault("weights_only", False)
    return _load(*a, **k)


torch.load = _cpu_load
torch.set_num_threads(os.cpu_count() or 4)
os.chdir(FR)
sys.path.insert(0, str(FR))
from fastreid.config import defaults  # noqa: E402
defaults._C.MODEL.DEVICE = "cpu"
sys.argv = ["ext_feats.py"] + sys.argv[1:]
runpy.run_path(str(FR / "ext_feats.py"), run_name="__main__")
