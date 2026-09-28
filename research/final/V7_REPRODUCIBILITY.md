# V7 reproducibility

Frozen policy: `configs/universal_acmot_policy_v7.json` (V7f), lock
`research/V7_POLICY_LOCK.json` (sha256 of the layer, config, registry and
runners; checked by `tests/test_v7_adaptive_layer.py::test_v7_policy_lock_integrity`),
freeze commit 488df9a (tag `universal-acmot-v7-freeze`; push pending by the
owner — the cloud git proxy refuses tags).

## Environment
- `bash scripts/setup_research_assets.sh envs caches external repos gmc paths`
  (Python 3.12 venvs; release `v7-dev-assets-1` caches, sha256 verified).
- Hybrid-SORT only: Python 3.11 venv with numpy 1.23.5, scipy 1.10.1,
  filterpy 1.4.5, lap 0.5.12 (the unmodified tracker needs numpy < 1.24).
- `source $ACMOT_WORK/acmot_env.sh`; `PYTHONPATH=.:$ACMOT_TRACKEVAL`.
- Hardware of every number: `V7_CLOUD_RUNS.md` (4 vCPU Xeon, no GPU).

## External (post-freeze)
```
git clone https://github.com/Wangyc2000/PD_SORT $ACMOT_EXT/PD_SORT   # @af21db6
git clone https://github.com/ymzis69/HybridSORT $ACMOT_EXT/HybridSORT # @396f8d3
.venv/bin/python tools/v7/external/pdsort_v7.py --system BASELINE --name PD_BASELINE
.venv/bin/python tools/v7/external/pdsort_v7.py --system V7f --name PD_V7f
PYTHONPATH= ~/acmot_work/hs_venv/bin/python tools/v7/external/hybridsort_v7.py --system BASELINE --name HS_BASELINE
PYTHONPATH= ~/acmot_work/hs_venv/bin/python tools/v7/external/hybridsort_v7.py --system V7f --name HS_V7f
$ACMOT_EXT/venv/bin/python tools/v7/mot17_eval_v7.py $ACMOT_EXT/runs/pdsort PD_BASELINE PD_V7f
$ACMOT_EXT/venv/bin/python tools/v7/mot17_eval_v7.py --boot $ACMOT_EXT/runs/pdsort PD_BASELINE PD_V7f
```

## Development matrix
```
# MOT17 hosts (streams st = floor 0.01, bt = floor 0.1)
.venv/bin/python tools/v7/external/mot17_bytetrack_v7.py --system V7f --stream st --host official --name BY_official_st_V7f
$ACMOT_EXT/venv/bin/python tools/v7/external/boosttrack_v7.py --system V7f --name BT7C_V7f_pf --pixel-free
# KITTI (frames from the official zip: tools/v7/kitti/remotezip.py; caches: tools/cache_detections.py)
V7_SPLIT=kitti V7_DETS=rtdetr .venv/bin/python tools/v7/dev.py track NATIVE V7f "V7f@trk:ocsort"
.venv/bin/python tools/v7/kitti/kitti_eval.py rtdetr NATIVE V7f
.venv/bin/python tools/v7/kitti/kitti_eval.py --boot rtdetr NATIVE V7f
# tests
.venv/bin/python -m pytest -q tests/test_v7_adaptive_layer.py tests/test_v7_bootstrap.py
# runtime
.venv/bin/python tools/v7/benchmark.py --weights yolov8n.pt --seqs 0001 0009 0019 --out research/final/V7_REALTIME_yolov8n.json
```
Every code change after the freeze would create a new version (V8), never a
new V7 result.
