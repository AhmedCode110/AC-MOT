# Reproducing Universal AC-MOT (V4)

Final method: tag `universal-acmot-v4-freeze` (policy file
`configs/universal_acmot_policy_v4.json`). Held-out and transfer locks:
`research/TESTDEV_LOCK_V4.json`, `research/TRANSFER_LOCK_*.json`.

Evaluation is the internal class-agnostic protocol (HOTA: TrackEval
12c8791; other metrics: motmetrics 1.4.0) — **not** official VisDrone.

## Environment
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt motmetrics
```
TrackEval checkout at commit 12c8791 (path set in `tools/eval_local.py`).

## Plug-and-play use
```python
from adapters.detectors.factory import create_detector
from adapters.trackers.bytetrack import ByteTrackAdapter
from universal_acmot import UniversalACMOT

system = UniversalACMOT(create_detector("rtdetr-l.pt"),
                        ByteTrackAdapter(buffer=45, match=0.86),
                        resolution=736)            # or resolution="auto", target_fps=30
for frame in frames:
    tracks = system(frame)
```

## One-command pipelines
1. Detection caches (inference only):
   `PYTHONPATH=. .venv/bin/python tools/cache_detections.py --weights W --dataset D --output-dir outputs/det_cache/<det>`
2. Development selection (val only): `ACMOT_GRID=v4 tools/optimize_protocol.py run` then `tools/v4_select.py`
3. Held-out VisDrone test-dev (once):
   `PYTHONPATH=. .venv/bin/python tools/heldout_run.py --lock research/TESTDEV_LOCK_V4.json --dataset <test-dev> --cache-root outputs/det_cache_testdev --out outputs/heldout_v4`
4. Transfer: same command with `research/TRANSFER_LOCK_FASTERRCNN.json` /
   `research/TRANSFER_LOCK_UAVDT.json` (UAVDT view: `tools/build_uavdt_view.py`).

Cache replay fidelity: live == replay byte-identical (RT-DETR V2cA and V4
on 0137), V1/V2b val-7 reproduce Colab metrics exactly.

## Official timing (Colab T4 only)
```
%cd /content/Universal-ACMOT
!git fetch origin && git checkout universal-adapters-v1 && git pull
# the timing harness was added after the freeze tag; verify the policy
# files still match the lock before timing:
!python -c "import json,hashlib;L=json.load(open('research/TESTDEV_LOCK_V4.json'));[print(p, hashlib.sha256(open(p,'rb').read()).hexdigest()==h) for p,h in L['file_sha256'].items()]"
!pip -q install motmetrics
!PYTHONPATH=. python tools/t4_benchmark.py --dataset /content/VisDrone2019-MOT-val \
   --weights yolov8n.pt rtdetr-l.pt weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth \
   --levels 640 736 832 --trackers bytetrack botsort --frames 300 \
   --out /content/drive/MyDrive/Universal-ACMOT-Results/t4_benchmark_v4.json
```
Reports detector / tracker / adaptive-layer time (mean, P95), FPS,
adaptive share of runtime and peak CUDA memory.
