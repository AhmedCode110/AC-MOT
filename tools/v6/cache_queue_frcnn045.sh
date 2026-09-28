#!/bin/zsh
# Faster R-CNN caches at NMS 0.45 / 736 so the frozen V4 reference (which
# requests NMS 0.45) can be replayed exactly. CACHING ONLY, no metrics.
cd /Users/ahmedgouda/Desktop/Universal-ACMOT
export PYTHONPATH=$(pwd) OMP_NUM_THREADS=2
PY=.venv/bin/python
W=weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
V=/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val
T="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@gmail.com/.shortcut-targets-by-id/1IvH3DmlX4Ce5k2cZWDvu0ixfbvxZ-67m/visdrone goda1/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev"
c() { $PY -u tools/cache_detections.py "$@" 2>&1 | grep -v -i warn; }
c --weights $W --dataset $V --output-dir outputs/det_cache/fasterrcnn --nms 0.45 --resolutions 736
c --weights $W --dataset "$T" --output-dir outputs/det_cache_testdev/fasterrcnn --nms 0.45 --resolutions 736
c --weights $W --dataset outputs/uavdt_view --output-dir outputs/det_cache_uavdt/fasterrcnn --nms 0.45 --resolutions 736
echo QUEUE_DONE
