#!/bin/zsh
# Mac/MPS cache queue for V5 (Colab not reachable). CACHING ONLY — no evaluation.
cd /Users/ahmedgouda/Desktop/Universal-ACMOT
export PYTHONPATH=$(pwd) OMP_NUM_THREADS=2
PY=.venv/bin/python
TR="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@gmail.com/My Drive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-train"
V=/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val
T="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@gmail.com/.shortcut-targets-by-id/1IvH3DmlX4Ce5k2cZWDvu0ixfbvxZ-67m/visdrone goda1/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev"
W=weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
echo "prefetch train images $(date +%T)"
find "$TR/sequences" -name "*.jpg" -print0 | xargs -0 -P 24 -n 25 cat > /dev/null
echo "prefetch done $(date +%T)"
$PY -u tools/cache_detections.py --weights /Users/ahmedgouda/Desktop/CUE_SELECTION/yolov8n.pt --dataset "$TR" --output-dir outputs/det_cache_train/yolov8 --resolutions 736 2>&1 | grep -v -i warn
$PY tools/visual_cues.py "$TR" outputs/det_cache_train 2>&1 | tail -1
$PY -u tools/cache_detections.py --weights rtdetr-l.pt --dataset "$TR" --output-dir outputs/det_cache_train/rtdetr --resolutions 736 2>&1 | grep -v -i warn
echo "TRAIN CACHES DONE $(date +%T)"
# Faster R-CNN (unseen detector): exactly the levels required by the transfer locks
$PY -u tools/cache_detections.py --weights $W --dataset $V --output-dir outputs/det_cache/fasterrcnn --resolutions 640 736 2>&1 | grep -v -i warn
$PY -u tools/cache_detections.py --weights $W --dataset $V --output-dir outputs/det_cache_nms050/fasterrcnn --nms 0.5 --resolutions 736 2>&1 | grep -v -i warn
$PY -u tools/cache_detections.py --weights $W --dataset "$T" --output-dir outputs/det_cache_testdev/fasterrcnn --resolutions 640 736 2>&1 | grep -v -i warn
$PY -u tools/cache_detections.py --weights $W --dataset outputs/uavdt_view --output-dir outputs/det_cache_uavdt/fasterrcnn --resolutions 736 2>&1 | grep -v -i warn
$PY tools/visual_cues.py "$T" outputs/det_cache_testdev 2>&1 | tail -1
$PY tools/visual_cues.py outputs/uavdt_view outputs/det_cache_uavdt 2>&1 | tail -1
echo "ALL V5 CACHES DONE $(date +%T)"
