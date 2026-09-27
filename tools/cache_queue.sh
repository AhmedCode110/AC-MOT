#!/bin/zsh
# Sequential MPS cache queue (priority order). Inference only; no metrics.
cd /Users/ahmedgouda/Desktop/Universal-ACMOT
export PYTHONPATH=$(pwd) OMP_NUM_THREADS=2
PY=.venv/bin/python
W=weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
Y=/Users/ahmedgouda/Desktop/CUE_SELECTION/yolov8n.pt
V=/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val
T="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@gmail.com/.shortcut-targets-by-id/1IvH3DmlX4Ce5k2cZWDvu0ixfbvxZ-67m/visdrone goda1/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev"
U=outputs/uavdt_view
until [ $(ls outputs/det_cache_testdev/rtdetr/*.npz 2>/dev/null | wc -l) -ge 17 ]; do sleep 30; done
echo "QUEUE start $(date +%T)"
$PY -u tools/cache_detections.py --weights $Y --dataset $U --output-dir outputs/det_cache_uavdt/yolov8 --resolutions 640 736
$PY -u tools/cache_detections.py --weights rtdetr-l.pt --dataset $U --output-dir outputs/det_cache_uavdt/rtdetr --resolutions 640 736
$PY -u tools/cache_detections.py --weights $W --dataset $V --output-dir outputs/det_cache/fasterrcnn --resolutions 640 736
$PY -u tools/cache_detections.py --weights $W --dataset "$T" --output-dir outputs/det_cache_testdev/fasterrcnn --resolutions 640 736
$PY -u tools/cache_detections.py --weights $W --dataset $U --output-dir outputs/det_cache_uavdt/fasterrcnn --resolutions 736
echo "QUEUE done $(date +%T)"
