#!/bin/zsh
# Mac/MPS cache queue for V5-TF development & transfer (CACHING ONLY, no metrics).
cd /Users/ahmedgouda/Desktop/Universal-ACMOT
export PYTHONPATH=$(pwd) OMP_NUM_THREADS=2
PY=.venv/bin/python
Y=/Users/ahmedgouda/Desktop/CUE_SELECTION/yolov8n.pt
W=weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
TR="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@gmail.com/My Drive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-train"
V=/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val
T="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@gmail.com/.shortcut-targets-by-id/1IvH3DmlX4Ce5k2cZWDvu0ixfbvxZ-67m/visdrone goda1/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev"
c() { $PY -u tools/cache_detections.py "$@" 2>&1 | grep -v -i warn; }
# development (train): V4 reference (0.45) + native (0.7) YOLO, RT-DETR (NMS-free), image cues
c --weights $Y --dataset "$TR" --output-dir outputs/det_cache_train/yolov8 --resolutions 736
$PY tools/visual_cues.py "$TR" outputs/det_cache_train | tail -1
c --weights rtdetr-l.pt --dataset "$TR" --output-dir outputs/det_cache_train/rtdetr --resolutions 736
c --weights $Y --dataset "$TR" --output-dir outputs/det_cache_train_native/yolov8 --nms 0.7 --resolutions 736
echo "TRAIN CACHES DONE $(date +%T)"
# transfer sets, detector-native suppression (Faster R-CNN 0.5, YOLO 0.7)
c --weights $W --dataset $V --output-dir outputs/det_cache_val_native/fasterrcnn --nms 0.5 --resolutions 640 736
c --weights $Y --dataset outputs/uavdt_view --output-dir outputs/det_cache_uavdt_native/yolov8 --nms 0.7 --resolutions 640 736
c --weights $W --dataset outputs/uavdt_view --output-dir outputs/det_cache_uavdt_native/fasterrcnn --nms 0.5 --resolutions 736
c --weights $W --dataset "$T" --output-dir outputs/det_cache_testdev_native/fasterrcnn --nms 0.5 --resolutions 640 736
c --weights $Y --dataset "$T" --output-dir outputs/det_cache_testdev_native/yolov8 --nms 0.7 --resolutions 640 736
$PY tools/visual_cues.py "$T" outputs/det_cache_testdev | tail -1
$PY tools/visual_cues.py outputs/uavdt_view outputs/det_cache_uavdt | tail -1
echo "ALL V5-TF CACHES DONE $(date +%T)"
