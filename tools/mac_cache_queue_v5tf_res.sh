#!/bin/zsh
# Mac/MPS cache queue for Amendment-7 R-res (CACHING ONLY, no metrics):
# 640/832 train caches for YOLOv8n (native NMS 0.7) and RT-DETR-L (NMS-free).
cd /Users/ahmedgouda/Desktop/Universal-ACMOT
export PYTHONPATH=$(pwd) OMP_NUM_THREADS=2
PY=.venv/bin/python
Y=/Users/ahmedgouda/Desktop/CUE_SELECTION/yolov8n.pt
TR="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@gmail.com/My Drive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-train"
c() { $PY -u tools/cache_detections.py "$@" 2>&1 | grep -v -i warn; }
c --weights $Y --dataset "$TR" --output-dir outputs/det_cache_train_res_extra/yolov8 --nms 0.7 --resolutions 640 832
c --weights rtdetr-l.pt --dataset "$TR" --output-dir outputs/det_cache_train_res_extra/rtdetr --resolutions 640 832
echo "RES CACHES DONE $(date +%T)"
