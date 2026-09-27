#!/bin/zsh
# Copy caches produced by notebooks/Colab_T4_session.ipynb (Part A) from the
# synced Drive folder into the local cache layout. SYNC ONLY — runs no
# evaluation (Faster R-CNN / UAVDT stay untouched until the final freeze).
cd /Users/ahmedgouda/Desktop/Universal-ACMOT
SRC="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@gmail.com/My Drive/Universal-ACMOT-Results/t4_caches"
copy_set() {  # $1 src subdir  $2 expected files  $3 dest
  until [ $(ls "$SRC/$1/"*.npz 2>/dev/null | wc -l) -ge $2 ]; do sleep 60; done
  sleep 30
  .venv/bin/python -c "import numpy as np,glob,sys; [np.load(f).files for f in glob.glob(sys.argv[1]+'/*.npz')]" "$SRC/$1" || { echo "BAD $1"; return 1; }
  mkdir -p "$3"; cp "$SRC/$1/"*.npz "$3/"; echo "synced $1 -> $3 ($(ls $3/*.npz | wc -l)) $(date +%T)"
}
NTRAIN=${NTRAIN:-56}
copy_set train/yolov8 $NTRAIN outputs/det_cache_train/yolov8
copy_set train/rtdetr $NTRAIN outputs/det_cache_train/rtdetr
copy_set train/visual_cues $NTRAIN outputs/det_cache_train/visual_cues
copy_set val/fasterrcnn 7 outputs/det_cache/fasterrcnn
copy_set testdev/fasterrcnn 17 outputs/det_cache_testdev/fasterrcnn
copy_set uavdt/fasterrcnn 20 outputs/det_cache_uavdt/fasterrcnn
copy_set val_nms050/fasterrcnn 7 outputs/det_cache_nms050/fasterrcnn
copy_set testdev/visual_cues 17 outputs/det_cache_testdev/visual_cues
copy_set uavdt/visual_cues 20 outputs/det_cache_uavdt/visual_cues
echo "T4 SESSION CACHES SYNCED $(date +%T)"
