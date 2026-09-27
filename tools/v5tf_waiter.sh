#!/bin/zsh
# Waits for all V5-TF development caches (736 + 640/832), merges levels, then
# runs E36 (families) and E39 (constant audit). Metrics on development-40 only.
cd /Users/ahmedgouda/Desktop/Universal-ACMOT
n() { ls $1/*.npz 2>/dev/null | wc -l; }
until [ $(n outputs/det_cache_train_native/yolov8) -ge 56 ] && [ $(n outputs/det_cache_train/rtdetr) -ge 56 ] \
   && [ $(n outputs/det_cache_train/visual_cues) -ge 56 ] && [ $(n outputs/det_cache_train_res_extra/yolov8) -ge 56 ] \
   && [ $(n outputs/det_cache_train_res_extra/rtdetr) -ge 56 ]; do sleep 120; done
sleep 60
export PYTHONPATH=$(pwd) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=.venv/bin/python
$PY tools/merge_cache_levels.py outputs/det_cache_train_native/yolov8 outputs/det_cache_train_res_extra/yolov8 outputs/det_cache_train_res/yolov8
$PY tools/merge_cache_levels.py outputs/det_cache_train/rtdetr outputs/det_cache_train_res_extra/rtdetr outputs/det_cache_train_res/rtdetr
[ -e outputs/det_cache_train_res/visual_cues ] || ln -s ../det_cache_train/visual_cues outputs/det_cache_train_res/visual_cues
echo "V5-TF DEV VALIDATION START $(date -u +%FT%TZ) commit $(git rev-parse HEAD)"
git status --short
$PY tools/v5tf_dev.py run 2>&1 | grep --line-buffered -v -i "warn\|BURST"
$PY tools/v5tf_dev.py sens 2>&1 | grep --line-buffered -v -i "warn\|BURST"
echo "V5-TF DEV VALIDATION DONE $(date -u +%FT%TZ)"
