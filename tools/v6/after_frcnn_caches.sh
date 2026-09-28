#!/bin/zsh
# Waits for the Faster R-CNN NMS-0.45 caches, then runs the locked Faster
# R-CNN transfer evaluations once (V4 + static baselines; V6-TF already ran
# on the native caches) and the reports. Evaluation only.
cd /Users/ahmedgouda/Desktop/Universal-ACMOT
export PYTHONPATH=$(pwd)
PY=.venv/bin/python
until grep -q QUEUE_DONE outputs/v6/cache_frcnn045.log; do sleep 60; done
for sp in testdev uavdt; do
  V6_DETS=fasterrcnn V6_SPLIT=$sp $PY tools/v6/dev.py run V4 shared_static static_default
  V6_DETS=fasterrcnn V6_SPLIT=$sp $PY tools/v6/dev.py report V4 shared_static static_default V6TF
done
V6_DETS=fasterrcnn V6_SPLIT=testdev $PY tools/v6/dev.py official V4 shared_static static_default V6TF
V6_DETS=fasterrcnn V6_SPLIT=testdev V6_TAG=_frcnn $PY tools/v6/confirm_report.py V6TF V4 shared_static static_default
V6_DETS=fasterrcnn V6_SPLIT=uavdt V6_PROTOCOLS=internal V6_TAG=_frcnn $PY tools/v6/confirm_report.py V6TF V4 shared_static static_default
echo AFTER_DONE
