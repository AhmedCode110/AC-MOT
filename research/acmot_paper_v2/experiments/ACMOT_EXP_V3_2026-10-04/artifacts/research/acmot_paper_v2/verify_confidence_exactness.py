"""
Verifies (or falsifies) that post-cache confidence-floor filtering is
mathematically equivalent to running the detector with that confidence
floor from the start -- precision-matched to how the cache was actually
generated (half=torch.cuda.is_available(), i.e. fp16 on the T4).

Logic: cached rows already survived NMS run at a permissive conf=0.01
floor. Standard greedy NMS always keeps the highest-scoring box in a
cluster and suppresses lower-scoring overlapping boxes, so raising the
confidence floor afterward (dropping rows below it) can only remove
already-kept boxes -- it can never resurrect a suppressed one, and the
survivor set for boxes >= the new floor is the same either way. This
script checks that this holds in practice (it is a real postprocess
implementation, not an abstract guarantee), using IoU-tolerant matching
(not exact float equality, since fp16 introduces small coordinate noise)
across several frames/sequences/floors.

Run on CUDA (the Lightning T4), matching the cache's own half=True setting
exactly -- a CPU/fp32 comparison is NOT a valid test of this (fp16 shifts
scores enough to flip membership for boxes near the threshold).
"""
from __future__ import annotations

import json

import numpy as np
import torch
from ultralytics import YOLO

BASE = "/teamspace/studios/this_studio"
CKPT = f"{BASE}/detector/frozen/yolo11m_visdrone_frozen.pt"
CACHE_ROOT = f"{BASE}/outputs/acmot_oatrack/cache/yolo11m_vd/visdrone_calib/r1536_n70"
DATA_ROOT = f"{BASE}/data/VisDrone2019-MOT-train/sequences"

CASES = [
    ("uav0000316_01288_v", 50), ("uav0000316_01288_v", 100),
    ("uav0000020_00406_v", 50), ("uav0000020_00406_v", 300),
    ("uav0000243_00001_v", 200), ("uav0000243_00001_v", 700),
]
FLOORS = [0.10, 0.25, 0.40]
IOU_MATCH = 0.90  # two boxes count as "the same detection" above this IoU

# This is exactly the class filter and remapping used to construct the cache
# in notebooks/lightning/make_cache_script.py.  The checkpoint predicts all
# ten VisDrone classes; the cache stores only this eval5 subset.
CKPT_TO_EVAL5 = {0: 0, 3: 1, 4: 2, 5: 3, 8: 4}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def match(cache_boxes, fresh_boxes):
    unmatched_cache, unmatched_fresh = list(range(len(cache_boxes))), set(range(len(fresh_boxes)))
    only_cache, only_fresh = [], []
    for i in list(unmatched_cache):
        best_j, best_iou = None, 0
        for j in unmatched_fresh:
            v = iou(cache_boxes[i], fresh_boxes[j])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= IOU_MATCH:
            unmatched_fresh.discard(best_j)
        else:
            only_cache.append(i)
    only_fresh = list(unmatched_fresh)
    return only_cache, only_fresh


HALF = torch.cuda.is_available()
assert HALF, "this check must run on CUDA with half=True to match the cache's own generation settings"
model = YOLO(CKPT)

results = []
for seq, frame in CASES:
    npz = np.load(f"{CACHE_ROOT}/{seq}.npz")
    det = npz["det"]
    img_path = f"{DATA_ROOT}/{seq}/{frame:07d}.jpg"
    for floor in FLOORS:
        cache_rows = det[(det[:, 0] == frame) & (det[:, 5] >= floor)]
        cache_boxes = cache_rows[:, 1:5].tolist()
        r = model.predict(img_path, imgsz=1536, conf=floor, iou=0.70, max_det=1000, half=HALF, verbose=False)[0]
        fresh_raw_boxes = r.boxes.xyxy.cpu().numpy()
        fresh_raw_classes = r.boxes.cls.cpu().numpy().astype(int)
        keep = np.isin(fresh_raw_classes, list(CKPT_TO_EVAL5))
        fresh_boxes = fresh_raw_boxes[keep].tolist()
        fresh_eval5_classes = [CKPT_TO_EVAL5[int(c)] for c in fresh_raw_classes[keep]]
        only_cache, only_fresh = match(cache_boxes, fresh_boxes)
        results.append(dict(seq=seq, frame=frame, floor=floor, cache_n=len(cache_boxes), fresh_n=len(fresh_boxes),
                             fresh_eval5_class_ids=fresh_eval5_classes,
                             only_in_cache=len(only_cache), only_in_fresh=len(only_fresh),
                             exact_match=len(only_cache) == 0 and len(only_fresh) == 0))
        print(results[-1])

n_cases = len(results)
n_exact = sum(r["exact_match"] for r in results)
summary = dict(iou_match_threshold=IOU_MATCH, total_cases=n_cases, exact_matches=n_exact,
               all_exact=n_exact == n_cases, class_filter="CKPT_TO_EVAL5", results=results)
json.dump(summary, open(f"{BASE}/outputs/acmot_oatrack/CONFIDENCE_EXACTNESS_CHECK_FIXED.json", "w"), indent=1)
print(json.dumps(dict(total_cases=n_cases, exact_matches=n_exact, all_exact=n_exact == n_cases), indent=1))
