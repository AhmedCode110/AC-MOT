"""
UAVDT transfer: detector cache generation for the frozen v3 AC-MOT
controller's 3 exact (resolution, NMS) action pairs only -- no new GPU
work beyond what the frozen policy actually needs.
  LOW=(1536,0.70), MEDIUM=(1280,0.45), HIGH=(1088,0.60)
Same frozen checkpoint, same verbatim multi-NMS-from-one-forward-pass
design. Detector confidence floor is applied post-hoc (exact, as
verified for VisDrone); not baked into the cache.
"""
import glob
import hashlib
import json
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

BASE = "/teamspace/studios/this_studio"
IMG_ROOT = f"{BASE}/data/uavdt_test20"
OUT = f"{BASE}/outputs/uavdt_cache"
FROZEN_CKPT = f"{BASE}/detector/frozen/yolo11m_visdrone_frozen.pt"
FROZEN_SHA256 = "c6f98247e7c731a567aaf37ca7b808beff588b687623e028b25a2e1d846e96e7"

# exactly the 3 action pairs the frozen v3 controller uses (no 3x3 grid -- only what's needed)
ACTION_RES_NMS = [(1536, 0.70), (1280, 0.45), (1088, 0.60)]
CKPT_TO_EVAL5 = {0: 0, 3: 1, 4: 2, 5: 3, 8: 4}  # pedestrian,car,van,truck,bus

SEQS = "M0203 M0205 M0208 M0209 M0403 M0601 M0602 M0606 M0701 M0801 M0802 M1001 M1004 M1007 M1009 M1101 M1301 M1302 M1303 M1401".split()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


assert _sha256(FROZEN_CKPT) == FROZEN_SHA256
det_model = YOLO(FROZEN_CKPT)
HALF = torch.cuda.is_available()
assert HALF, "No CUDA GPU visible."


def multi_nms_predict_single(model, path, imgsz, nms_val):
    r = model.predict(path, imgsz=imgsz, conf=0.01, iou=nms_val, max_det=1000, half=HALF, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.orig_shape


t0 = time.time()
forward_passes = 0
for res, nms in ACTION_RES_NMS:
    d = f"{OUT}/r{res}_n{int(round(100 * nms))}"
    os.makedirs(d, exist_ok=True)
    for s in SEQS:
        out_f = f"{d}/{s}.npz"
        if os.path.exists(out_f):
            print("skip", res, nms, s)
            continue
        frames = sorted(glob.glob(f"{IMG_ROOT}/{s}/*.jpg"))
        rows = []
        shape = None
        for t_, f in enumerate(frames, 1):
            xyxy, conf, cls, shape = multi_nms_predict_single(det_model, f, res, nms)
            forward_passes += 1
            for b, sc, c in zip(xyxy, conf, cls):
                if int(c) not in CKPT_TO_EVAL5:
                    continue
                rows.append([t_, *b.tolist(), float(sc), CKPT_TO_EVAL5[int(c)]])
        np.savez_compressed(out_f, det=np.asarray(rows, np.float64).reshape(-1, 7),
                            shape=np.asarray(shape), frames=len(frames))
        print(res, nms, s, "frames", len(frames), "dets", len(rows), "elapsed", round(time.time() - t0, 1))

print("DONE. forward_passes", forward_passes, "wall_seconds", time.time() - t0)
json.dump(dict(forward_passes=forward_passes, wall_seconds=time.time() - t0, action_res_nms=ACTION_RES_NMS),
          open(f"{OUT}/UAVDT_CACHE_STATS.json", "w"), indent=1)
