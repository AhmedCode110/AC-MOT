"""
Cache extension: calibration-split-only, resolutions {896, 1728}, NMS
{0.45, 0.60, 0.70}. Same frozen checkpoint, same verbatim multi-NMS
one-forward-pass design as run_cache_generation.py (not re-derived here;
imported logic is copy-identical by inspection, see
notebooks/lightning/make_cache_script.py). Per
research/acmot_paper_v2/controller_search/PREDECLARATION_BROADENED.json.
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
TR = f"{BASE}/data/VisDrone2019-MOT-train"
OUT = f"{BASE}/outputs/acmot_oatrack"
FROZEN_CKPT = f"{BASE}/detector/frozen/yolo11m_visdrone_frozen.pt"
FROZEN_SHA256 = "c6f98247e7c731a567aaf37ca7b808beff588b687623e028b25a2e1d846e96e7"

NEW_RES = [896, 1728]
NMS_IOU = [0.45, 0.60, 0.70]
CALIBRATION = ["uav0000316_01288_v", "uav0000289_06922_v", "uav0000013_00000_v", "uav0000076_00720_v",
               "uav0000307_00000_v", "uav0000020_00406_v", "uav0000315_00000_v", "uav0000243_00001_v"]
CKPT_TO_EVAL5 = {0: 0, 3: 1, 4: 2, 5: 3, 8: 4}


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


def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(v) for v in x)
    return x


def multi_nms_predict(model, path, imgsz, nms_list):
    model.predict(path, imgsz=imgsz, conf=0.01, iou=nms_list[0], max_det=1000, half=HALF, verbose=False)
    p = model.predictor
    im0 = cv2.imread(path)
    with torch.inference_mode():
        im = p.preprocess([im0])
        raw = p.inference(im)
        out = {}
        for tau in nms_list:
            p.args.iou = tau
            b = p.postprocess(_clone(raw), im, [im0])[0].boxes
            out[tau] = torch.cat([b.xyxy, b.conf[:, None], b.cls[:, None]], 1).cpu().numpy()
    return out, im0.shape[:2]


t0 = time.time()
forward_passes = 0
for res in NEW_RES:
    dirs = {t: f"{OUT}/cache/yolo11m_vd/visdrone_calib/r{res}_n{int(round(100 * t))}" for t in NMS_IOU}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    for s in CALIBRATION:
        if all(os.path.exists(f"{d}/{s}.npz") for d in dirs.values()):
            print("skip", res, s)
            continue
        frames = sorted(glob.glob(f"{TR}/sequences/{s}/*.jpg"))
        rows = {t: [] for t in NMS_IOU}
        for t_, f in enumerate(frames, 1):
            out, shape = multi_nms_predict(det_model, f, res, NMS_IOU)
            forward_passes += 1
            for tau, a in out.items():
                for r in a:
                    ck_cls = int(r[5])
                    if ck_cls not in CKPT_TO_EVAL5:
                        continue
                    rows[tau].append([t_, *r[:4].tolist(), float(r[4]), CKPT_TO_EVAL5[ck_cls]])
        for tau, d in dirs.items():
            np.savez_compressed(f"{d}/{s}.npz", det=np.asarray(rows[tau], np.float64).reshape(-1, 7),
                                shape=np.asarray(shape), frames=len(frames))
        print(res, s, {t: len(v) for t, v in rows.items()}, "frames", len(frames), "elapsed", time.time() - t0)

print("DONE. forward_passes", forward_passes, "wall_seconds", time.time() - t0)
json.dump(dict(forward_passes=forward_passes, wall_seconds=time.time() - t0, new_res=NEW_RES, nms=NMS_IOU),
          open(f"{OUT}/CACHE_EXTENSION_STATS.json", "w"), indent=1)
