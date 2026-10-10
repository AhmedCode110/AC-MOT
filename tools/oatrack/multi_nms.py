"""One detector forward pass, several NMS IoU operating points, each exactly
equal to Ultralytics `model.predict(..., iou=tau)` (same preprocessing and
the predictor's own postprocess; the raw output is cloned per call because
NMS modifies its input in place)."""
from __future__ import annotations

import cv2
import torch


def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(v) for v in x)
    return x


def multi_nms_predict(model, path, imgsz, nms_list, conf=0.01, max_det=1000, half=False):
    """-> {tau: (N, 6) array [x1, y1, x2, y2, score, cls]}, original (h, w)."""
    model.predict(path, imgsz=imgsz, conf=conf, iou=nms_list[0], max_det=max_det, half=half, verbose=False)
    p = model.predictor
    im0 = cv2.imread(path)
    with torch.inference_mode():
        im = p.preprocess([im0])
        raw = p.inference(im)
        out = {}
        for tau in nms_list:
            p.args.iou = tau
            r = p.postprocess(_clone(raw), im, [im0])[0]
            b = r.boxes
            out[tau] = torch.cat([b.xyxy, b.conf[:, None], b.cls[:, None]], 1).cpu().numpy()
    return out, im0.shape[:2]
