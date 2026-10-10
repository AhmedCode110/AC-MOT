"""
Generic crop-refinement capability (any detector that accepts an image):
fixed 2 x 2 tile grid with overlap, tile detections mapped back to frame
coordinates, and a merge that adds a tile box only when it is new.

Constants: overlap 0.2 (the default overlap ratio of SAHI, Akyon et al.
2022); merge IoU 0.5 (the duplicate rule of V7f, acmot_v7 V7Spec.dup_iou).
"""
from __future__ import annotations

import numpy as np

OVERLAP = 0.2
MERGE_IOU = 0.5


def tile_boxes(width, height, overlap=OVERLAP):
    """Four tiles (x0, y0, x1, y1) in the order top-left, top-right,
    bottom-left, bottom-right; each covers 1 / (2 - overlap) of a side."""
    tw, th = int(round(width / (2 - overlap))), int(round(height / (2 - overlap)))
    return [(0, 0, tw, th), (width - tw, 0, width, th), (0, height - th, tw, height),
            (width - tw, height - th, width, height)]


def _iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ub = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(ua[:, None] + ub[None, :] - inter, 1e-12)


def merge(base, tiles, iou=MERGE_IOU):
    """base: N x 6 (x1, y1, x2, y2, score, class); tiles: M x 6. Returns the
    base rows plus every tile row (descending score) that has no same-class
    partner with IoU >= iou among the rows kept so far."""
    base = np.asarray(base, np.float64).reshape(-1, 6)
    tiles = np.asarray(tiles, np.float64).reshape(-1, 6)
    if not len(tiles):
        return base
    kept = [base]
    cur = base
    for r in tiles[np.argsort(-tiles[:, 4])]:
        same = cur[cur[:, 5] == r[5]]
        if len(same) and _iou(r[None, :4], same[:, :4]).max() >= iou:
            continue
        cur = np.vstack([cur, r[None]])
    return cur
