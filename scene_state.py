"""
Generic causal Scene/Tracking-State analyzer (V5 development).

State used to decide frame t is built only from frame t's image statistics
and from detector / tracker outputs of frames < t. No detector or tracker
identity; detector cues use calibration-invariant quantities (ECDF rank u
and the Platt-invariant z-logit gap to the recent leader), geometric cues
are normalised by image size.

Cue families
  A image    : img_edges, img_brightness, img_blur, img_motion, img_motion_resp
  B detector : det_count, det_activity, det_log_area, det_crowd,
               det_overlap, det_gap
  C tracker  : trk_count, trk_survival, trk_birth, trk_persist, trk_match
  (legacy)   : legacy_sci (baseline only)
Each cue is exponentially smoothed with the same 10-frame memory used by
the other V3/V4 stream statistics (decay 0.9; sensitivity analysed in E28).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

FAMILIES = {
    "image": ["img_edges", "img_brightness", "img_blur", "img_motion",
              "img_motion_resp"],
    "detector": ["det_count", "det_activity", "det_log_area", "det_crowd",
                 "det_overlap", "det_gap"],
    "tracker": ["trk_count", "trk_survival", "trk_birth", "trk_persist",
                "trk_match"],
}
ALL_CUES = [c for f in FAMILIES.values() for c in f]
DECAY = 0.9
Z_REF = 0.75       # z-gap defining "leader-like" candidates (= V4 tau)


def _logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(x / (1 - x))


def image_stats(image, prev_small=None):
    """Frame-level image statistics (cheap: 1/4-resolution grayscale)."""
    small = cv2.cvtColor(cv2.resize(image, None, fx=0.25, fy=0.25,
                                    interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
    out = dict(edges=float(cv2.Canny(small, 50, 120).mean() / 255.0),
               brightness=float(small.mean()),
               blur=float(cv2.Laplacian(small, cv2.CV_64F).var()),
               motion=0.0, motion_resp=1.0)
    if prev_small is not None and prev_small.shape == small.shape:
        (dx, dy), resp = cv2.phaseCorrelate(np.float32(prev_small),
                                            np.float32(small))
        diag = float(np.hypot(*small.shape))
        out["motion"] = float(np.hypot(dx, dy) / diag)
        out["motion_resp"] = float(resp)
    return out, small


class SceneStateAnalyzer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ema = {}
        self.ids = deque(maxlen=3)
        self.image_shape = None

    def _smooth(self, key, value):
        if value is None or not np.isfinite(value):
            return
        old = self.ema.get(key)
        self.ema[key] = value if old is None else \
            DECAY * old + (1 - DECAY) * value

    # --- frame t, before the detector ------------------------------------
    def observe_image(self, visual, image_shape):
        self.image_shape = image_shape
        if not visual:
            return
        self._smooth("img_edges", visual.get("edges"))
        self._smooth("img_brightness", visual.get("brightness"))
        blur = visual.get("blur")
        self._smooth("img_blur", np.log1p(blur) if blur is not None else None)
        self._smooth("img_motion", visual.get("motion"))
        self._smooth("img_motion_resp", visual.get("motion_resp"))

    def state(self):
        return {c: self.ema.get(c, np.nan) for c in ALL_CUES}

    # --- after frame t (used from t+1 on) ---------------------------------
    def observe_detector(self, boxes, raw_scores, rank_u, leader_logit,
                         logit_iqr):
        """boxes: (n,4) xyxy; raw_scores: raw detector scores; rank_u: ECDF
        ranks of the scores in the detector's recent stream."""
        if self.image_shape is None or not len(boxes):
            self._smooth("det_count", 0.0)
            return
        h, w = self.image_shape
        z = (_logit(raw_scores) - leader_logit) / max(logit_iqr, 1e-6) \
            if leader_logit is not None else np.zeros(len(raw_scores))
        lead = z >= -Z_REF
        self._smooth("det_count", float(np.log1p(lead.sum())))
        self._smooth("det_activity", float((np.asarray(rank_u) >= 0.9).mean()
                                           / 0.1))
        order = np.argsort(-np.asarray(raw_scores))[:10]
        self._smooth("det_gap", float(np.median(z[order])))
        b = np.asarray(boxes, dtype=np.float64)[lead]
        if len(b) == 0:
            return
        area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1.0)
        self._smooth("det_log_area", float(np.median(np.log(area / (h * w)))))
        if len(b) >= 2:
            c = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
            d = np.hypot(c[:, None, 0] - c[None, :, 0],
                         c[:, None, 1] - c[None, :, 1])
            np.fill_diagonal(d, np.inf)
            self._smooth("det_crowd", float(np.median(
                np.log(d.min(1) / np.sqrt(area)))))
            ix = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) -
                         np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
            iy = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) -
                         np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
            inter = ix * iy
            np.fill_diagonal(inter, 0)
            self._smooth("det_overlap", float((inter.max(1) > 0).mean()))

    def observe_tracks(self, track_ids, n_association_candidates):
        ids = set(int(i) for i in track_ids)
        self.ids.appendleft(ids)
        self._smooth("trk_count", float(np.log1p(len(ids))))
        self._smooth("trk_match", float(len(ids) /
                                        max(n_association_candidates, 1)))
        if len(self.ids) >= 2:
            s1, s2 = self.ids[0], self.ids[1]
            self._smooth("trk_survival", len(s1 & s2) / len(s2) if s2 else 1.0)
            self._smooth("trk_birth", len(s1 - s2) / len(s1) if s1 else 0.0)
        if len(self.ids) == 3:
            s1, s2, s3 = self.ids
            self._smooth("trk_persist",
                         len(s1 & s2 & s3) / len(s1) if s1 else 1.0)
