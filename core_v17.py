"""AC-MOT v17 presentation-aligned controller.

This module intentionally does not modify core.py or any archived version.
It restores the v10 SmartCalibrator semantics described in the seminar:
SCI -> adaptive detector confidence, NMS IoU and inference resolution.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np

from core import boxes


@dataclass(frozen=True)
class PresentationSpec:
    name: str
    adaptive_threshold: bool = True
    adaptive_resolution: bool = True
    smoothing_window: int = 7
    analysis_stride: int = 10
    tracker_profile: str = "tuned"

    def validate(self):
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >=1")
        if self.analysis_stride < 1:
            raise ValueError("analysis_stride must be >=1")
        if self.tracker_profile not in {"default", "tuned"}:
            raise ValueError("tracker_profile must be default or tuned")
        return self


class PresentationController:
    """Exact v10 SCI + SmartCalibrator mapping, with explicit ablation switches."""
    def __init__(self, spec: PresentationSpec):
        self.spec = spec.validate()
        self.history = deque(maxlen=spec.smoothing_window)
        self.sci = 0.0
        self.tiny = 0.0
        self.scene = "clear"
        self.last_params = dict(conf=0.25, nms=0.45, size=640, sci=0.0, scene="clear")

    def choose(self, frame, visual, previous):
        s = self.spec
        analyze = frame == 1 or (frame - 1) % s.analysis_stride == 0
        if analyze:
            previous = boxes(previous)
            # Match the v10 analyzer: all previous boxes are cues; detector confidence
            # filtering happens before this controller in the runtime pipeline.
            n = len(previous)
            self.tiny = float(np.mean(
                (previous[:, 2] - previous[:, 0]) * (previous[:, 3] - previous[:, 1]) < 32 * 32
            )) if n else 0.0
            crowd = min(n / 30.0, 1.0)
            edge = float(visual["edges"])
            brightness = float(visual["brightness"])
            blur = float(visual["blur"])

            raw = 0.30 * crowd + 0.20 * min(edge / 0.14, 1.0) + 0.30 * self.tiny
            if brightness < 80:
                raw += 0.10
            if blur < 180:
                raw += 0.05
            self.history.append(float(np.clip(raw, 0.0, 1.0)))
            self.sci = float(np.mean(self.history))

            if brightness < 80:
                self.scene = "night"
            elif blur < 180:
                self.scene = "blur"
            elif self.tiny > 0.50:
                self.scene = "tiny"
            elif crowd > 0.65 or edge > 0.13:
                self.scene = "crowded"
            else:
                self.scene = "clear"

        conf, nms, size = 0.25, 0.45, 640
        if s.adaptive_threshold:
            # Exact v10 SmartCalibrator mapping recovered from AC_MOT_v10.ipynb.
            conf = 0.245 - 0.050 * self.sci
            nms = 0.490 - 0.050 * self.sci
            if self.scene in {"crowded", "tiny", "night"}:
                conf -= 0.012
            if self.scene == "blur":
                nms -= 0.012
            conf = float(np.clip(conf, 0.19, 0.28))
            nms = float(np.clip(nms, 0.40, 0.52))

        if s.adaptive_resolution:
            if self.sci > 0.60 or self.tiny > 0.50:
                size = 832
            elif self.sci > 0.35 or self.scene in {"crowded", "tiny"}:
                size = 736

        self.last_params = dict(
            conf=float(conf), nms=float(nms), size=int(size),
            sci=float(self.sci), scene=self.scene,
        )
        return dict(self.last_params)


def tracker_settings(profile: str):
    if profile == "default":
        return dict(high=0.25, low=0.10, new=0.25, buffer=30, match=0.80, fuse=True)
    if profile == "tuned":
        return dict(high=0.18, low=0.04, new=0.20, buffer=45, match=0.86, fuse=True)
    raise ValueError(profile)
