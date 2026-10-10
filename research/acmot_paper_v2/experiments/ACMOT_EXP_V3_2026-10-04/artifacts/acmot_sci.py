"""
Scene layer of the SCI + V7f architecture: the hand-designed Scene
Complexity Index (SCI) of the original AC-MOT, reduced to ONE output, a
generic compute level (LOW / MEDIUM / HIGH) for the detector.

Responsibilities (and nothing else):
  image statistics of frame t + object geometry of frames < t
      -> SCI (moving mean of the last `window` analysis steps)
      -> compute level for frame t
The layer never sees a detector score, never sets a confidence, NMS or
tracker threshold, and contains no detector, tracker or dataset name. The
detector-native meaning of a level (an input resolution, for example) is
defined by a detector adapter (adapters/detectors/compute_profile.py).

Source of every constant: the resolution rule of the historical controller
`core.Controller` (policy "adaptive", stable=True, detector_feedback=True;
the configuration of run_universal_acmot.build_config). The rule is copied,
not re-selected; tests/test_sci_v7.py asserts that the level sequence equals
core.Controller's resolution sequence on the same inputs (its three
resolutions, in increasing order, read as LOW / MEDIUM / HIGH).

One deliberate change: the historical crowd / tiny cues counted the
detections kept at frame t-1 above a raw detector score of 0.18, which is a
detector-specific score constant. Here they count the objects the tracker
reported at frame t-1 (any tracker reports boxes), so no score enters the
scene layer.

Causality: decide(t) uses the image statistics of frame t and the object
boxes passed to observe() for frames < t; observe(frame t) affects frames
> t only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

LEVELS = ("LOW", "MEDIUM", "HIGH")
_RANK = {lv: i for i, lv in enumerate(LEVELS)}


@dataclass(frozen=True)
class SceneSpec:
    # analysis stride (frames) and moving-mean window (analysis steps)
    stride: int = 10
    window: int = 7
    # SCI = sum of weighted cues (historical weights and normalisers)
    w_crowd: float = 0.30
    w_tiny: float = 0.30
    w_edge: float = 0.20
    w_dark: float = 0.10
    w_blur: float = 0.05
    crowd_norm: float = 30.0          # objects
    tiny_area: float = 1024.0         # px^2 in source-image pixels (32 x 32)
    edge_norm: float = 0.14           # Canny edge fraction
    dark_level: float = 80.0          # mean gray level
    blur_level: float = 180.0         # Laplacian variance
    # scene labels
    tiny_scene: float = 0.50
    crowd_scene: float = 0.65
    edge_scene: float = 0.13
    # level rule: HIGH if sci > t_high or tiny > tiny_scene,
    # MEDIUM if sci > t_med or scene in (crowded, tiny), else LOW
    t_med: float = 0.35
    t_high: float = 0.60
    # hysteresis ("stable"): a lower level is taken only when the index has
    # left the hold region, and never within `dwell` frames of a change
    hold_high_sci: float = 0.50
    hold_high_tiny: float = 0.40
    hold_med_sci: float = 0.25
    dwell: int = 30
    # recovery probe: HIGH for up to `probe_steps` analysis steps after the
    # object count collapses below drop_ratio of its decaying peak
    probe: bool = True
    peak_min: int = 5
    drop_ratio: float = 0.40
    peak_decay: float = 0.80
    probe_steps: int = 3
    start_level: str = "LOW"


@dataclass
class SceneDecision:
    frame: int
    level: str
    sci: float
    scene: str
    analysed: bool
    cues: dict


class SceneLayer:
    """Generic scene-dependent compute controller (no scores, no names)."""

    def __init__(self, spec: SceneSpec = SceneSpec()):
        self.spec = spec
        self.reset()

    def reset(self):
        s = self.spec
        self.history = deque(maxlen=s.window)
        self.level = s.start_level
        self.last_change = 1
        self.sci = 0.0
        self.tiny = 0.0
        self.scene = "clear"
        self.peak_count = 0
        self.drop_age = 0
        self.prev_boxes = np.zeros((0, 4))
        self.cues = {}

    def observe(self, boxes):
        """Object boxes (x1, y1, x2, y2) reported for frame t; used from t+1."""
        self.prev_boxes = np.asarray(boxes, np.float64).reshape(-1, 4)

    def decide(self, frame: int, image_stats: dict) -> SceneDecision:
        """Compute level for frame t (1-based). image_stats: edges,
        brightness, blur of frame t (analyze_visual)."""
        s = self.spec
        analysed = frame == 1 or frame % s.stride == 1
        if analysed:
            b = self.prev_boxes
            n = len(b)
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            self.tiny = float(np.mean(area < s.tiny_area)) if n else 0.0
            crowd = min(n / s.crowd_norm, 1.0)
            e, br, bl = image_stats["edges"], image_stats["brightness"], image_stats["blur"]
            raw = (s.w_crowd * crowd + s.w_tiny * self.tiny + s.w_edge * min(e / s.edge_norm, 1.0)
                   + s.w_dark * (br < s.dark_level) + s.w_blur * (bl < s.blur_level))
            self.history.append(raw)
            self.sci = float(np.mean(self.history))
            self.scene = ("night" if br < s.dark_level else "blur" if bl < s.blur_level else
                          "tiny" if self.tiny > s.tiny_scene else
                          "crowded" if crowd > s.crowd_scene or e > s.edge_scene else "clear")
            dropped = self.peak_count >= s.peak_min and n < s.drop_ratio * self.peak_count
            self.drop_age = self.drop_age + 1 if dropped else 0
            self.peak_count = max(n, int(self.peak_count * s.peak_decay))
            target = ("HIGH" if self.sci > s.t_high or self.tiny > s.tiny_scene else
                      "MEDIUM" if self.sci > s.t_med or self.scene in ("crowded", "tiny") else "LOW")
            if s.probe and 0 < self.drop_age <= s.probe_steps:
                target = "HIGH"
            if _RANK[target] < _RANK[self.level]:
                if self.level == "HIGH" and (self.sci > s.hold_high_sci or self.tiny > s.hold_high_tiny):
                    target = "HIGH"
                elif self.level == "MEDIUM" and self.sci > s.hold_med_sci:
                    target = "MEDIUM"
            if frame - self.last_change < s.dwell:
                target = self.level
            if target != self.level:
                self.level, self.last_change = target, frame
            self.cues = dict(n=n, crowd=crowd, tiny=self.tiny, edges=float(e),
                             brightness=float(br), blur=float(bl), raw=float(raw))
        return SceneDecision(frame, self.level, self.sci, self.scene, analysed, dict(self.cues))


def describe(spec: SceneSpec = SceneSpec()):
    return asdict(spec)
