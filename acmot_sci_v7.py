"""
SCI + V7f: frame -> scene layer (SCI) -> compute level -> detector adapter
-> frozen detector -> candidates -> frozen V7f score layer -> tracker
adapter -> frozen tracker -> tracks -> history for frame t+1.

Separation of responsibilities:
  scene layer (acmot_sci.SceneLayer)   compute level only
  detector adapter (ComputeProfileAdapter)  level -> detector-native setting
  V7f (acmot_v7.V7Layer, frozen, unchanged)  score interpretation, bands,
      candidate admission, host thresholds, regime, rescue, host protection
  tracker adapter                      host thresholds / tolerance -> tracker
Neither layer reads or writes the other's state: the scene layer never sees
a score or a threshold, and V7f never sees the level.

Level sources (for ablations): a SceneLayer, one fixed level, or an
explicit per-frame schedule (budget-matched controls).

Causality, per frame t (recorded in `trace` when enabled):
  scene.decide(t)        image statistics of t, object boxes of frames < t
  detect(t)              at the chosen setting
  v7.step(t)             V7f state of frames < t (frozen implementation)
  tracker.update(t)
  v7.observe(t), scene.observe(t)   used from frame t+1 on
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from acmot_sci import LEVELS, SceneLayer


@dataclass
class LevelSource:
    """Exactly one of: scene (SceneLayer), fixed (level), schedule (list)."""
    scene: SceneLayer | None = None
    fixed: str | None = None
    schedule: list | None = None

    def __post_init__(self):
        if sum(x is not None for x in (self.scene, self.fixed, self.schedule)) != 1:
            raise ValueError("give exactly one of scene / fixed / schedule")
        if self.fixed is not None and self.fixed not in LEVELS:
            raise ValueError(self.fixed)
        if self.schedule is not None:
            self.schedule = list(self.schedule)

    def decide(self, frame, image_stats):
        if self.scene is not None:
            d = self.scene.decide(frame, image_stats)
            return d.level, dict(sci=d.sci, scene=d.scene, analysed=d.analysed)
        if self.fixed is not None:
            return self.fixed, {}
        return self.schedule[frame - 1], {}

    def observe(self, boxes):
        if self.scene is not None:
            self.scene.observe(boxes)


@dataclass
class SciV7Pipeline:
    levels: LevelSource
    adapter: object                 # ComputeProfileAdapter
    layer: object                   # acmot_v7.V7Layer (frozen)
    tracker: object                 # tracker adapter: update(...), set_association_tolerance(...)
    make_detection: object          # (raw detection, new score) -> detection passed to the tracker
    record_trace: bool = False
    trace: list = field(default_factory=list)
    frame: int = 0

    def _t(self, *ev):
        if self.record_trace:
            self.trace.append(ev)

    def step(self, image_stats, detect, shape, motion=None, image=None):
        """One frame. detect(setting) -> raw detections (x1, y1, x2, y2,
        confidence, class_id attributes) of this frame at that setting."""
        self.frame += 1
        t = self.frame
        level, scene_log = self.levels.decide(t, image_stats)
        setting = self.adapter.resolution(level)
        self._t("scene", t, level)
        raw = detect(setting)
        self._t("detect", t, setting)
        b = np.array([[d.x1, d.y1, d.x2, d.y2] for d in raw]).reshape(-1, 4)
        s = np.array([d.confidence for d in raw])
        dec = self.layer.step(b, s, motion, classes=[d.class_id for d in raw])
        self._t("v7", t)
        dets = [self.make_detection(raw[k], float(v)) for k, v in zip(dec.keep, dec.scores)]
        self.tracker.set_association_tolerance(dec.match)
        tracks = self.tracker.update(dets, shape, association_threshold=dec.assoc,
                                     birth_threshold=dec.birth, image=image)
        self._t("track", t)
        boxes = [[x.x1, x.y1, x.x2, x.y2] for x in tracks]
        self.layer.observe(boxes, [x.track_id for x in tracks])
        self.levels.observe(boxes)
        self._t("observe", t)
        audit = dict(dec.log, level=level, setting=setting, tracks=len(tracks), **scene_log)
        return tracks, audit
