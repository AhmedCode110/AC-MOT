"""
General AC-MOT G2: generic compute actions upstream of the frozen V7f layer.

  frame -> generic causal observations -> compute controller -> action
        -> detector adapter (native setting) -> frozen detector (or reuse)
        -> canonical detections -> frozen V7f -> tracker adapter -> tracks

A compute action is (profile, call):
  profile  an opaque profile id of the detector adapter with a normalized
           cost (the controller never sees a native setting)
  call     False = no detector call this frame: the last canonical
           detections are passed on again (cost 0)
The controller interface is decide(t, observation) -> action and
observe(t, feedback) after the frame; feedback of frame t is used from t+1.
V7f is called exactly as in the frozen record (acmot_v7.V7Layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ComputeAction:
    profile: str
    call: bool = True


class StaticSchedule:
    """Fixed profile with a fixed call period k (k = 1: every frame)."""

    def __init__(self, profile, period=1):
        self.profile, self.period, self.since = profile, int(period), None

    def decide(self, t, obs):
        call = self.since is None or self.since >= self.period - 1
        self.since = 0 if call else self.since + 1
        return ComputeAction(self.profile, call)

    def observe(self, t, feedback):
        pass


class Schedule:
    """Per-frame list of (profile, period): a diagnostic or control schedule."""

    def __init__(self, items):
        self.items, self.since = list(items), None

    def decide(self, t, obs):
        profile, period = self.items[t - 1]
        call = self.since is None or self.since >= int(period) - 1
        self.since = 0 if call else self.since + 1
        return ComputeAction(profile, call)

    def observe(self, t, feedback):
        pass


@dataclass
class G2Pipeline:
    controller: object             # decide(t, obs) -> ComputeAction ; observe(t, feedback)
    detector: object               # detect(profile) -> canonical detections (x1, y1, x2, y2, confidence, class_id)
    cost: dict                     # profile -> normalized cost of one detector call
    layer: object                  # acmot_v7.V7Layer (frozen)
    tracker: object                # tracker adapter
    make_detection: object
    record_trace: bool = False
    trace: list = field(default_factory=list)
    frame: int = 0
    last: list = field(default_factory=list)

    def step(self, obs, shape, motion=None, image=None):
        import numpy as np
        self.frame += 1
        t = self.frame
        act = self.controller.decide(t, obs)
        if self.record_trace:
            self.trace.append(("decide", t, act.profile, act.call))
        called = act.call or t == 1          # the first frame always calls the detector
        if called:
            raw = self.detector(act.profile)
            self.last = raw
            spent = self.cost[act.profile]
        else:
            raw = self.last
            spent = 0.0
        if self.record_trace:
            self.trace.append(("detect", t))
        b = np.array([[d.x1, d.y1, d.x2, d.y2] for d in raw]).reshape(-1, 4)
        s = np.array([d.confidence for d in raw])
        dec = self.layer.step(b, s, motion, classes=[d.class_id for d in raw])
        dets = [self.make_detection(raw[k], float(v)) for k, v in zip(dec.keep, dec.scores)]
        self.tracker.set_association_tolerance(dec.match)
        tracks = self.tracker.update(dets, shape, association_threshold=dec.assoc,
                                     birth_threshold=dec.birth, image=image)
        if self.record_trace:
            self.trace.append(("v7_track", t))
        boxes = [[x.x1, x.y1, x.x2, x.y2] for x in tracks]
        self.layer.observe(boxes, [x.track_id for x in tracks])
        self.controller.observe(t, dict(tracks=tracks, detections=raw, called=called,
                                        profile=act.profile, v7=dec.log))
        if self.record_trace:
            self.trace.append(("observe", t))
        audit = dict(dec.log, profile=act.profile, called=bool(called), cost=spent,
                     tracks=len(tracks))
        return tracks, audit
