"""
OATrack (Qi, Wang, Jiang, Sensors 26(16):5222, 2026), re-implemented from
the paper; the authors released no code.

Main configuration of the paper:
  q_det = s (alpha=1, beta=0, b=0, global scale prior disabled)
  q_motion(d,t) = lam*exp(-0.5*dc^2) + (1-lam)*IoU(b_d, b_t),  lam = 0.5,
      dc = centre distance / sqrt(area of the track box)
  Q(d,t) = q_det^gamma * q_motion^(1-gamma),  gamma = 0.6
  r_scale(d,t) = 0.5 + 0.5*exp(-|log A_d - m_A|)*exp(-|log r_d - m_r|)
  Q~ = Q * r_scale  (track prototype: log area m_A, log aspect m_r)
  candidates: s >= min_conf (0.40) and q_det >= tau_buffer (0.20), top 80 by q_det
  stage 1: active tracks x candidates, Hungarian on -Q~, accept Q~ >= 0.70
  stage 2: unmatched active tracks x unmatched candidates, accept >= 0.35
  stage 3: recently lost tracks x remaining candidates, accept >= 0.25
  birth: unmatched candidate with q_det >= tau_high (0.65)
  prototype update with momentum 0.9 when the match quality >= max(0.2, 0.5*tau_update) = 0.3
  unmatched active tracks become lost; lost tracks are removed after 30 frames

Choices the paper does not state (documented, kept fixed):
  * no motion prediction: the track box is the latest observed box
    (the paper matches against "the latest observed box");
  * the prototype of a new track is its first box;
  * output per frame: tracks matched or born in this frame (MOTChallenge
    convention), with the detection box and score.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from adapters.types import DetectionList, Track, TrackList

PARAMS = dict(min_conf=0.40, tau_buffer=0.20, max_dets=80, lam=0.50, gamma=0.60,
              stages=(0.70, 0.35, 0.25), tau_high=0.65, tau_update=0.60, momentum=0.90, max_lost=30)


def _iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _log_area_aspect(box):
    w, h = max(box[2] - box[0], 1e-6), max(box[3] - box[1], 1e-6)
    return math.log(w * h), math.log(w / h)


class _T:
    __slots__ = ("id", "box", "score", "cls", "mA", "mr", "lost")

    def __init__(self, tid, box, score, cls):
        self.id, self.box, self.score, self.cls, self.lost = tid, np.asarray(box, float), score, cls, 0
        self.mA, self.mr = _log_area_aspect(self.box)


class OATrack:
    def __init__(self, **overrides):
        self.p = dict(PARAMS, **overrides)
        self.active, self.lost = [], []
        self.next_id = 1

    def quality(self, tracks, boxes, scores):
        """Q~ matrix (tracks x detections)."""
        p = self.p
        tb = np.array([t.box for t in tracks], float)
        iou = _iou(tb, boxes)
        tc = (tb[:, :2] + tb[:, 2:]) / 2
        dcen = (boxes[:, :2] + boxes[:, 2:]) / 2
        tarea = np.maximum((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1]), 1e-6)
        dc = np.linalg.norm(tc[:, None, :] - dcen[None, :, :], axis=2) / np.sqrt(tarea)[:, None]
        q_motion = p["lam"] * np.exp(-0.5 * dc ** 2) + (1 - p["lam"]) * iou
        q_det = scores[None, :]
        Q = q_det ** p["gamma"] * q_motion ** (1 - p["gamma"])
        w = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6)
        h = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)
        lA, lr = np.log(w * h), np.log(w / h)
        mA = np.array([t.mA for t in tracks])[:, None]
        mr = np.array([t.mr for t in tracks])[:, None]
        r_scale = 0.5 + 0.5 * np.exp(-np.abs(lA[None, :] - mA)) * np.exp(-np.abs(lr[None, :] - mr))
        return Q * r_scale

    def _stage(self, tracks, cand, boxes, scores, thr):
        if not tracks or not cand:
            return [], list(range(len(tracks))), cand
        Q = self.quality(tracks, boxes[cand], scores[cand])
        r, c = linear_sum_assignment(-Q)
        pairs, mt, md = [], set(), set()
        for i, j in zip(r, c):
            if Q[i, j] >= thr:
                pairs.append((i, cand[j], Q[i, j]))
                mt.add(i)
                md.add(cand[j])
        return pairs, [i for i in range(len(tracks)) if i not in mt], [d for d in cand if d not in md]

    def _apply(self, t, box, score, cls, q):
        t.box, t.score, t.cls, t.lost = np.asarray(box, float), score, cls, 0
        if q >= max(0.2, 0.5 * self.p["tau_update"]):
            a, r = _log_area_aspect(t.box)
            m = self.p["momentum"]
            t.mA, t.mr = m * t.mA + (1 - m) * a, m * t.mr + (1 - m) * r

    def update_arrays(self, boxes, scores, classes):
        p = self.p
        boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
        keep = np.where((scores >= p["min_conf"]) & (scores >= p["tau_buffer"]))[0]
        keep = keep[np.argsort(-scores[keep], kind="stable")][:p["max_dets"]]
        cand = [int(k) for k in keep]
        out = []
        th1, th2, th3 = p["stages"]
        pairs1, un_a, cand = self._stage(self.active, cand, boxes, scores, th1)
        rest = [self.active[i] for i in un_a]
        pairs2, un_a2, cand = self._stage(rest, cand, boxes, scores, th2)
        pairs3, _, cand = self._stage(self.lost, cand, boxes, scores, th3)
        matched = []
        for i, d, q in pairs1:
            t = self.active[i]
            self._apply(t, boxes[d], scores[d], classes[d], q)
            matched.append(t)
        for i, d, q in pairs2:
            t = rest[i]
            self._apply(t, boxes[d], scores[d], classes[d], q)
            matched.append(t)
        revived = set()
        for i, d, q in pairs3:
            t = self.lost[i]
            self._apply(t, boxes[d], scores[d], classes[d], q)
            matched.append(t)
            revived.add(id(t))
        newly_lost = [rest[i] for i in un_a2]
        for t in newly_lost:
            t.lost = 1
        still_lost = []
        for t in self.lost:
            if id(t) in revived:
                continue
            t.lost += 1
            if t.lost <= p["max_lost"]:
                still_lost.append(t)
        born = []
        for d in cand:
            if scores[d] >= p["tau_high"]:
                t = _T(self.next_id, boxes[d], scores[d], classes[d])
                self.next_id += 1
                born.append(t)
        self.active = matched + born
        self.lost = still_lost + newly_lost
        for t in self.active:
            out.append((t.box.copy(), t.id, t.score, t.cls))
        return out


class OATrackAdapter:
    """Host interface of this repository (same as the other tracker adapters)."""

    def __init__(self, **overrides):
        self.t = OATrack(**overrides)

    def set_association_tolerance(self, m):
        pass

    def update(self, dets: DetectionList, shape=None, association_threshold=None, birth_threshold=None,
               image=None) -> TrackList:
        if dets:
            boxes = [[d.x1, d.y1, d.x2, d.y2] for d in dets]
            scores = [d.confidence for d in dets]
            classes = [d.class_id for d in dets]
        else:
            boxes, scores, classes = np.zeros((0, 4)), [], []
        return [Track(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), track_id=int(i),
                      confidence=float(s), class_id=int(c))
                for b, i, s, c in self.t.update_arrays(boxes, scores, classes)]
