"""
Universal AC-MOT with explicit, named candidate-control policies.

All policies share ONE code path and contain no detector names or
detector-specific constants. They differ only in which universal controls
are active, so V1 / V2b / V2c-* / V2d are reproducible ablations of the
same pipeline.

Causality: every decision for frame t uses only state updated after
frame t-1 (normalizer histogram, raw-count EMA, SCI history, track-ID
history). Frame t observations are folded in only after its tracks exist.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace

import numpy as np

from core import Controller
from adapters.detectors.candidate_density import CandidateDensityController
from adapters.detectors.generic_controls import resolve_generic_score_controls
from adapters.detectors.online_normalizer import OnlineScoreNormalizer
from adapters.types import Detection


def replace_det(d, confidence):
    return Detection(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                     confidence=float(confidence), class_id=d.class_id)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    # Density controller + Top-K candidate budget.
    density: bool = False
    topk: bool = False
    # Per-threshold source: "base" (AC generic controls) or
    # "density" (= max(base, density-adjusted); never more permissive).
    low: str = "base"
    high: str = "base"
    new: str = "base"
    # Feedback to SCI: "normalized" (V1: all normalized dets >= high)
    # or "accepted" (only dets actually passed to the tracker, >= high).
    feedback: str = "accepted"
    # V2d: scale the SCI-driven part of the budget by the temporal
    # reliability of the previous tracker output.
    reliability_budget: bool = False
    # V2d-B: density-adjusted high/new are blended in by (1 - reliability).
    # With new="prop": the birth tail is additionally scaled by reliability.
    reliability_birth: bool = False
    # With high="prop" and reliability_birth: scale the high tail too.
    reliability_high: bool = False
    reliability_ema: float = 0.0   # 0 = use the previous frame value only
    # V2f: closed-loop trust u in [trust_min, 1] scaling the whole budget.
    # u <- clip(u * reliability / trust_setpoint). 0 disables.
    trust_setpoint: float = 0.0
    trust_min: float = 0.1
    # V2g: leader-relative gate. A candidate whose RAW score is below
    # leader_rho * (EMA of previous frames' top raw score) is "demoted"
    # (kept only for ByteTrack's low-score association: cannot start a
    # track or join the first association) or "dropped". 0 disables.
    # Percentile normalization keeps only rank; this restores the
    # within-detector score drop-off relative to its own recent leaders.
    leader_rho: float = 0.0
    leader_mode: str = "demote"
    leader_decay: float = 0.9


POLICIES = {
    "V1": PolicySpec("V1", feedback="normalized"),
    "V2b": PolicySpec("V2b", density=True, topk=True, low="density",
                      high="density", new="density"),
    "V2cA": PolicySpec("V2cA", density=True, topk=True),
    "V2cB": PolicySpec("V2cB", density=True, topk=True, high="density",
                       new="density"),
    "V2cC": PolicySpec("V2cC", density=True, topk=True, new="density"),
}


class TemporalReliability:
    """
    Detector-agnostic reliability of the tracker's recent output.

    From the track-ID sets of frames t-1, t-2, t-3 (never frame t):
      survival  = |S1 & S2| / |S2|        (tracks that survived one step)
      persist3  = |S1 & S2 & S3| / |S1|   (current tracks seen 3 frames)
      reliability = persist3 * survival   (continuous, no threshold)
    """

    def __init__(self, ema: float = 0.0):
        self.ids = deque(maxlen=3)
        self.ema = float(ema)
        self.value = 1.0
        self.survival = 1.0
        self.persist3 = 1.0
        self.new_fraction = 0.0

    def reset(self):
        self.__init__(self.ema)

    def observe(self, track_ids):
        self.ids.appendleft(set(int(i) for i in track_ids))
        if len(self.ids) < 3:
            return
        s1, s2, s3 = self.ids
        self.survival = len(s1 & s2) / len(s2) if s2 else 1.0
        self.persist3 = len(s1 & s2 & s3) / len(s1) if s1 else 1.0
        self.new_fraction = len(s1 - s2) / len(s1) if s1 else 0.0
        current = self.persist3 * self.survival
        self.value = (self.ema * self.value + (1 - self.ema) * current
                      if self.ema > 0 else current)


class UniversalPolicyPipeline:
    def __init__(self, config, detector, tracker, policy: PolicySpec,
                 raw_confidence_floor: float = 0.01,
                 density_kwargs: dict | None = None):
        self.config = config.validate()
        self.policy = policy
        self.detector = detector
        self.tracker = tracker
        self.raw_confidence_floor = float(raw_confidence_floor)
        self.density_kwargs = dict(density_kwargs or {})
        self.reset()

    def reset(self):
        self.controller = Controller(self.config)
        self.normalizer = OnlineScoreNormalizer(bins=64, decay=0.95,
                                                update_every=10)
        self.density = CandidateDensityController(**self.density_kwargs)
        self.reliability = TemporalReliability(self.policy.reliability_ema)
        self.trust = 1.0
        self.leader_ref = None
        self.tracker.reset()
        self.previous = []

    def _thresholds(self, base, dens, rel):
        p = self.policy

        # "prop": keep the base AC policy's shape and only shrink its
        # candidate count. Every threshold's upper-tail fraction (1 - thr)
        # is multiplied by the same factor s = effective / base sensitivity
        # (s <= 1, so never more permissive than base). Additive offsets
        # (high = low + 0.18) would instead collapse high/new to the top
        # ~1 candidate once low is near 0.85.
        s = (dens.sensitivity / base.sensitivity) if dens else 1.0
        s_birth = s * (rel if p.reliability_birth else 1.0)

        def pick(mode, b, d, scale=s):
            if mode == "base" or dens is None:
                return b
            if mode == "prop":
                return 1.0 - (1.0 - b) * scale
            return max(b, d)

        low = pick(p.low, base.low_threshold,
                   dens.low_threshold if dens else 0)
        high = pick(p.high, base.high_threshold,
                    dens.high_threshold if dens else 0,
                    s_birth if p.reliability_high else s)
        new = pick(p.new, base.new_track_threshold,
                   dens.new_track_threshold if dens else 0, s_birth)
        if p.reliability_birth and dens is not None and p.new != "prop":
            w = 1.0 - rel
            high = base.high_threshold + w * max(
                0.0, dens.high_threshold - base.high_threshold)
            new = base.new_track_threshold + w * max(
                0.0, dens.new_track_threshold - base.new_track_threshold)
        return low, high, max(high, new)

    def process(self, frame_number, image, visual):
        p = self.policy
        params = self.controller.choose(frame_number, visual, self.previous)
        raw = self.detector.detect(image, confidence=self.raw_confidence_floor,
                                   suppression=params["nms"],
                                   resolution=params["size"])
        normalized = self.normalizer.process(raw)
        base = resolve_generic_score_controls(
            sci=params["sci"], scene=params["scene"],
            recovery=self.config.recovery)

        rel = self.reliability.value
        dens = None
        if p.density:
            budget_sci = params["sci"] * rel if p.reliability_budget \
                else params["sci"]
            dens = self.density.decide(
                base_sensitivity=base.sensitivity, sci=budget_sci,
                budget_scale=self.trust)

        low, high, new = self._thresholds(base, dens, rel)

        demoted = 0
        leader_ref = self.leader_ref
        if p.leader_rho > 0 and leader_ref is not None and raw:
            floor = p.leader_rho * leader_ref
            gated = []
            for r, d in zip(raw, normalized):
                if r.confidence >= floor:
                    gated.append(d)
                elif p.leader_mode == "demote":
                    demoted += 1
                    gated.append(replace_det(d, min(d.confidence,
                                                    high - 1e-6)))
                else:
                    demoted += 1
            normalized = gated
        if raw:
            top = max(r.confidence for r in raw)
            self.leader_ref = top if self.leader_ref is None else (
                p.leader_decay * self.leader_ref
                + (1 - p.leader_decay) * top)

        eligible = sorted((d for d in normalized if d.confidence >= low),
                          key=lambda d: d.confidence, reverse=True)
        target = dens.target_candidates if (dens and p.topk) else None
        detections = eligible[:target] if target is not None else eligible

        tracks = self.tracker.update(
            detections, image.shape[:2],
            association_threshold=high, birth_threshold=new,
            image=image if getattr(self.tracker, "needs_image", False)
            else None)

        # Frame t observations only affect frame t+1 onward.
        if p.density:
            self.density.observe(len(raw))
        self.reliability.observe(t.track_id for t in tracks)
        trust_used = self.trust
        if p.trust_setpoint > 0:
            self.trust = float(np.clip(
                self.trust * self.reliability.value / p.trust_setpoint,
                p.trust_min, 1.0))

        if p.feedback == "persistent":
            # SCI sees only tracks alive in frames t, t-1 and t-2, so
            # transient (mostly false) candidates cannot inflate crowd/tiny.
            ids = self.reliability.ids
            stable = (ids[0] & ids[1] & ids[2]) if len(ids) == 3 else set()
            feedback = [t for t in tracks if t.track_id in stable]
            self.previous = [[t.x1, t.y1, t.x2, t.y2,
                              min(max(t.confidence, 0.5), 1 - 1e-6),
                              t.class_id] for t in feedback]
        elif self.config.detector_feedback:
            pool = normalized if p.feedback == "normalized" else detections
            feedback = [d for d in pool if d.confidence >= high]
            self.previous = [[d.x1, d.y1, d.x2, d.y2, d.confidence,
                              d.class_id] for d in feedback]
        else:
            feedback = [t for t in tracks if t.confidence >= high]
            self.previous = [[t.x1, t.y1, t.x2, t.y2, t.confidence,
                              t.class_id] for t in feedback]

        audit = dict(
            frame=frame_number, sci=float(params["sci"]),
            scene=params["scene"], resolution=int(params["size"]),
            raw_count=len(raw),
            raw_count_ema=(dens.raw_count_ema if dens and
                           dens.raw_count_ema is not None else np.nan),
            base_sensitivity=base.sensitivity,
            effective_sensitivity=(dens.sensitivity if dens
                                   else base.sensitivity),
            low_threshold=low, high_threshold=high, new_track_threshold=new,
            target_candidates=(target if target is not None else -1),
            eligible_before_topk=len(eligible),
            accepted_after_topk=len(detections),
            topk_hit=int(target is not None and len(eligible) > target),
            tracks=len(tracks),
            feedback_count=len(feedback),
            reliability=rel,
            trust=trust_used,
            leader_ref=(leader_ref if leader_ref is not None else np.nan),
            demoted=demoted,
            obs_survival=self.reliability.survival,
            obs_persist3=self.reliability.persist3,
            obs_new_fraction=self.reliability.new_fraction,
        )
        return dict(params=params, raw_detections=raw,
                    detections=detections, tracks=tracks, audit=audit)


def describe(policy: PolicySpec, density_kwargs=None):
    d = asdict(policy)
    d.update({f"density_{k}": v for k, v in (density_kwargs or {}).items()})
    return d


__all__ = ["PolicySpec", "POLICIES", "UniversalPolicyPipeline", "describe",
           "replace"]
