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
from adapters.detectors.online_normalizer import (EmpiricalCDFNormalizer,
                                                  OnlineScoreNormalizer)
from adapters.types import Detection


def _logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(x / (1 - x))


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
    # V3 (calibration invariance):
    # normalizer "hist" = 64-bin histogram (V1..V2g); "ecdf" = order-only
    # empirical CDF (exactly invariant to monotone recalibration).
    normalizer: str = "hist"
    # gate_stat "ratio" = raw s / leader (invariant to scaling only);
    # "zlogit" = (logit s - EMA leader logit) / IQR(logits of candidates in
    # the previous gate_window frames): exactly invariant to any Platt /
    # temperature recalibration logit' = a*logit + b (a > 0).
    # Demote if z < -gate_tau.
    gate_stat: str = "ratio"
    gate_tau: float = 0.0
    gate_window: int = 10
    # Raw-score floor applied inside the policy (a raw-scale constant).
    # V3 sets 0: candidate emission is left to the detector adapter.
    policy_raw_floor: float = 0.01
    # Controller audit switches (0 = legacy SCI-driven behaviour):
    # fixed_resolution: bypass SCI resolution selection.
    # fixed_sensitivity: bypass the SCI -> generic sensitivity mapping.
    fixed_resolution: int = 0
    fixed_sensitivity: float = 0.0
    # Cue audit: "cue:<name>" allocates 832 when the cue is at or above its
    # own running median (this sequence, analysis steps so far), else 640;
    # "random" allocates 832 with probability 0.5 per analysis step
    # (deterministic hash). Empty = legacy SCI resolution rule.
    res_policy: str = ""
    # V4: no scene controller in the decision path (cue audit E25/E26:
    # no scene cue beats random resolution allocation at matched compute).
    # Resolution comes from fixed_resolution (compute budget) and the
    # score thresholds from fixed_sensitivity + the two offsets below.
    scene_controller: bool = True
    assoc_offset: float = 0.18
    birth_offset: float = 0.05
    # Tracker retention/association: "ac" = legacy AC values (buffer 45,
    # match 0.86); "native" = the tracker's own defaults (30, 0.8).
    tracker_defaults: str = "ac"
    # ECDF memory: one score sample every ecdf_stride frames, last
    # ecdf_window samples (default = 200-frame span, same as V1 histogram).
    # Generic suppression request passed to the detector adapter when the
    # scene controller is off (0.45 = legacy AC request).
    nms_request: float | None = 0.45   # None = detector-native suppression
    # V5 development: log the generic scene/tracking state every frame and
    # (optionally) let a learned controller override per-frame decisions.
    scene_state: bool = False
    controller_spec: str = ""        # JSON; empty = no V5 controller
    tracker_buffer: int = 0           # fixed retention override (frames)
    # V5-TF (Amendment 6): training-free online candidate handling.
    # "otsu3_window": 3-class Otsu on candidate logits of frames < t;
    # "otsu3_frame": on frame t's own candidates. Empty = legacy paths.
    candidate_mode: str = ""
    # F3: association tolerance scaled by relative global motion.
    assoc_motion: bool = False
    ecdf_stride: int = 10
    ecdf_window: int = 20


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
        self.normalizer = (
            EmpiricalCDFNormalizer(update_every=self.policy.ecdf_stride,
                                   window=self.policy.ecdf_window)
            if self.policy.normalizer == "ecdf" else
            OnlineScoreNormalizer(bins=64, decay=0.95, update_every=10))
        from scene_state import SceneStateAnalyzer
        from v5_controller import V5Controller
        self.analyzer = SceneStateAnalyzer()
        self.v5 = (V5Controller.from_json(self.policy.controller_spec)
                   if self.policy.controller_spec else None)
        self.cue_hist = []
        self.cue_size = 640
        self.gate_ref_logit = None
        self.gate_logits = deque(maxlen=self.policy.gate_window)
        from online_calibration import RobustHistory
        self.otsu_window = deque(maxlen=self.policy.gate_window)
        self.motion_hist = RobustHistory()
        self.tf_log = {}
        self.density = CandidateDensityController(**self.density_kwargs)
        self.reliability = TemporalReliability(self.policy.reliability_ema)
        self.trust = 1.0
        self.leader_ref = None
        self.tracker.reset()
        if self.policy.tracker_buffer:
            self.tracker.set_retention(self.policy.tracker_buffer)
        self.previous = []

    def _otsu_bands(self, raw, normalized, mode):
        """V5-TF candidate handling: 3-class Otsu on logits (frames < t for
        the window mode; frame t for the frame mode). Returns the banded
        candidates and the tracker thresholds (band boundaries)."""
        from online_calibration import logits, otsu3
        if not raw:
            return [], 0.1, 0.5, 0.5, np.empty(0)
        L = logits([r.confidence for r in raw])
        ref = (np.concatenate(self.otsu_window)
               if mode == "otsu3_window" and self.otsu_window else L)
        th = otsu3(ref)
        out = []
        if th is None:
            t1, t2, eta = -np.inf, -np.inf, 0.0
        else:
            t1, t2, eta = th
        n_prim = n_sec = 0
        for li, d in zip(L, normalized):
            u = d.confidence
            if li >= t2:
                out.append(replace_det(d, 0.5 + 0.5 * u))
                n_prim += 1
            elif li >= t1:
                out.append(replace_det(d, 0.1 + 0.4 * u))
                n_sec += 1
        self.tf_log.update(otsu_t1=t1, otsu_t2=t2, otsu_eta=eta,
                           n_primary=n_prim, n_secondary=n_sec)
        return out, 0.1, 0.5, 0.5, L

    def _cues(self, visual):
        """Causal scene cues: detector feedback from frames < t and image
        statistics of frame t."""
        b = np.asarray(self.previous, dtype=float).reshape(-1, 6)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return dict(
            n_fb=float(len(b)),
            tiny=float((area < 1024).mean()) if len(b) else 0.0,
            neg_log_area=float(-np.median(np.log(np.maximum(area, 1))))
            if len(b) else 0.0,
            edges=float(visual["edges"]),
            darkness=float(-visual["brightness"]),
            blurriness=float(-visual["blur"]),
            legacy_sci=float(self.controller.sci),
        )

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
        if p.scene_controller:
            params = self.controller.choose(frame_number, visual,
                                            self.previous)
        else:
            params = dict(size=int(p.fixed_resolution or 736),
                          nms=(None if p.nms_request is None
                               else float(p.nms_request)),
                          sci=0.0, scene="clear")
        use_state = p.scene_state or self.v5 is not None
        state = {}
        decision = {}
        if use_state:
            self.analyzer.observe_image(visual, image.shape[:2])
            state = self.analyzer.state()
        if self.v5 is not None:
            decision = self.v5.decide(state)
            p = replace(p, **{k: v for k, v in decision.items()
                              if k in ("fixed_sensitivity", "gate_tau",
                                       "assoc_offset")})
            if "resolution" in decision:
                params = dict(params, size=int(decision["resolution"]))
            if "retention" in decision:
                self.tracker.set_retention(int(decision["retention"]))
        if p.fixed_resolution:
            params = dict(params, size=int(p.fixed_resolution))
        cues = self._cues(visual) if p.scene_controller else {}
        if p.res_policy:
            if frame_number == 1 or frame_number % 10 == 1:
                if p.res_policy == "random":
                    h = (frame_number * 2654435761) & 0xFFFFFFFF
                    self.cue_size = 832 if (h >> 16) & 1 else 640
                else:
                    c = cues[p.res_policy.split(":", 1)[1]]
                    self.cue_hist.append(c)
                    med = float(np.median(self.cue_hist))
                    self.cue_size = 832 if c >= med and \
                        len(self.cue_hist) > 1 else 640
            params = dict(params, size=self.cue_size)
        raw = self.detector.detect(image, confidence=p.policy_raw_floor,
                                   suppression=params["nms"],
                                   resolution=params["size"])
        normalized = self.normalizer.process(raw)
        base = resolve_generic_score_controls(
            sci=params["sci"], scene=params["scene"],
            recovery=self.config.recovery)
        if p.fixed_sensitivity:
            fs = float(p.fixed_sensitivity)
            hi = min(0.95, 1 - fs + p.assoc_offset)
            base = type(base)(sensitivity=fs, low_threshold=1 - fs,
                              high_threshold=hi,
                              new_track_threshold=min(0.98,
                                                      hi + p.birth_offset))

        rel = self.reliability.value
        dens = None
        if p.density:
            budget_sci = params["sci"] * rel if p.reliability_budget \
                else params["sci"]
            dens = self.density.decide(
                base_sensitivity=base.sensitivity, sci=budget_sci,
                budget_scale=self.trust)

        low, high, new = self._thresholds(base, dens, rel)
        otsu_L = None
        if p.candidate_mode:
            normalized, low, high, new, otsu_L = self._otsu_bands(
                raw, normalized, p.candidate_mode)
        if p.assoc_motion:
            m = (visual or {}).get("motion")
            r = self.motion_hist.ratio(m)
            m0 = self.tracker.native_match
            self.tracker.set_association_tolerance(
                min(0.95, 1.0 - (1.0 - m0) / max(1.0, r)))
            self.tf_log["motion_ratio"] = r
            self.motion_hist.push(m)

        demoted = 0
        leader_ref = self.leader_ref
        if p.candidate_mode:            # V5-TF: no gate / density path
            leader_ref = None
        norm_u = [d.confidence for d in normalized]
        gate_ref_prev = self.gate_ref_logit
        gate_iqr_prev = (float(np.subtract(*np.percentile(
            np.concatenate(self.gate_logits), [75, 25])))
            if self.gate_logits else 1.0)
        if not p.candidate_mode and p.gate_stat == "zlogit" and p.gate_tau > 0 and raw \
                and self.gate_ref_logit is not None and self.gate_logits:
            lg = _logit(np.array([r.confidence for r in raw]))
            q75, q25 = np.percentile(np.concatenate(self.gate_logits),
                                     [75, 25])
            z = (lg - self.gate_ref_logit) / max(q75 - q25, 1e-6)
            gated = []
            for zi, d in zip(z, normalized):
                if zi >= -p.gate_tau:
                    gated.append(d)
                elif p.leader_mode == "demote":
                    demoted += 1
                    gated.append(replace_det(d, min(d.confidence,
                                                    high - 1e-6)))
                else:
                    demoted += 1
            normalized = gated
        if p.gate_stat == "zlogit" and raw:
            lg_all = _logit(np.array([r.confidence for r in raw]))
            top = float(lg_all.max())
            self.gate_ref_logit = top if self.gate_ref_logit is None else (
                p.leader_decay * self.gate_ref_logit
                + (1 - p.leader_decay) * top)
            self.gate_logits.append(lg_all)
        if p.gate_stat == "ratio" and p.leader_rho > 0 \
                and leader_ref is not None and raw:
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

        if otsu_L is not None:
            self.otsu_window.append(otsu_L)
        if use_state:
            self.analyzer.observe_detector(
                np.array([[r.x1, r.y1, r.x2, r.y2] for r in raw]).reshape(-1, 4),
                np.array([r.confidence for r in raw]), norm_u,
                gate_ref_prev, gate_iqr_prev)
            self.analyzer.observe_tracks(
                [t.track_id for t in tracks],
                sum(d.confidence >= high for d in detections))

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
            **{f"cue_{k}": v for k, v in cues.items()},
            **{f"s_{k}": v for k, v in state.items()},
            **{f"d_{k}": v for k, v in decision.items()},
            **{f"tf_{k}": v for k, v in self.tf_log.items()},
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
