"""
Universal AC-MOT — plug-and-play entry point.

    from universal_acmot import UniversalACMOT
    from adapters.detectors.factory import create_detector
    from adapters.trackers.bytetrack import ByteTrackAdapter

    system = UniversalACMOT(detector=create_detector("rtdetr-l.pt"),
                            tracker=ByteTrackAdapter())
    for frame in frames:            # BGR numpy arrays, in order
        tracks = system(frame)      # list[adapters.types.Track]

The system never sees the detector's or tracker's identity: it only uses
the DetectorAdapter.detect() / TrackerAdapter.update() contracts. All
operating parameters come from one shared policy file
(configs/universal_acmot_policy.json) — no per-detector values.

Causal ordering for frame t:
  1. visual cues of frame t (detector-independent image statistics)
  2. controller reads state built from frames < t  → resolution, SCI
  3. detector runs on frame t → raw candidates
  4. normalizer maps raw scores with histogram from frames < t
  5. leader gate compares raw scores with leader EMA from frames < t
  6. density controller: budget from SCI (frames < t), raw-count EMA
     (frames < t) → Top-K on frame-t candidates
  7. tracker update with generic association / birth thresholds
  8. state updates with frame-t observations (used from t+1 on)
"""
from __future__ import annotations

import json
from pathlib import Path

from run_universal_acmot import analyze_visual, build_config
from universal_policy_pipeline import (POLICIES, PolicySpec,
                                       UniversalPolicyPipeline, replace)

CONFIGS = Path(__file__).with_name("configs")
V3_POLICY_FILE = CONFIGS / "universal_acmot_policy.json"      # V3 checkpoint
DEFAULT_POLICY_FILE = CONFIGS / "universal_acmot_policy_v4.json"  # final


def load_policy(path=DEFAULT_POLICY_FILE):
    spec = json.loads(Path(path).read_text())
    base = POLICIES[spec["base"]]
    policy = replace(base, **spec.get("overrides", {}))
    return policy, spec.get("density_kwargs", {})


class ResolutionBudget:
    """
    Detector control by compute budget (V4). The scene-complexity rule was
    removed (no scene cue beat random allocation at matched compute), so
    the controller spends pixels by budget: it times the detector at each
    level on the first frames (online self-calibration, detector-agnostic)
    and keeps the largest level whose median latency fits 1/target_fps.
    """

    def __init__(self, levels=(640, 736, 832), target_fps=None, probes=3):
        self.levels = sorted(levels)
        self.target_fps = target_fps
        self.probes = int(probes)
        self.samples = {r: [] for r in self.levels}
        self.chosen = None if target_fps else self.levels[-1]

    def level_for(self, frame_number):
        if self.chosen is not None:
            return self.chosen
        k = (frame_number - 1) // (self.probes + 1)   # +1 warm-up per level
        return self.levels[min(k, len(self.levels) - 1)]

    def observe(self, level, seconds, frame_number):
        if self.chosen is not None:
            return
        if (frame_number - 1) % (self.probes + 1) != 0:  # skip warm-up
            self.samples[level].append(seconds)
        if all(len(v) >= self.probes for v in self.samples.values()):
            import numpy as np
            budget = 1.0 / self.target_fps
            ok = [r for r in self.levels
                  if np.median(self.samples[r]) <= budget]
            self.chosen = ok[-1] if ok else self.levels[0]


class UniversalACMOT:
    def __init__(self, detector, tracker, policy: PolicySpec | None = None,
                 density_kwargs: dict | None = None, policy_file=None,
                 resolution=None, target_fps=None):
        if policy is None:
            policy, dk = load_policy(policy_file or DEFAULT_POLICY_FILE)
            density_kwargs = dk if density_kwargs is None else density_kwargs
        self.budget = None
        if resolution == "auto":
            self.budget = ResolutionBudget(target_fps=target_fps)
        elif resolution:
            policy = replace(policy, fixed_resolution=int(resolution))
        self.config = build_config()
        self.pipeline = UniversalPolicyPipeline(
            self.config, detector, tracker, policy,
            density_kwargs=density_kwargs)
        self.frame_number = 0
        self.last = None

    def __call__(self, frame):
        from time import perf_counter
        self.frame_number += 1
        if self.budget is not None:
            level = self.budget.level_for(self.frame_number)
            self.pipeline.policy = replace(self.pipeline.policy,
                                           fixed_resolution=level)
        t0 = perf_counter()
        self.last = self.pipeline.process(self.frame_number, frame,
                                          analyze_visual(frame))
        if self.budget is not None:
            self.budget.observe(level, perf_counter() - t0,
                                self.frame_number)
        return self.last["tracks"]

    def reset(self):
        self.pipeline.reset()
        self.frame_number = 0
        self.last = None
