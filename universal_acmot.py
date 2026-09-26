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

DEFAULT_POLICY_FILE = Path(__file__).with_name("configs") / \
    "universal_acmot_policy.json"


def load_policy(path=DEFAULT_POLICY_FILE):
    spec = json.loads(Path(path).read_text())
    base = POLICIES[spec["base"]]
    policy = replace(base, **spec.get("overrides", {}))
    return policy, spec.get("density_kwargs", {})


class UniversalACMOT:
    def __init__(self, detector, tracker, policy: PolicySpec | None = None,
                 density_kwargs: dict | None = None, policy_file=None):
        if policy is None:
            policy, dk = load_policy(policy_file or DEFAULT_POLICY_FILE)
            density_kwargs = dk if density_kwargs is None else density_kwargs
        self.config = build_config()
        self.pipeline = UniversalPolicyPipeline(
            self.config, detector, tracker, policy,
            density_kwargs=density_kwargs)
        self.frame_number = 0
        self.last = None

    def __call__(self, frame):
        self.frame_number += 1
        self.last = self.pipeline.process(self.frame_number, frame,
                                          analyze_visual(frame))
        return self.last["tracks"]

    def reset(self):
        self.pipeline.reset()
        self.frame_number = 0
        self.last = None
