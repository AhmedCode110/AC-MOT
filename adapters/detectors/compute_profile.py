"""
Detector side of the generic compute level (LOW / MEDIUM / HIGH) chosen by
the scene layer (acmot_sci.py). A profile maps each level to a setting the
detector actually supports; the adapter refuses a profile that names an
unsupported setting instead of inventing one.

Profiles live in configs/sci_v7_profiles.json, one per detector. This is
the only place where a level gets a detector-native meaning.
"""
from __future__ import annotations

import json
from pathlib import Path

from acmot_sci import LEVELS

PROFILES = Path(__file__).resolve().parents[2] / "configs" / "sci_v7_profiles.json"


class ComputeProfileAdapter:
    def __init__(self, profile: dict, supported=None, cost=None):
        missing = [lv for lv in LEVELS if lv not in profile]
        if missing:
            raise ValueError(f"profile has no setting for {missing}")
        if supported is not None:
            bad = {lv: v for lv, v in profile.items() if v not in supported}
            if bad:
                raise ValueError(f"detector does not support {bad} (supported: {sorted(supported)})")
        self.profile = {lv: int(profile[lv]) for lv in LEVELS}
        # relative compute of one frame at a setting (default: pixels of a
        # square input, r^2)
        self.cost = cost or (lambda r: float(r) ** 2)

    @classmethod
    def from_config(cls, detector: str, supported=None, path=PROFILES, key="resolution"):
        cfg = json.loads(Path(path).read_text())
        return cls(cfg["detectors"][detector][key], supported)

    def resolution(self, level: str) -> int:
        return self.profile[level]

    def relative_cost(self, level: str, reference: int) -> float:
        return self.cost(self.profile[level]) / self.cost(reference)
