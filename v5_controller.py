"""
V5 scene-adaptive decision layer: an interpretable mapping from the generic
scene/tracking state (scene_state.py) to per-frame operating decisions.

Spec (JSON): {"targets": {<target>: <node>}}, target in
  resolution | fixed_sensitivity | gate_tau | assoc_offset | retention
node = constant value, or
       {"feature": name, "threshold": t, "le": node, "gt": node}
A missing/NaN feature (e.g. before the first observation) takes the "le"
branch. Trees are learned offline on development sequences only
(tools/v5_*.py); the deployed controller uses causal state only.
"""
from __future__ import annotations

import json
import math


def _linear_score(node, state):
    """C3 node: weighted mean of development-quantile-normalised cues."""
    import numpy as np
    num, den = 0.0, 0.0
    for t in node["terms"]:
        v = state.get(t["feature"])
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        u = float(np.interp(v, t["knots"], np.linspace(0, 1, len(t["knots"]))))
        num += t["weight"] * ((1 - u) if t["invert"] else u)
        den += t["weight"]
    return None if den == 0 else num / den


def _eval(node, state):
    while isinstance(node, dict):
        if node.get("type") == "linear":
            z = _linear_score(node, state)
            node = node["gt"] if (z is not None and z > node["threshold"]) \
                else node["le"]
            continue
        v = state.get(node["feature"])
        go_gt = v is not None and not (isinstance(v, float) and math.isnan(v)) \
            and v > node["threshold"]
        node = node["gt"] if go_gt else node["le"]
    return node


class V5Controller:
    def __init__(self, targets):
        self.targets = dict(targets)

    @classmethod
    def from_json(cls, spec):
        d = json.loads(spec) if isinstance(spec, str) else spec
        return cls(d["targets"])

    def decide(self, state):
        return {t: _eval(n, state) for t, n in self.targets.items()}

    def features(self):
        out = set()

        def walk(n):
            if isinstance(n, dict) and n.get("type") == "linear":
                out.update(t["feature"] for t in n["terms"])
                walk(n["le"])
                walk(n["gt"])
            elif isinstance(n, dict):
                out.add(n["feature"])
                walk(n["le"])
                walk(n["gt"])
        for n in self.targets.values():
            walk(n)
        return sorted(out)
