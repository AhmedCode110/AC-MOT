"""Detector adapters for G2 tools. Native suppression per family (None =
the adapter's own native value). Locked files (adapters/detectors/factory.py,
tools/sci_v7/build_sweep_cache.py) are not modified."""
from __future__ import annotations

from pathlib import Path

NATIVE_NMS = {"yolov8": 0.7, "rtdetr": None, "fasterrcnn": 0.5, "retinanet": None, "fcos": None}


def make_detector(weights):
    name = Path(weights).name.lower()
    if "fcos" in name:
        from adapters.detectors.fcos import FCOSAdapter
        return FCOSAdapter(weights)
    from tools.sci_v7.build_sweep_cache import make_detector as base
    return base(weights)
