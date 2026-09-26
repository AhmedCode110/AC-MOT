from __future__ import annotations

from pathlib import Path

from adapters.detectors.yolov8 import YOLOv8Adapter
from adapters.detectors.rtdetr import RTDETRAdapter


def infer_detector_family(weights: str) -> str:
    """
    Infer supported detector family from weights filename.

    Current supported families:
      - Ultralytics YOLO family
      - RT-DETR

    No detector-specific calibration is required.
    """

    name = Path(weights).name.lower()

    if (
        "rtdetr" in name
        or "rt-detr" in name
        or "rt_detr" in name
    ):
        return "rtdetr"

    # YOLO-family models:
    # yolov8n.pt, yolo11n.pt, yolov10n.pt, etc.
    if "yolo" in name:
        return "yolo"

    # Generic custom Ultralytics weights are commonly called best.pt.
    # Default to YOLO-family; caller can override if needed.
    return "yolo"


def create_detector(
    weights: str,
    family: str = "auto",
):
    """
    Universal detector factory.

    Examples:
        create_detector("yolov8n.pt")
        create_detector("yolo11n.pt")
        create_detector("rtdetr-l.pt")

    family may be:
        auto
        yolo
        rtdetr
    """

    weights = str(
        Path(weights).expanduser()
    )

    if not Path(weights).exists():
        raise FileNotFoundError(
            f"Detector weights not found: {weights}"
        )

    family = family.lower().strip()

    if family == "auto":
        family = infer_detector_family(
            weights
        )

    if family in {
        "yolo",
        "yolov8",
        "yolov10",
        "yolo11",
        "yolo12",
    }:
        return YOLOv8Adapter(
            weights
        )

    if family in {
        "rtdetr",
        "rt-detr",
        "rt_detr",
    }:
        return RTDETRAdapter(
            weights
        )

    raise ValueError(
        f"Unsupported detector family: {family}"
    )
