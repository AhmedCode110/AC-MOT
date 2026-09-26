from __future__ import annotations

from typing import Iterable

import torch
from ultralytics import RTDETR

from adapters.detectors.base import DetectorAdapter
from adapters.detectors.capabilities import DetectorCapabilities
from adapters.types import Detection, DetectionList


DEFAULT_CLASSES = [0, 2, 5, 7]


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


class RTDETRAdapter(DetectorAdapter):
    """
    RT-DETR adapter for Universal AC-MOT.

    Initial conservative capability declaration:
    - adaptive confidence: supported
    - adaptive resolution: not enabled until explicitly validated
    - adaptive suppression/NMS: unsupported

    AC confidence is applied after low-floor detector inference.
    """

    def __init__(
        self,
        weights: str,
        device: str | None = None,
        classes: Iterable[int] = DEFAULT_CLASSES,
        max_det: int = 1000,
        inference_conf_floor: float = 0.01,
        fixed_resolution: int = 640,
    ):
        self.weights = weights
        self.device = device or auto_device()
        self.classes = list(classes)
        self.max_det = int(max_det)
        self.inference_conf_floor = float(inference_conf_floor)
        self.fixed_resolution = int(fixed_resolution)

        self.model = RTDETR(weights)

    @property
    def capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities(
            adaptive_confidence=True,
            adaptive_resolution=True,
            adaptive_suppression=False,
        )

    def detect(
        self,
        frame,
        confidence: float,
        suppression: float,
        resolution: int,
    ) -> DetectionList:

        # RT-DETR does not use conventional NMS.
        # Resolution is temporarily frozen until dynamic-size
        # behavior is validated explicitly.
        _ = suppression
        _ = resolution

        result = self.model.predict(
            frame,
            conf=self.inference_conf_floor,
            imgsz=int(resolution),
            classes=self.classes,
            max_det=self.max_det,
            half=False,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        rows = (
            result.boxes.data
            .detach()
            .cpu()
            .numpy()
        )

        detections: DetectionList = []

        for row in rows:
            x1, y1, x2, y2, score, class_id = row[:6]

            if float(score) < float(confidence):
                continue

            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(score),
                    class_id=int(class_id),
                )
            )

        return detections
