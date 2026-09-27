from __future__ import annotations

from typing import Iterable

import torch
from ultralytics import YOLO

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


class YOLOv8Adapter(DetectorAdapter):

    @property
    def capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities(
            adaptive_confidence=True,
            adaptive_resolution=True,
            adaptive_suppression=True,
        )

    """
    YOLOv8 detector adapter for Universal AC-MOT.

    Important:
    - AC confidence is applied AFTER YOLO inference.
    - YOLO inference uses a low internal confidence floor (0.01).
    - This preserves the original AC-MOT detector behavior.
    """

    def __init__(
        self,
        weights: str,
        device: str | None = None,
        classes: Iterable[int] = DEFAULT_CLASSES,
        max_det: int = 1000,
        inference_conf_floor: float = 0.01,
    ):
        self.weights = weights
        self.device = device or auto_device()
        self.classes = list(classes)
        self.max_det = int(max_det)
        self.inference_conf_floor = float(inference_conf_floor)

        self.model = YOLO(weights)

    def detect(
        self,
        frame,
        confidence: float,
        suppression: float,
        resolution: int,
    ) -> DetectionList:

        nms_kw = {} if suppression is None else {"iou": float(suppression)}
        result = self.model.predict(
            frame,
            conf=self.inference_conf_floor,
            **nms_kw,
            imgsz=int(resolution),
            classes=self.classes,
            max_det=self.max_det,
            half=False,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        rows = result.boxes.data.detach().cpu().numpy()

        detections: DetectionList = []

        for row in rows:
            x1, y1, x2, y2, score, class_id = row[:6]

            # AC-MOT confidence threshold:
            # applied after detector inference to preserve legacy behavior.
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
