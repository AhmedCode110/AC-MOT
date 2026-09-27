from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import torch

from adapters.detectors.base import DetectorAdapter
from adapters.detectors.capabilities import DetectorCapabilities
from adapters.types import Detection, DetectionList

# Project class ids are COCO-80 (0 person, 2 car, 5 bus, 7 truck);
# torchvision detection models use COCO-91 category ids.
COCO80_TO_91 = {0: 1, 2: 3, 5: 6, 7: 8}
DEFAULT_CLASSES = [0, 2, 5, 7]


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FasterRCNNAdapter(DetectorAdapter):
    """
    torchvision Faster R-CNN ResNet50-FPN v2 (COCO) adapter.

    Two-stage detector: a different family from the one-stage YOLO and the
    set-prediction RT-DETR used during development. Added AFTER the
    Universal AC-MOT policy was frozen, as an unseen transfer test.

    Only API translation lives here (no performance tuning):
      confidence -> box_score_thresh (low floor, same 0.01 as the others)
      suppression -> roi_heads.nms_thresh (generic IoU request)
      resolution -> longest image side (min_size = max_size = resolution),
                    the same "longest side" meaning as Ultralytics imgsz
      classes -> COCO-91 ids, returned as the project's COCO-80 ids
    Everything else stays at torchvision defaults.
    """

    def __init__(self, weights: str, device: str | None = None,
                 classes: Iterable[int] = DEFAULT_CLASSES,
                 max_det: int = 1000, inference_conf_floor: float = 0.01):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

        self.weights = weights
        self.device = device or auto_device()
        self.classes = list(classes)
        self.inference_conf_floor = float(inference_conf_floor)
        self.to80 = {v: k for k, v in COCO80_TO_91.items()}
        self.keep91 = torch.tensor([COCO80_TO_91[c] for c in self.classes])

        model = fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None,
            box_score_thresh=self.inference_conf_floor,
            box_detections_per_img=int(max_det))
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state)
        self.model = model.eval().to(self.device)

    @property
    def capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities(adaptive_confidence=True,
                                    adaptive_resolution=True,
                                    adaptive_suppression=True)

    @torch.inference_mode()
    def detect(self, frame, confidence: float, suppression: float,
               resolution: int) -> DetectionList:
        m = self.model
        m.transform.min_size = (int(resolution),)
        m.transform.max_size = int(resolution)
        m.roi_heads.nms_thresh = float(suppression)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        x = x.to(self.device, dtype=torch.float32).div_(255.0)
        out = m([x])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        keep = np.isin(labels, self.keep91.numpy()) & (scores >= confidence)
        return [Detection(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]),
                          y2=float(b[3]), confidence=float(s),
                          class_id=int(self.to80[int(l)]))
                for b, s, l in zip(boxes[keep], scores[keep], labels[keep])]
