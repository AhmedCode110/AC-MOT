from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import torch

from adapters.detectors.base import DetectorAdapter
from adapters.detectors.capabilities import DetectorCapabilities
from adapters.detectors.fasterrcnn import COCO80_TO_91, DEFAULT_CLASSES, auto_device
from adapters.types import Detection, DetectionList


class FCOSAdapter(DetectorAdapter):
    """
    torchvision FCOS ResNet50-FPN (COCO) adapter.

    Anchor-free one-stage detector whose score is sqrt(classification x
    centerness), a family and a confidence scale used nowhere in the
    project's development. Chosen as the unseen detector of General AC-MOT
    G2 before any G2 tuning (research/final/G2_EXPERIMENT_LEDGER.md); run
    only after G2 is frozen and its transfer lock is committed.

    Only API translation lives here (no performance tuning):
      confidence -> score_thresh (low floor, same 0.01 as the others)
      suppression -> nms_thresh (generic IoU request; None = native 0.6)
      resolution -> longest image side (min_size = max_size = resolution),
                    the same meaning as for the other adapters
      classes -> COCO-91 ids, returned as the project's COCO-80 ids
    detections_per_img = 1000 as for the Faster R-CNN adapter; everything
    else stays at torchvision defaults.
    """

    def __init__(self, weights: str, device: str | None = None,
                 classes: Iterable[int] = DEFAULT_CLASSES,
                 max_det: int = 1000, inference_conf_floor: float = 0.01):
        from torchvision.models.detection import fcos_resnet50_fpn

        self.weights = weights
        self.device = device or auto_device()
        self.classes = list(classes)
        self.inference_conf_floor = float(inference_conf_floor)
        self.to80 = {v: k for k, v in COCO80_TO_91.items()}
        self.keep91 = torch.tensor([COCO80_TO_91[c] for c in self.classes])
        model = fcos_resnet50_fpn(
            weights=None, weights_backbone=None,
            score_thresh=self.inference_conf_floor,
            detections_per_img=int(max_det))
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model = model.eval().to(self.device)
        self.native_nms = float(model.nms_thresh)       # torchvision 0.6

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
        m.nms_thresh = self.native_nms if suppression is None else float(suppression)
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
