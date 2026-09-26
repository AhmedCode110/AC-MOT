from adapters.detectors.base import DetectorAdapter
from adapters.detectors.capabilities import DetectorCapabilities
from adapters.detectors.yolov8 import YOLOv8Adapter
from adapters.detectors.rtdetr import RTDETRAdapter

__all__ = [
    "DetectorAdapter",
    "DetectorCapabilities",
    "YOLOv8Adapter",
    "RTDETRAdapter",
]
