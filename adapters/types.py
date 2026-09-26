from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


@dataclass
class Track:
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    confidence: float
    class_id: int


DetectionList = List[Detection]
TrackList = List[Track]
