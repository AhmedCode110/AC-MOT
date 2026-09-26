from abc import ABC, abstractmethod
from adapters.types import DetectionList


class DetectorAdapter(ABC):

    @abstractmethod
    def detect(
        self,
        frame,
        confidence: float,
        suppression: float,
        resolution: int,
    ) -> DetectionList:
        raise NotImplementedError
