from abc import ABC, abstractmethod

from adapters.detectors.capabilities import DetectorCapabilities
from adapters.types import DetectionList


class DetectorAdapter(ABC):

    @property
    @abstractmethod
    def capabilities(self) -> DetectorCapabilities:
        raise NotImplementedError

    @abstractmethod
    def detect(
        self,
        frame,
        confidence: float,
        suppression: float,
        resolution: int,
    ) -> DetectionList:
        raise NotImplementedError
