from abc import ABC, abstractmethod
from adapters.types import DetectionList, TrackList


class TrackerAdapter(ABC):

    @abstractmethod
    def update(
        self,
        detections: DetectionList,
        frame_shape,
    ) -> TrackList:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
