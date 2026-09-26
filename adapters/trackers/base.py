from abc import ABC, abstractmethod
from adapters.types import DetectionList, TrackList


class TrackerAdapter(ABC):

    @abstractmethod
    def update(
        self,
        detections: DetectionList,
        frame_shape,
        high_thresh: float | None = None,
        new_track_thresh: float | None = None,
    ) -> TrackList:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
