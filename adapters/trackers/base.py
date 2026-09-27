from abc import ABC, abstractmethod

from adapters.types import DetectionList, TrackList


class TrackerAdapter(ABC):
    """
    Generic tracker interface used by the Universal AC-MOT policy.

    The policy only speaks in tracker-independent terms:

    association_threshold:
        candidates at or above it take part in primary association;
        candidates below it may only extend already-existing tracks
        (secondary / low-score association where the tracker has one).
    birth_threshold:
        minimum (normalized) confidence for an unmatched candidate to start
        a new track.

    Each adapter maps these onto its own API. `needs_image` tells callers
    whether update() requires the frame (e.g. camera-motion compensation).
    """

    needs_image: bool = False

    @abstractmethod
    def update(
        self,
        detections: DetectionList,
        frame_shape,
        association_threshold: float | None = None,
        birth_threshold: float | None = None,
        image=None,
    ) -> TrackList:
        raise NotImplementedError

    native_match: float = 0.8

    def set_association_tolerance(self, value: float) -> None:
        """Generic association-tolerance command (max matching cost);
        default: unsupported."""
        return None

    def set_retention(self, frames: int) -> None:
        """Generic retention command: how many frames a lost track is kept.
        Adapters translate it to their own API; default: unsupported."""
        return None

    @abstractmethod
    def reset(self):
        raise NotImplementedError
