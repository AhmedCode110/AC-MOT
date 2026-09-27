from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from adapters.trackers.base import TrackerAdapter
from adapters.types import DetectionList, Track, TrackList


class ByteTrackAdapter(TrackerAdapter):
    """
    Behavior-preserving adapter for the ByteTrack path used by AC-MOT.
    """

    def __init__(
        self,
        high: float = 0.25,
        low: float = 0.10,
        new: float = 0.25,
        buffer: int = 30,
        match: float = 0.80,
        fuse: bool = True,
        frame_rate: int = 30,
    ):
        self.high = float(high)
        self.low = float(low)
        self.new = float(new)
        self.buffer = int(buffer)
        self.match = float(match)
        self.fuse = bool(fuse)
        self.frame_rate = int(frame_rate)

        self.tracker = self._make_tracker()

    def _make_tracker(self):
        args = SimpleNamespace(
            track_high_thresh=self.high,
            track_low_thresh=self.low,
            new_track_thresh=self.new,
            track_buffer=self.buffer,
            match_thresh=self.match,
            fuse_score=self.fuse,
        )

        return BYTETracker(
            args,
            frame_rate=self.frame_rate,
        )

    @staticmethod
    def _detections_to_numpy(
        detections: DetectionList,
    ) -> np.ndarray:

        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        result = np.asarray(
            [
                [
                    d.x1,
                    d.y1,
                    d.x2,
                    d.y2,
                    d.confidence,
                    d.class_id,
                ]
                for d in detections
            ],
            dtype=np.float32,
        ).reshape(-1, 6)

        return result

    def update(
        self,
        detections: DetectionList,
        frame_shape,
        association_threshold: float | None = None,
        birth_threshold: float | None = None,
        image=None,
        *,
        high_thresh: float | None = None,
        new_track_thresh: float | None = None,
    ) -> TrackList:
        # Generic names map onto ByteTrack's own; legacy keyword names are
        # kept so the V1 pipeline and equivalence tests are unchanged.
        if high_thresh is None:
            high_thresh = association_threshold
        if new_track_thresh is None:
            new_track_thresh = birth_threshold

        dets = self._detections_to_numpy(detections)

        if high_thresh is not None:
            self.tracker.args.track_high_thresh = float(high_thresh)

        if new_track_thresh is not None:
            self.tracker.args.new_track_thresh = float(new_track_thresh)

        result = np.asarray(
            self._update_backend(
                Boxes(dets, tuple(frame_shape)), image
            ),
            dtype=float,
        ).reshape(-1, 8)

        if len(result):
            ids = result[:, 4]

            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate track IDs")

        tracks: TrackList = []

        for row in result:
            tracks.append(
                Track(
                    x1=float(row[0]),
                    y1=float(row[1]),
                    x2=float(row[2]),
                    y2=float(row[3]),
                    track_id=int(row[4]),
                    confidence=float(row[5]),
                    class_id=int(row[6]),
                )
            )

        return tracks

    def _update_backend(self, boxes, image):
        return self.tracker.update(boxes)

    def set_retention(self, frames: int) -> None:
        # ultralytics BYTETracker: max_time_lost = frame_rate/30 * buffer
        self.tracker.max_time_lost = int(self.frame_rate / 30.0 * int(frames))

    def reset(self):
        self.tracker = self._make_tracker()
