from __future__ import annotations

from core import Controller


class UniversalACMOTPipeline:
    """
    Adapter-based AC-MOT pipeline.

    The existing AC-MOT Controller is preserved unchanged.
    Detector and tracker implementations are injected through adapters.
    """

    def __init__(
        self,
        config,
        detector,
        tracker,
    ):
        self.config = config.validate()
        self.controller = Controller(self.config)

        self.detector = detector
        self.tracker = tracker

        # Previous standardized observations used causally
        # by the controller for the next decision.
        self.previous = []

    def process(
        self,
        frame_number,
        image,
        visual,
    ):
        """
        Process one frame.

        Order is intentionally:

        previous observations
            -> Controller
            -> detector settings
            -> Detector Adapter
            -> Tracker Adapter
            -> observations for next frame
        """

        params = self.controller.choose(
            frame_number,
            visual,
            self.previous,
        )

        detections = self.detector.detect(
            image,
            confidence=params["conf"],
            suppression=params["nms"],
            resolution=params["size"],
        )

        tracks = self.tracker.update(
            detections,
            image.shape[:2],
            high_thresh=params["high"],
            new_track_thresh=params["new"],
        )

        if self.config.detector_feedback:
            self.previous = [
                [
                    d.x1,
                    d.y1,
                    d.x2,
                    d.y2,
                    d.confidence,
                    d.class_id,
                ]
                for d in detections
            ]

        else:
            self.previous = [
                [
                    t.x1,
                    t.y1,
                    t.x2,
                    t.y2,
                    t.confidence,
                    t.class_id,
                ]
                for t in tracks
            ]

        return {
            "params": params,
            "detections": detections,
            "tracks": tracks,
        }

    def reset(self):
        self.controller = Controller(self.config)
        self.tracker.reset()
        self.previous = []
