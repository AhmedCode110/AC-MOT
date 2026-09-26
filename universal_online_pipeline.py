from __future__ import annotations

from time import perf_counter

from core import Controller
from adapters.detectors.online_normalizer import (
    OnlineScoreNormalizer,
)
from adapters.detectors.generic_controls import (
    resolve_generic_score_controls,
)


class UniversalOnlineACMOTPipeline:
    """
    Universal AC-MOT using online score normalization.

    Detector runs at a low raw confidence floor.
    AC controls are applied to normalized scores.
    """

    def __init__(
        self,
        config,
        detector,
        tracker,
        raw_confidence_floor: float = 0.01,
    ):
        self.config = config.validate()

        self.controller = Controller(
            self.config
        )

        self.detector = detector
        self.tracker = tracker

        self.raw_confidence_floor = float(
            raw_confidence_floor
        )

        self.normalizer = OnlineScoreNormalizer(
            bins=64,
            decay=0.95,
            update_every=10,
        )

        self.previous = []

    def process(
        self,
        frame_number,
        image,
        visual,
    ):
        params = self.controller.choose(
            frame_number,
            visual,
            self.previous,
        )

        # Detector-specific raw score is NOT controlled
        # directly by AC anymore.
        raw_detections = self.detector.detect(
            image,
            confidence=self.raw_confidence_floor,
            suppression=params["nms"],
            resolution=params["size"],
        )

        # Convert detector-specific scores to a generic scale.
        t_norm = perf_counter()

        normalized_detections = (
            self.normalizer.process(
                raw_detections
            )
        )

        normalizer_ms = (
            perf_counter() - t_norm
        ) * 1000.0

        # Convert scene complexity into detector-independent
        # percentile-space controls.
        generic_controls = resolve_generic_score_controls(
            sci=params["sci"],
            scene=params["scene"],
            recovery=self.config.recovery,
        )

        # Candidate gate now works on normalized scores,
        # not detector-specific raw confidence.
        detections = [
            d
            for d in normalized_detections
            if d.confidence
            >= generic_controls.low_threshold
        ]

        tracks = self.tracker.update(
            detections,
            image.shape[:2],
            high_thresh=generic_controls.high_threshold,
            new_track_thresh=generic_controls.new_track_threshold,
        )

        # SCI feedback should use confident observations only.
        # This preserves the semantic role of the legacy >=0.18
        # confidence filter while operating in normalized score space.
        feedback_detections = [
            d
            for d in normalized_detections
            if d.confidence
            >= generic_controls.high_threshold
        ]

        feedback_tracks = [
            t
            for t in tracks
            if t.confidence
            >= generic_controls.high_threshold
        ]

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
                for d in feedback_detections
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
                for t in feedback_tracks
            ]

        return {
            "params": params,
            "generic_controls": generic_controls,

            # Useful for auditing.
            "raw_detections": raw_detections,
            "normalized_detections": normalized_detections,

            # Detections actually passed to tracker.
            "detections": detections,

            "tracks": tracks,

            "feedback_detections":
                feedback_detections,

            "feedback_tracks":
                feedback_tracks,

            "normalizer_initialized":
                self.normalizer.initialized,

            "normalizer_ms":
                normalizer_ms,
        }

    def reset(self):
        self.controller = Controller(
            self.config
        )

        self.tracker.reset()
        self.normalizer.reset()

        self.previous = []
