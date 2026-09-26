from __future__ import annotations

import numpy as np

from adapters.types import Detection, DetectionList


class OnlineScoreNormalizer:
    """
    Lightweight causal online score normalization.

    - No offline detector calibration.
    - Fixed 64-bin rolling histogram.
    - CDF is rebuilt only when histogram updates.
    - Current frame is normalized using previous statistics.
    """

    def __init__(
        self,
        bins: int = 64,
        decay: float = 0.95,
        update_every: int = 10,
    ):
        self.bins = int(bins)
        self.decay = float(decay)
        self.update_every = int(update_every)

        self.hist = np.zeros(
            self.bins,
            dtype=np.float32,
        )

        self.cdf = np.zeros(
            self.bins,
            dtype=np.float32,
        )

        self.frame_count = 0
        self.initialized = False

    def reset(self):
        self.hist.fill(0.0)
        self.cdf.fill(0.0)

        self.frame_count = 0
        self.initialized = False

    def _score_indices(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:

        idx = (
            np.clip(
                scores,
                0.0,
                0.999999,
            )
            * self.bins
        ).astype(
            np.int32,
            copy=False,
        )

        return np.minimum(
            idx,
            self.bins - 1,
        )

    def _rebuild_cdf(self):
        total = float(
            self.hist.sum()
        )

        if total <= 0.0:
            self.cdf.fill(0.0)
            return

        np.cumsum(
            self.hist,
            out=self.cdf,
        )

        self.cdf /= total

    def _update_histogram(
        self,
        scores: np.ndarray,
    ):
        if scores.size == 0:
            return

        idx = self._score_indices(
            scores
        )

        counts = np.bincount(
            idx,
            minlength=self.bins,
        ).astype(
            np.float32,
            copy=False,
        )

        if not self.initialized:
            self.hist[:] = counts
            self.initialized = True
        else:
            self.hist *= self.decay

            self.hist += (
                1.0 - self.decay
            ) * counts

        self._rebuild_cdf()

    def _normalize_scores(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:

        if not self.initialized:
            return np.clip(
                scores,
                0.0,
                1.0,
            )

        idx = self._score_indices(
            scores
        )

        normalized = self.cdf[idx]

        # Keep scores strictly inside (0, 1).
        # Legacy core.boxes() validates confidence values
        # and must never receive exact boundary values.
        return np.clip(
            normalized,
            1e-6,
            1.0 - 1e-6,
        )

    def process(
        self,
        detections: DetectionList,
    ) -> DetectionList:

        self.frame_count += 1

        if not detections:
            return []

        scores = np.fromiter(
            (
                d.confidence
                for d in detections
            ),
            dtype=np.float32,
            count=len(detections),
        )

        # Bootstrap only.
        if not self.initialized:
            self._update_histogram(
                scores
            )

        normalized_scores = (
            self._normalize_scores(
                scores
            )
        )

        result = [
            Detection(
                x1=d.x1,
                y1=d.y1,
                x2=d.x2,
                y2=d.y2,
                confidence=float(score),
                class_id=d.class_id,
            )
            for d, score in zip(
                detections,
                normalized_scores,
            )
        ]

        # Causal:
        # current frame affects only future frames.
        if (
            self.frame_count > 1
            and
            self.frame_count
            % self.update_every
            == 0
        ):
            self._update_histogram(
                scores
            )

        return result


class EmpiricalCDFNormalizer:
    """
    Order-only online score normalization (V3).

    Maps a raw score to its mid-rank position among raw candidate scores
    sampled from PREVIOUS frames (one frame every `update_every`, the last
    `window` samples — the same data and memory span as the 64-bin
    histogram, without binning). Depends only on the order of scores, so it
    is exactly invariant to any strictly increasing recalibration of the
    detector. Frame 1 (no history) uses within-frame mid-ranks.
    """

    def __init__(self, update_every: int = 10, window: int = 20):
        self.update_every = int(update_every)
        self.window = int(window)
        self.reset()

    def reset(self):
        from collections import deque
        self.samples = deque(maxlen=self.window)
        self.sorted = np.empty(0)
        self.frame_count = 0
        self.initialized = False

    @staticmethod
    def _midrank(sorted_ref, scores):
        lo = np.searchsorted(sorted_ref, scores, side="left")
        hi = np.searchsorted(sorted_ref, scores, side="right")
        return (lo + hi) / (2.0 * len(sorted_ref))

    def process(self, detections: DetectionList) -> DetectionList:
        self.frame_count += 1
        if not detections:
            return []
        scores = np.fromiter((d.confidence for d in detections),
                             dtype=np.float64, count=len(detections))
        ref = self.sorted if self.initialized else np.sort(scores)
        norm = np.clip(self._midrank(ref, scores), 1e-6, 1.0 - 1e-6)
        out = [Detection(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                         confidence=float(s), class_id=d.class_id)
               for d, s in zip(detections, norm)]
        # Frame t is folded in after use (affects frames > t only).
        if self.frame_count == 1 or self.frame_count % self.update_every == 0:
            self.samples.append(scores)
            self.sorted = np.sort(np.concatenate(self.samples))
            self.initialized = True
        return out
