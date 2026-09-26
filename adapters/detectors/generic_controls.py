from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenericScoreControls:
    sensitivity: float
    low_threshold: float
    high_threshold: float
    new_track_threshold: float


def resolve_generic_score_controls(
    *,
    sci: float,
    scene: str,
    recovery: bool,
) -> GenericScoreControls:
    """
    Detector-independent controls operating on normalized
    percentile-like confidence scores.

    sensitivity:
        approximate fraction of detector candidates allowed
        into the tracking candidate pool.

    This is a universal policy, not detector-specific calibration.
    """

    sensitivity = (
        0.22
        + 0.18 * float(sci)
    )

    if scene in {
        "crowded",
        "tiny",
        "night",
    }:
        sensitivity += 0.05

    if recovery:
        sensitivity += 0.08

    sensitivity = max(
        0.20,
        min(0.50, sensitivity),
    )

    # CDF-normalized scores:
    # 0.70 means roughly top 30%.
    low_threshold = (
        1.0 - sensitivity
    )

    high_threshold = min(
        0.95,
        low_threshold + 0.18,
    )

    new_track_threshold = min(
        0.98,
        high_threshold + 0.05,
    )

    return GenericScoreControls(
        sensitivity=sensitivity,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        new_track_threshold=new_track_threshold,
    )
