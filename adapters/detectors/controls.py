from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EffectiveDetectorControls:
    requested_confidence: float
    requested_resolution: int
    requested_suppression: float

    effective_confidence: float | None
    effective_resolution: int | None
    effective_suppression: float | None

    confidence_supported: bool
    resolution_supported: bool
    suppression_supported: bool


def resolve_effective_controls(
    detector,
    *,
    confidence: float,
    resolution: int,
    suppression: float,
) -> EffectiveDetectorControls:

    caps = detector.capabilities

    effective_confidence = (
        float(confidence)
        if caps.adaptive_confidence
        else getattr(detector, "fixed_confidence", None)
    )

    effective_resolution = (
        int(resolution)
        if caps.adaptive_resolution
        else getattr(detector, "fixed_resolution", None)
    )

    effective_suppression = (
        float(suppression)
        if caps.adaptive_suppression
        else getattr(detector, "fixed_suppression", None)
    )

    return EffectiveDetectorControls(
        requested_confidence=float(confidence),
        requested_resolution=int(resolution),
        requested_suppression=float(suppression),

        effective_confidence=effective_confidence,
        effective_resolution=effective_resolution,
        effective_suppression=effective_suppression,

        confidence_supported=bool(
            caps.adaptive_confidence
        ),
        resolution_supported=bool(
            caps.adaptive_resolution
        ),
        suppression_supported=bool(
            caps.adaptive_suppression
        ),
    )
