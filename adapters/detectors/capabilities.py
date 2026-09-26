from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorCapabilities:
    """
    Runtime controls supported by a detector adapter.
    """

    adaptive_confidence: bool
    adaptive_resolution: bool
    adaptive_suppression: bool
