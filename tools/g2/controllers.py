"""G2 compute controllers (generic: no detector, tracker or dataset names).
Populated only for adaptation spaces that pass the oracle gate
(research/final/G2_EXPERIMENT_LEDGER.md)."""
from __future__ import annotations

REGISTRY = {}


def make(name, cost, n_frames):
    if name not in REGISTRY:
        raise ValueError(f"unknown controller {name}")
    return REGISTRY[name](cost=cost, n_frames=n_frames)
