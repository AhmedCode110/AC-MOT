"""Tracker hosts for the SCI + V7f runner, each behind the same tracker
interface (update(dets, shape, association_threshold, birth_threshold,
image), set_association_tolerance) with its own host contract. Host-specific
code lives here and in adapters/trackers, never in the generic layers."""
from __future__ import annotations

from tools.v7.systems import HOST_BYTETRACK

HOSTS = ("bytetrack", "botsort", "ocsort")


class _OCSortTracker:
    """Official OC-SORT (noahcao/OC_SORT @ 8462e7e) through the V7 record's host
    wrapper; single threshold, so the host gets min(association, birth)."""

    def __init__(self, contract):
        from tools.v7.dev import _ocsort_host
        self.t = _ocsort_host(contract)

    def set_association_tolerance(self, m):
        self.t.set_association_tolerance(m)

    def update(self, dets, shape, association_threshold=None, birth_threshold=None, image=None):
        from tools.v7.dev import _ocsort_classes
        tracks = self.t.update([[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in dets], shape,
                               min(association_threshold, birth_threshold))
        _ocsort_classes(tracks, dets, self.t)
        return tracks


def make_host(name):
    """-> (tracker, contract dict, needs_image)"""
    if name == "bytetrack":
        from adapters.trackers.bytetrack import ByteTrackAdapter
        h = HOST_BYTETRACK
        return ByteTrackAdapter(high=h["assoc"], low=h["low"], new=h["birth"], buffer=30, match=h["match"],
                                fuse=True), h, False
    if name == "botsort":
        from adapters.trackers.botsort import BoTSORTAdapter
        h = HOST_BYTETRACK          # ultralytics botsort.yaml shares bytetrack.yaml's thresholds
        return BoTSORTAdapter(high=h["assoc"], low=h["low"], new=h["birth"], buffer=30, match=h["match"],
                              fuse=True), h, True
    if name == "ocsort":
        from tools.v7.dev import HOST_OCSORT
        return _OCSortTracker(HOST_OCSORT), HOST_OCSORT, False
    raise ValueError(name)
