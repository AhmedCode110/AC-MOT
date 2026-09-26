from __future__ import annotations

from types import SimpleNamespace

from ultralytics.trackers.bot_sort import BOTSORT

from adapters.trackers.bytetrack import ByteTrackAdapter


class BoTSORTAdapter(ByteTrackAdapter):
    """
    BoT-SORT (ultralytics) behind the generic tracker interface.

    Same two-stage association semantics as ByteTrack plus camera-motion
    compensation (sparse optical flow), so it needs the frame. ReID is off
    (no appearance model), matching the ultralytics default.
    Only the generic association/birth thresholds are driven by the AC
    policy; all BoT-SORT-specific options stay at ultralytics defaults.
    """

    needs_image = True

    def __init__(self, *args, gmc_method: str = "sparseOptFlow", **kwargs):
        self.gmc_method = gmc_method
        super().__init__(*args, **kwargs)

    def _make_tracker(self):
        args = SimpleNamespace(
            track_high_thresh=self.high,
            track_low_thresh=self.low,
            new_track_thresh=self.new,
            track_buffer=self.buffer,
            match_thresh=self.match,
            fuse_score=self.fuse,
            gmc_method=self.gmc_method,
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto",
        )
        return BOTSORT(args, frame_rate=self.frame_rate)

    def _update_backend(self, boxes, image):
        return self.tracker.update(boxes, image)
