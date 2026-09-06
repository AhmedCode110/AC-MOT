"""Candidate A3 controller. Values are experimental, not optimized results."""
class StableCalibrator:
    def __init__(self, adaptive_threshold=True, adaptive_resolution=True):
        self.adaptive_threshold = adaptive_threshold
        self.adaptive_resolution = adaptive_resolution
        self.current = 640
        self.pending = None
        self.count = 0
        self.last_state = None

    def params(self, state):
        target = 640
        if self.adaptive_resolution:
            if state.sci > .60 or state.tiny_ratio > .50:
                target = 832
            elif state.sci > .35 or state.scene in ['crowded', 'tiny']:
                target = 736
        # Count distinct analyzer updates, not repeated calls on the same state.
        if state is not self.last_state:
            self.count = self.count + 1 if target == self.pending else 1
            self.pending = target
            self.last_state = state
            if self.count >= 3:
                self.current = target
        # Allow ByteTrack's low-confidence recovery stage to see detections.
        # Track birth remains controlled by new_track_thresh in tracker YAML.
        conf = .04 if self.adaptive_threshold else .25
        iou = max(.40, min(.52, .490 - .050 * state.sci)) if self.adaptive_threshold else .45
        return dict(conf=conf, iou=iou, imgsz=self.current)
