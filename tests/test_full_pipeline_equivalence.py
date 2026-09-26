import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from core import Config, Controller, boxes
from experiment import make_tracker, track, visual

from adapters.types import Detection
from adapters.trackers.bytetrack import ByteTrackAdapter
from universal_pipeline import UniversalACMOTPipeline


WEIGHTS = Path(os.environ["ACMOT_WEIGHTS"])
SEQUENCE = Path(os.environ["ACMOT_SEQUENCE"])

NUM_FRAMES = 20

CLASSES = [0, 2, 5, 7]
SIZES = [640, 736, 832]
NMS = 0.45


# ------------------------------------------------------------
# Representative adaptive AC-MOT configuration
# ------------------------------------------------------------

CFG = Config(
    name="integration_equivalence",
    policy="adaptive",
    stable=True,
    recovery=True,
    detector_feedback=True,
    adaptive_birth=True,

    high=0.18,
    low=0.04,
    new=0.20,
    buffer=45,
    match=0.86,

    nms=NMS,
    fuse=True,
).validate()


# ------------------------------------------------------------
# Detector cache adapter
#
# Both legacy and universal paths receive EXACTLY the same
# detector outputs. This isolates pipeline architecture.
# ------------------------------------------------------------

class CachedDetectorAdapter:

    def __init__(self, cache):
        self.cache = cache
        self.index = 0

    def detect(
        self,
        frame,
        confidence,
        suppression,
        resolution,
    ):
        if self.index >= len(self.cache):
            raise RuntimeError("Detector cache exhausted")

        item = self.cache[self.index]
        self.index += 1

        key = (
            int(resolution),
            round(float(suppression), 2),
        )

        if key not in item["bank"]:
            raise KeyError(
                f"Missing detector bank: {key}"
            )

        raw = item["bank"][key]

        # Same position as current AC-MOT confidence filtering.
        filtered = raw[
            raw[:, 4] >= float(confidence)
        ]

        return [
            Detection(
                x1=float(row[0]),
                y1=float(row[1]),
                x2=float(row[2]),
                y2=float(row[3]),
                confidence=float(row[4]),
                class_id=int(row[5]),
            )
            for row in filtered
        ]


# ------------------------------------------------------------
# Helper conversions
# ------------------------------------------------------------

def detections_array(items):
    if not items:
        return np.empty((0, 6), dtype=np.float32)

    return np.asarray(
        [
            [
                d.x1,
                d.y1,
                d.x2,
                d.y2,
                d.confidence,
                d.class_id,
            ]
            for d in items
        ],
        dtype=np.float32,
    ).reshape(-1, 6)


def tracks_array(items):
    if not items:
        return np.empty((0, 7), dtype=float)

    return np.asarray(
        [
            [
                t.x1,
                t.y1,
                t.x2,
                t.y2,
                t.track_id,
                t.confidence,
                t.class_id,
            ]
            for t in items
        ],
        dtype=float,
    ).reshape(-1, 7)


def compare_params(frame, legacy, universal):

    if legacy.keys() != universal.keys():
        raise AssertionError(
            f"Frame {frame}: parameter keys differ"
        )

    for key in legacy:

        a = legacy[key]
        b = universal[key]

        if isinstance(a, (float, np.floating)):
            if not np.isclose(a, b, atol=1e-12):
                raise AssertionError(
                    f"Frame {frame}: "
                    f"parameter {key} differs: "
                    f"{a} vs {b}"
                )
        else:
            if a != b:
                raise AssertionError(
                    f"Frame {frame}: "
                    f"parameter {key} differs: "
                    f"{a} vs {b}"
                )


# ------------------------------------------------------------
# Load test frames
# ------------------------------------------------------------

paths = sorted(SEQUENCE.glob("*.jpg"))[:NUM_FRAMES]

if len(paths) < NUM_FRAMES:
    raise RuntimeError(
        f"Need {NUM_FRAMES} frames, "
        f"found {len(paths)}"
    )


# ------------------------------------------------------------
# Cache detector outputs ONCE
#
# 20 frames x three resolutions.
#
# No FPS claims are made from this test.
# ------------------------------------------------------------

print("==========================================")
print("Building shared detector cache")
print("==========================================")

model = YOLO(str(WEIGHTS))

cache = []

for frame_number, path in enumerate(paths, 1):

    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(
            f"Could not read {path}"
        )

    bank = {}

    for size in SIZES:

        result = model.predict(
            image,
            conf=0.01,
            iou=NMS,
            imgsz=size,
            classes=CLASSES,
            max_det=1000,
            half=False,
            device="mps",
            verbose=False,
        )[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            raw = np.empty(
                (0, 6),
                dtype=np.float32,
            )
        else:
            raw = boxes(
                result.boxes.data
                .detach()
                .cpu()
                .numpy()[:, :6]
            )

        bank[(size, NMS)] = raw.copy()

    scene_visual = (
        visual(image)
        if frame_number == 1
        or frame_number % 10 == 1
        else {}
    )

    cache.append(
        {
            "path": path,
            "shape": image.shape[:2],
            "visual": scene_visual,
            "bank": bank,
        }
    )

    print(
        f"Cached frame "
        f"{frame_number:02d}/{NUM_FRAMES}"
    )


# ============================================================
# RUN 1
# LEGACY AC-MOT PATH
# ============================================================

print()
print("==========================================")
print("Running legacy AC-MOT path")
print("==========================================")

BYTETracker.reset_id()

legacy_controller = Controller(CFG)

legacy_tracker = make_tracker(CFG)

legacy_previous = []

legacy_outputs = []


for frame_number, item in enumerate(cache, 1):

    params = legacy_controller.choose(
        frame_number,
        item["visual"],
        legacy_previous,
    )

    key = (
        params["size"],
        round(params["nms"], 2),
    )

    raw = item["bank"][key]

    tracks, detections = track(
        legacy_tracker,
        raw,
        item["shape"],
        params,
    )

    if CFG.detector_feedback:
        legacy_previous = detections
    else:
        legacy_previous = tracks[
            :,
            [0, 1, 2, 3, 5, 6],
        ]

    legacy_outputs.append(
        {
            "params": dict(params),
            "detections": detections.copy(),
            "tracks": tracks[:, :7].copy(),
        }
    )


# ============================================================
# RUN 2
# UNIVERSAL ADAPTER PATH
# ============================================================

print()
print("==========================================")
print("Running Universal AC-MOT path")
print("==========================================")

BYTETracker.reset_id()

universal_detector = CachedDetectorAdapter(
    cache
)

universal_tracker = ByteTrackAdapter(
    high=CFG.high,
    low=CFG.low,
    new=CFG.new,
    buffer=CFG.buffer,
    match=CFG.match,
    fuse=CFG.fuse,
    frame_rate=30,
)

pipeline = UniversalACMOTPipeline(
    config=CFG,
    detector=universal_detector,
    tracker=universal_tracker,
)

universal_outputs = []


for frame_number, item in enumerate(cache, 1):

    image = cv2.imread(
        str(item["path"])
    )

    output = pipeline.process(
        frame_number,
        image,
        item["visual"],
    )

    universal_outputs.append(
        {
            "params": dict(
                output["params"]
            ),
            "detections": detections_array(
                output["detections"]
            ),
            "tracks": tracks_array(
                output["tracks"]
            ),
        }
    )


# ============================================================
# EXACT FRAME-BY-FRAME COMPARISON
# ============================================================

print()
print("==========================================")
print("Comparing complete pipelines")
print("==========================================")

for frame_number, (legacy, universal) in enumerate(
    zip(
        legacy_outputs,
        universal_outputs,
    ),
    1,
):

    # AC decision
    compare_params(
        frame_number,
        legacy["params"],
        universal["params"],
    )

    # Detector output after AC confidence filtering
    if not np.allclose(
        legacy["detections"],
        universal["detections"],
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError(
            f"Frame {frame_number}: "
            "detections differ"
        )

    # Tracker output including track IDs
    if not np.allclose(
        legacy["tracks"],
        universal["tracks"],
        atol=1e-6,
        rtol=1e-6,
    ):
        print()
        print("LEGACY TRACKS:")
        print(legacy["tracks"])

        print()
        print("UNIVERSAL TRACKS:")
        print(universal["tracks"])

        raise AssertionError(
            f"Frame {frame_number}: "
            "tracks differ"
        )

    params = universal["params"]

    ids = (
        universal["tracks"][:, 4]
        .astype(int)
        .tolist()
        if len(universal["tracks"])
        else []
    )

    print(
        f"Frame {frame_number:02d} | "
        f"SCI={params['sci']:.4f} | "
        f"size={params['size']} | "
        f"conf={params['conf']:.4f} | "
        f"tracks={len(ids):2d} | "
        f"PASS"
    )


print()
print("================================================")
print("PASS: FULL Universal AC-MOT matches legacy path")
print("================================================")
