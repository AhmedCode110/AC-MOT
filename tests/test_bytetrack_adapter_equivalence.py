import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from adapters.trackers.bytetrack import ByteTrackAdapter
from adapters.types import Detection


WEIGHTS = Path(os.environ["ACMOT_WEIGHTS"])
SEQUENCE = Path(os.environ["ACMOT_SEQUENCE"])

CONF = 0.25
NMS = 0.45
IMGSZ = 640
CLASSES = [0, 2, 5, 7]

HIGH = 0.25
LOW = 0.10
NEW = 0.25
BUFFER = 30
MATCH = 0.80
FUSE = True

NUM_FRAMES = 20


def make_legacy_tracker():
    args = SimpleNamespace(
        track_high_thresh=HIGH,
        track_low_thresh=LOW,
        new_track_thresh=NEW,
        track_buffer=BUFFER,
        match_thresh=MATCH,
        fuse_score=FUSE,
    )

    return BYTETracker(
        args,
        frame_rate=30,
    )


def detect(model, img):
    result = model.predict(
        img,
        conf=0.01,
        iou=NMS,
        imgsz=IMGSZ,
        classes=CLASSES,
        max_det=1000,
        half=False,
        device="mps",
        verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    dets = (
        result.boxes.data
        .detach()
        .cpu()
        .numpy()[:, :6]
        .astype(np.float32)
    )

    return dets[dets[:, 4] >= CONF]


def to_detection_objects(dets):
    return [
        Detection(
            x1=float(row[0]),
            y1=float(row[1]),
            x2=float(row[2]),
            y2=float(row[3]),
            confidence=float(row[4]),
            class_id=int(row[5]),
        )
        for row in dets
    ]


# ============================================================
# CACHE EXACT SAME DETECTIONS FOR BOTH TRACKER RUNS
# ============================================================

frames = sorted(SEQUENCE.glob("*.jpg"))[:NUM_FRAMES]

if len(frames) < NUM_FRAMES:
    raise RuntimeError(
        f"Need {NUM_FRAMES} frames, found {len(frames)}"
    )

model = YOLO(str(WEIGHTS))

cached = []

print("Caching identical detector outputs...")

for frame_path in frames:
    img = cv2.imread(str(frame_path))

    if img is None:
        raise RuntimeError(f"Could not read {frame_path}")

    dets = detect(model, img)

    cached.append(
        {
            "dets": dets.copy(),
            "shape": img.shape[:2],
        }
    )

print("Detection cache complete.")
print()


# ============================================================
# RUN 1 — LEGACY BYTETRACK
# ============================================================

BYTETracker.reset_id()

legacy_tracker = make_legacy_tracker()

legacy_outputs = []

for item in cached:

    result = np.asarray(
        legacy_tracker.update(
            Boxes(
                item["dets"].copy(),
                item["shape"],
            )
        ),
        dtype=float,
    ).reshape(-1, 8)

    legacy_outputs.append(
        result[:, :7].copy()
    )


# ============================================================
# RUN 2 — ADAPTER BYTETRACK
# ============================================================

BYTETracker.reset_id()

adapter_tracker = ByteTrackAdapter(
    high=HIGH,
    low=LOW,
    new=NEW,
    buffer=BUFFER,
    match=MATCH,
    fuse=FUSE,
    frame_rate=30,
)

adapter_outputs = []

for item in cached:

    tracks = adapter_tracker.update(
        to_detection_objects(
            item["dets"].copy()
        ),
        item["shape"],
        high_thresh=HIGH,
        new_track_thresh=NEW,
    )

    result = np.asarray(
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
            for t in tracks
        ],
        dtype=float,
    ).reshape(-1, 7)

    adapter_outputs.append(result)


# ============================================================
# FRAME-BY-FRAME EQUIVALENCE
# ============================================================

print("Sequence:", SEQUENCE)
print("Frames  :", NUM_FRAMES)
print()

for frame_number, (legacy, adapted, item) in enumerate(
    zip(
        legacy_outputs,
        adapter_outputs,
        cached,
    ),
    1,
):

    if len(legacy) != len(adapted):
        raise AssertionError(
            f"Frame {frame_number}: "
            f"track count mismatch "
            f"legacy={len(legacy)} "
            f"adapter={len(adapted)}"
        )

    if len(legacy):

        if not np.allclose(
            legacy,
            adapted,
            atol=1e-6,
            rtol=1e-6,
        ):
            print()
            print("LEGACY:")
            print(legacy)

            print()
            print("ADAPTER:")
            print(adapted)

            raise AssertionError(
                f"Frame {frame_number}: "
                "ByteTrack output mismatch"
            )

    ids = (
        adapted[:, 4].astype(int).tolist()
        if len(adapted)
        else []
    )

    print(
        f"Frame {frame_number:02d} | "
        f"dets={len(item['dets']):2d} | "
        f"tracks={len(adapted):2d} | "
        f"IDs={ids} | PASS"
    )


print()
print("==========================================")
print("PASS: ByteTrack Adapter matches legacy path")
print("==========================================")
