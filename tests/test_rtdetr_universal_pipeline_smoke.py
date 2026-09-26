import os
from pathlib import Path

import cv2

from core import Config
from experiment import visual

from adapters.detectors.rtdetr import RTDETRAdapter
from adapters.trackers.bytetrack import ByteTrackAdapter
from universal_pipeline import UniversalACMOTPipeline


WEIGHTS = Path(os.environ["RTDETR_WEIGHTS"])
SEQUENCE = Path(os.environ["ACMOT_SEQUENCE"])

NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "20"))


cfg = Config(
    name="rtdetr_universal_smoke",
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

    nms=0.45,
    fuse=True,
).validate()


detector = RTDETRAdapter(
    weights=str(WEIGHTS),
)

tracker = ByteTrackAdapter(
    high=cfg.high,
    low=cfg.low,
    new=cfg.new,
    buffer=cfg.buffer,
    match=cfg.match,
    fuse=cfg.fuse,
    frame_rate=30,
)

pipeline = UniversalACMOTPipeline(
    config=cfg,
    detector=detector,
    tracker=tracker,
)


paths = sorted(SEQUENCE.glob("*.jpg"))[:NUM_FRAMES]

if len(paths) < NUM_FRAMES:
    raise RuntimeError(
        f"Need {NUM_FRAMES} frames, found {len(paths)}"
    )


print("============================================")
print("RT-DETR + ByteTrack + Universal AC-MOT")
print("============================================")

for frame_number, path in enumerate(paths, 1):

    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"Could not read {path}")

    scene_visual = (
        visual(image)
        if frame_number == 1
        or frame_number % 10 == 1
        else {}
    )

    output = pipeline.process(
        frame_number,
        image,
        scene_visual,
    )

    p = output["params"]

    print(
        f"Frame {frame_number:02d} | "
        f"SCI={p['sci']:.4f} | "
        f"size={p['size']} | "
        f"conf={p['conf']:.4f} | "
        f"dets={len(output['detections']):3d} | "
        f"tracks={len(output['tracks']):3d} | "
        f"PASS"
    )


print()
print("================================================")
print("PASS: RT-DETR works through Universal AC-MOT")
print("================================================")
