import os
import cv2
import numpy as np
from ultralytics import YOLO

from adapters.detectors.yolov8 import YOLOv8Adapter, auto_device


WEIGHTS = os.environ["ACMOT_WEIGHTS"]
FRAME = os.environ["ACMOT_FRAME"]

CONF = 0.25
NMS = 0.45
IMGSZ = 640
CLASSES = [0, 2, 5, 7]


img = cv2.imread(FRAME)

if img is None:
    raise RuntimeError(f"Could not read frame: {FRAME}")


device = auto_device()

print("Device :", device)
print("Weights:", WEIGHTS)
print("Frame  :", FRAME)
print()


# ------------------------------------------------------------
# Legacy AC-MOT detector path
# ------------------------------------------------------------

legacy_model = YOLO(WEIGHTS)

legacy_result = legacy_model.predict(
    img,
    conf=0.01,
    iou=NMS,
    imgsz=IMGSZ,
    classes=CLASSES,
    max_det=1000,
    half=False,
    device=device,
    verbose=False,
)[0]


if legacy_result.boxes is None or len(legacy_result.boxes) == 0:
    legacy = np.empty((0, 6), dtype=np.float32)
else:
    legacy = (
        legacy_result.boxes.data
        .detach()
        .cpu()
        .numpy()[:, :6]
        .astype(np.float32)
    )


# AC-MOT confidence filtering happens after detector output.
legacy = legacy[legacy[:, 4] >= CONF]


# ------------------------------------------------------------
# Universal AC-MOT detector adapter
# ------------------------------------------------------------

adapter = YOLOv8Adapter(
    weights=WEIGHTS,
    device=device,
    classes=CLASSES,
)

adapter_objects = adapter.detect(
    img,
    confidence=CONF,
    suppression=NMS,
    resolution=IMGSZ,
)


adapted = np.asarray(
    [
        [
            d.x1,
            d.y1,
            d.x2,
            d.y2,
            d.confidence,
            d.class_id,
        ]
        for d in adapter_objects
    ],
    dtype=np.float32,
).reshape(-1, 6)


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

print("Legacy detections :", len(legacy))
print("Adapter detections:", len(adapted))

same_count = len(legacy) == len(adapted)

print("Same count        :", same_count)


if not same_count:
    raise AssertionError(
        f"Detection count mismatch: "
        f"legacy={len(legacy)} adapter={len(adapted)}"
    )


if len(legacy):
    max_difference = float(
        np.max(np.abs(legacy - adapted))
    )
else:
    max_difference = 0.0


print("Max difference    :", max_difference)


if len(legacy) and not np.allclose(
    legacy,
    adapted,
    atol=1e-4,
    rtol=1e-4,
):
    print("\nLEGACY:")
    print(legacy)

    print("\nADAPTER:")
    print(adapted)

    raise AssertionError(
        "Adapter output differs from legacy detector output"
    )


print()
print("========================================")
print("PASS: YOLOv8 Adapter matches legacy path")
print("========================================")
