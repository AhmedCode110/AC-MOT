from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from core import Config
from adapters.detectors.yolov8 import YOLOv8Adapter
from adapters.detectors.rtdetr import RTDETRAdapter
from adapters.trackers.bytetrack import ByteTrackAdapter
from universal_online_pipeline import UniversalOnlineACMOTPipeline


def visual(image):
    small = cv2.resize(
        image,
        None,
        fx=0.25,
        fy=0.25,
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY,
    )

    edges = float(
        cv2.Canny(
            gray,
            50,
            120,
        ).mean() / 255.0
    )

    brightness = float(
        gray.mean()
    )

    blur = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    return {
        "edges": edges,
        "brightness": brightness,
        "blur": blur,
    }


def tracker_from_config(cfg):
    return ByteTrackAdapter(
        high=cfg.high,
        low=cfg.low,
        new=cfg.new,
        buffer=cfg.buffer,
        match=cfg.match,
        fuse=cfg.fuse,
    )


def score_stats(detections):
    if not detections:
        return (0.0, 0.0, 0.0)

    scores = np.asarray(
        [d.confidence for d in detections],
        dtype=float,
    )

    return (
        float(scores.min()),
        float(np.median(scores)),
        float(scores.max()),
    )


def run_detector(
    name,
    detector,
    sequence,
    num_frames,
):
    cfg = Config(
        name=f"online_{name}",
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

    tracker = tracker_from_config(cfg)

    pipeline = UniversalOnlineACMOTPipeline(
        config=cfg,
        detector=detector,
        tracker=tracker,
        raw_confidence_floor=0.01,
    )

    paths = sorted(
        sequence.glob("*.jpg")
    )[:num_frames]

    if not paths:
        raise RuntimeError(
            f"No frames found in {sequence}"
        )

    norm_times = []

    print()
    print("=" * 110)
    print(name)
    print("=" * 110)

    print(
        f"{'Frm':>3} "
        f"{'SCI':>7} "
        f"{'Size':>5} "
        f"{'Sens':>6} "
        f"{'Low':>6} "
        f"{'High':>6} "
        f"{'New':>6} "
        f"{'Raw':>5} "
        f"{'Keep':>5} "
        f"{'Keep%':>6} "
        f"{'Trk':>5} "
        f"{'Feed':>5} "
        f"{'RawMed':>7} "
        f"{'NormMed':>8} "
        f"{'Norm ms':>8}"
    )

    print("-" * 110)

    for frame_number, path in enumerate(
        paths,
        1,
    ):
        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(
                f"Could not read {path}"
            )

        result = pipeline.process(
            frame_number,
            image,
            visual(image),
        )

        raw_min, raw_med, raw_max = (
            score_stats(
                result["raw_detections"]
            )
        )

        norm_min, norm_med, norm_max = (
            score_stats(
                result[
                    "normalized_detections"
                ]
            )
        )

        norm_ms = float(
            result["normalizer_ms"]
        )

        norm_times.append(norm_ms)

        params = result["params"]

        generic = result["generic_controls"]

        raw_count = len(
            result["raw_detections"]
        )

        keep_count = len(
            result["detections"]
        )

        keep_percent = (
            100.0 * keep_count / raw_count
            if raw_count
            else 0.0
        )

        print(
            f"{frame_number:3d} "
            f"{params['sci']:7.4f} "
            f"{params['size']:5d} "
            f"{generic.sensitivity:6.3f} "
            f"{generic.low_threshold:6.3f} "
            f"{generic.high_threshold:6.3f} "
            f"{generic.new_track_threshold:6.3f} "
            f"{raw_count:5d} "
            f"{keep_count:5d} "
            f"{keep_percent:6.1f} "
            f"{len(result['tracks']):5d} "
            f"{len(result['feedback_detections']):5d} "
            f"{raw_med:7.3f} "
            f"{norm_med:8.3f} "
            f"{norm_ms:8.4f}"
        )

    norm_times = np.asarray(
        norm_times,
        dtype=float,
    )

    print()
    print("Normalizer timing")
    print("-----------------")
    print(
        "Mean:",
        f"{norm_times.mean():.4f} ms",
    )
    print(
        "P95 :",
        f"{np.percentile(norm_times,95):.4f} ms",
    )
    print(
        "Max :",
        f"{norm_times.max():.4f} ms",
    )


def main():
    sequence = os.environ.get(
        "ACMOT_SEQUENCE"
    )

    if not sequence:
        raise RuntimeError(
            "ACMOT_SEQUENCE is not set"
        )

    sequence = Path(sequence)

    num_frames = int(
        os.environ.get(
            "ONLINE_FRAMES",
            "20",
        )
    )

    yolov8_weights = Path(
        "/Users/ahmedgouda/Desktop/"
        "CUE_SELECTION/yolov8n.pt"
    )

    rtdetr_weights = Path(
        os.path.expanduser(
            "~/Desktop/Universal-ACMOT/"
            "rtdetr-l.pt"
        )
    )

    print()
    print("Universal AC-MOT Online Normalization")
    print("Sequence :", sequence)
    print("Frames   :", num_frames)

    run_detector(
        "YOLOv8",
        YOLOv8Adapter(
            str(yolov8_weights)
        ),
        sequence,
        num_frames,
    )

    run_detector(
        "RT-DETR",
        RTDETRAdapter(
            str(rtdetr_weights)
        ),
        sequence,
        num_frames,
    )


if __name__ == "__main__":
    main()
