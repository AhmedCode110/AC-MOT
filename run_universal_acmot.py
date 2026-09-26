from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from core import Config
from adapters.detectors.factory import create_detector
from adapters.trackers.bytetrack import ByteTrackAdapter
from universal_online_pipeline import UniversalOnlineACMOTPipeline


def analyze_visual(frame):
    small = cv2.resize(
        frame,
        None,
        fx=0.25,
        fy=0.25,
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY,
    )

    return {
        "edges": float(
            cv2.Canny(
                gray, 50, 120
            ).mean() / 255.0
        ),
        "brightness": float(
            gray.mean()
        ),
        "blur": float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        ),
    }


def build_config():
    return Config(
        name="universal_online",
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


def build_tracker(cfg):
    return ByteTrackAdapter(
        high=cfg.high,
        low=cfg.low,
        new=cfg.new,
        buffer=cfg.buffer,
        match=cfg.match,
        fuse=cfg.fuse,
    )


def image_paths(folder):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    paths = [
        p
        for p in Path(folder).iterdir()
        if p.suffix.lower() in extensions
    ]

    return sorted(paths)


def draw_tracks(frame, tracks):
    out = frame.copy()

    for track in tracks:
        x1 = int(track.x1)
        y1 = int(track.y1)
        x2 = int(track.x2)
        y2 = int(track.y2)

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"ID {int(track.track_id)} "
            f"{float(track.confidence):.2f}"
        )

        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return out


def make_writer(path, frame, fps):
    if not path:
        return None

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    h, w = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        fps,
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open output video: {path}"
        )

    return writer


def run_image_folder(
    source,
    pipeline,
    output,
    max_frames,
):
    paths = image_paths(source)

    if not paths:
        raise RuntimeError(
            f"No images found in {source}"
        )

    fps = 30.0
    writer = None

    timings = []
    norm_timings = []

    for frame_number, path in enumerate(
        paths,
        start=1,
    ):
        if (
            max_frames > 0
            and frame_number > max_frames
        ):
            break

        frame = cv2.imread(
            str(path)
        )

        if frame is None:
            continue

        t0 = perf_counter()

        result = pipeline.process(
            frame_number,
            frame,
            analyze_visual(frame),
        )

        total_ms = (
            perf_counter() - t0
        ) * 1000.0

        timings.append(total_ms)
        norm_timings.append(
            result["normalizer_ms"]
        )

        if output:
            annotated = draw_tracks(
                frame,
                result["tracks"],
            )

            if writer is None:
                writer = make_writer(
                    output,
                    annotated,
                    fps,
                )

            writer.write(
                annotated
            )

        print_status(
            frame_number,
            result,
            total_ms,
        )

    if writer is not None:
        writer.release()

    print_summary(
        timings,
        norm_timings,
    )


def run_video(
    source,
    pipeline,
    output,
    max_frames,
):
    cap = cv2.VideoCapture(
        str(source)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {source}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if fps <= 0:
        fps = 30.0

    writer = None

    timings = []
    norm_timings = []

    frame_number = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_number += 1

        if (
            max_frames > 0
            and frame_number > max_frames
        ):
            break

        t0 = perf_counter()

        result = pipeline.process(
            frame_number,
            frame,
            analyze_visual(frame),
        )

        total_ms = (
            perf_counter() - t0
        ) * 1000.0

        timings.append(total_ms)
        norm_timings.append(
            result["normalizer_ms"]
        )

        if output:
            annotated = draw_tracks(
                frame,
                result["tracks"],
            )

            if writer is None:
                writer = make_writer(
                    output,
                    annotated,
                    fps,
                )

            writer.write(
                annotated
            )

        print_status(
            frame_number,
            result,
            total_ms,
        )

    cap.release()

    if writer is not None:
        writer.release()

    print_summary(
        timings,
        norm_timings,
    )


def print_status(
    frame_number,
    result,
    total_ms,
):
    # Print every frame for first five,
    # then every ten frames.
    if (
        frame_number > 5
        and frame_number % 10 != 0
    ):
        return

    params = result["params"]
    generic = result[
        "generic_controls"
    ]

    raw = len(
        result["raw_detections"]
    )

    keep = len(
        result["detections"]
    )

    tracks = len(
        result["tracks"]
    )

    keep_pct = (
        100.0 * keep / raw
        if raw
        else 0.0
    )

    print(
        f"Frame {frame_number:5d} | "
        f"SCI={params['sci']:.3f} | "
        f"size={params['size']:4d} | "
        f"sens={generic.sensitivity:.3f} | "
        f"raw={raw:4d} | "
        f"keep={keep:4d} "
        f"({keep_pct:5.1f}%) | "
        f"tracks={tracks:4d} | "
        f"norm={result['normalizer_ms']:.3f} ms | "
        f"total={total_ms:.2f} ms"
    )


def print_summary(
    timings,
    norm_timings,
):
    if not timings:
        print("No frames processed.")
        return

    times = np.asarray(
        timings,
        dtype=float,
    )

    norm = np.asarray(
        norm_timings,
        dtype=float,
    )

    print()
    print("=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)

    print(
        f"Frames              : {len(times)}"
    )

    print(
        f"Pipeline mean       : "
        f"{times.mean():.3f} ms"
    )

    print(
        f"Pipeline P95        : "
        f"{np.percentile(times, 95):.3f} ms"
    )

    print(
        f"Development FPS     : "
        f"{1000.0 / times.mean():.2f}"
    )

    print(
        f"Normalizer mean     : "
        f"{norm.mean():.3f} ms"
    )

    print(
        f"Normalizer P95      : "
        f"{np.percentile(norm, 95):.3f} ms"
    )

    overhead = (
        100.0
        * norm.mean()
        / times.mean()
    )

    print(
        f"Normalizer overhead : "
        f"{overhead:.2f}%"
    )

    print()
    print(
        "NOTE: Mac runtime is for development only."
    )
    print(
        "Official real-time benchmark must be run on T4."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Universal AC-MOT with automatic "
            "detector selection and online "
            "score normalization."
        )
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Detector weights",
    )

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Video file or image-sequence folder"
        ),
    )

    parser.add_argument(
        "--family",
        default="auto",
        choices=[
            "auto",
            "yolo",
            "rtdetr",
        ],
        help=(
            "Detector family. "
            "Default: auto"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional annotated MP4 output"
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "0 = process all frames"
        ),
    )

    args = parser.parse_args()

    weights = str(
        Path(args.weights)
        .expanduser()
    )

    source = Path(
        args.source
    ).expanduser()

    if not source.exists():
        raise FileNotFoundError(
            f"Source not found: {source}"
        )

    print()
    print("=" * 70)
    print("Universal AC-MOT")
    print("=" * 70)

    detector = create_detector(
        weights,
        family=args.family,
    )

    cfg = build_config()

    tracker = build_tracker(
        cfg
    )

    pipeline = (
        UniversalOnlineACMOTPipeline(
            config=cfg,
            detector=detector,
            tracker=tracker,
            raw_confidence_floor=0.01,
        )
    )

    print(
        "Weights :",
        weights,
    )

    print(
        "Adapter :",
        detector.__class__.__name__,
    )

    print(
        "Source  :",
        source,
    )

    print(
        "Mode    : Online normalization"
    )

    print()

    if source.is_dir():
        run_image_folder(
            source,
            pipeline,
            args.output,
            args.max_frames,
        )
    else:
        run_video(
            source,
            pipeline,
            args.output,
            args.max_frames,
        )


if __name__ == "__main__":
    main()
