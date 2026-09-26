from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from adapters.detectors.factory import create_detector
from adapters.trackers.bytetrack import ByteTrackAdapter
from universal_online_pipeline import UniversalOnlineACMOTPipeline
from run_universal_acmot import analyze_visual, build_config


def make_tracker(cfg):
    return ByteTrackAdapter(
        high=cfg.high,
        low=cfg.low,
        new=cfg.new,
        buffer=cfg.buffer,
        match=cfg.match,
        fuse=cfg.fuse,
    )


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def run_sequence(
    detector,
    sequence: Path,
    max_frames: int,
):
    cfg = build_config()

    pipeline = UniversalOnlineACMOTPipeline(
        config=cfg,
        detector=detector,
        tracker=make_tracker(cfg),
        raw_confidence_floor=0.01,
    )

    frames = sorted(
        sequence.glob("*.jpg")
    )

    if max_frames > 0:
        frames = frames[:max_frames]

    keep_percentages = []
    track_counts = []
    sci_values = []
    norm_times = []
    resolutions = []

    processed = 0

    for frame_number, path in enumerate(
        frames,
        start=1,
    ):
        image = cv2.imread(str(path))

        if image is None:
            continue

        result = pipeline.process(
            frame_number,
            image,
            analyze_visual(image),
        )

        raw = len(
            result["raw_detections"]
        )

        keep = len(
            result["detections"]
        )

        keep_pct = (
            100.0 * keep / raw
            if raw
            else 0.0
        )

        keep_percentages.append(
            keep_pct
        )

        track_counts.append(
            len(result["tracks"])
        )

        sci_values.append(
            float(result["params"]["sci"])
        )

        resolutions.append(
            int(result["params"]["size"])
        )

        norm_times.append(
            float(result["normalizer_ms"])
        )

        processed += 1

    return {
        "sequence": sequence.name,
        "frames": processed,
        "mean_sci": float(
            np.mean(sci_values)
        ) if sci_values else 0.0,
        "mean_keep_pct": float(
            np.mean(keep_percentages)
        ) if keep_percentages else 0.0,
        "mean_tracks": float(
            np.mean(track_counts)
        ) if track_counts else 0.0,
        "normalizer_mean_ms": float(
            np.mean(norm_times)
        ) if norm_times else 0.0,
        "normalizer_p95_ms": percentile(
            norm_times,
            95,
        ),
        "resolutions": "/".join(
            str(x)
            for x in sorted(
                set(resolutions)
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--family",
        default="auto",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    dataset = Path(
        args.dataset
    ).expanduser()

    sequence_root = (
        dataset / "sequences"
    )

    if not sequence_root.is_dir():
        raise RuntimeError(
            f"Missing sequences folder: "
            f"{sequence_root}"
        )

    sequences = sorted(
        p
        for p in sequence_root.iterdir()
        if p.is_dir()
    )

    print("=" * 70)
    print("Universal AC-MOT Validation Smoke")
    print("=" * 70)
    print("Dataset   :", dataset)
    print("Sequences :", len(sequences))
    print("Frames/seq:", args.max_frames)
    print()

    detector = create_detector(
        args.weights,
        family=args.family,
    )

    print(
        "Adapter   :",
        detector.__class__.__name__,
    )
    print()

    rows = []

    for index, sequence in enumerate(
        sequences,
        start=1,
    ):
        print(
            f"[{index}/{len(sequences)}] "
            f"{sequence.name}"
        )

        row = run_sequence(
            detector,
            sequence,
            args.max_frames,
        )

        rows.append(row)

        print(
            f"  frames={row['frames']} "
            f"SCI={row['mean_sci']:.3f} "
            f"keep={row['mean_keep_pct']:.1f}% "
            f"tracks={row['mean_tracks']:.1f} "
            f"norm={row['normalizer_mean_ms']:.3f} ms "
            f"res={row['resolutions']}"
        )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sequence",
        "frames",
        "mean_sci",
        "mean_keep_pct",
        "mean_tracks",
        "normalizer_mean_ms",
        "normalizer_p95_ms",
        "resolutions",
    ]

    with output.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    all_norm = [
        r["normalizer_mean_ms"]
        for r in rows
    ]

    all_keep = [
        r["mean_keep_pct"]
        for r in rows
    ]

    print()
    print("=" * 70)
    print("VALIDATION SMOKE SUMMARY")
    print("=" * 70)

    print(
        "Sequences              :",
        len(rows),
    )

    print(
        "Mean Keep%             :",
        f"{np.mean(all_keep):.2f}%"
    )

    print(
        "Mean Normalizer latency:",
        f"{np.mean(all_norm):.3f} ms"
    )

    print(
        "Saved                  :",
        output,
    )


if __name__ == "__main__":
    main()
