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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family", default="auto")

    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser()
    seq_root = dataset / "sequences"

    if not seq_root.is_dir():
        raise RuntimeError(
            f"Missing sequences folder: {seq_root}"
        )

    sequences = sorted(
        p for p in seq_root.iterdir()
        if p.is_dir()
    )

    output_root = Path(args.output_dir)
    tracks_root = output_root / "tracks"

    tracks_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = create_detector(
        args.weights,
        family=args.family,
    )

    print("=" * 72)
    print("Universal AC-MOT — FULL VALIDATION")
    print("=" * 72)
    print("Dataset   :", dataset)
    print("Sequences :", len(sequences))
    print("Adapter   :", detector.__class__.__name__)
    print()

    summary_rows = []

    for seq_index, sequence in enumerate(
        sequences,
        start=1,
    ):
        print(
            f"[{seq_index}/{len(sequences)}] "
            f"{sequence.name}"
        )

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

        output_file = (
            tracks_root
            / f"{sequence.name}.txt"
        )

        sci_values = []
        keep_values = []
        track_values = []
        resolutions = []

        processed = 0

        with output_file.open("w") as f:
            for frame_number, frame_path in enumerate(
                frames,
                start=1,
            ):
                image = cv2.imread(
                    str(frame_path)
                )

                if image is None:
                    continue

                result = pipeline.process(
                    frame_number,
                    image,
                    analyze_visual(image),
                )

                raw_count = len(
                    result["raw_detections"]
                )

                keep_count = len(
                    result["detections"]
                )

                keep_pct = (
                    100.0 * keep_count / raw_count
                    if raw_count
                    else 0.0
                )

                sci_values.append(
                    float(
                        result["params"]["sci"]
                    )
                )

                keep_values.append(
                    keep_pct
                )

                track_values.append(
                    len(result["tracks"])
                )

                resolutions.append(
                    int(
                        result["params"]["size"]
                    )
                )

                for track in result["tracks"]:
                    x = float(track.x1)
                    y = float(track.y1)

                    w = max(
                        0.0,
                        float(track.x2)
                        - float(track.x1),
                    )

                    h = max(
                        0.0,
                        float(track.y2)
                        - float(track.y1),
                    )

                    # MOT/VisDrone-style output:
                    # frame,id,x,y,w,h,score,class,-1,-1
                    f.write(
                        f"{frame_number},"
                        f"{int(track.track_id)},"
                        f"{x:.3f},"
                        f"{y:.3f},"
                        f"{w:.3f},"
                        f"{h:.3f},"
                        f"{float(track.confidence):.6f},"
                        f"{int(track.class_id)},"
                        f"-1,-1\n"
                    )

                processed += 1

                if (
                    frame_number % 500 == 0
                    or frame_number == len(frames)
                ):
                    print(
                        f"  {frame_number}/"
                        f"{len(frames)} frames"
                    )

        row = {
            "sequence": sequence.name,
            "frames": processed,
            "mean_sci": (
                float(np.mean(sci_values))
                if sci_values else 0.0
            ),
            "mean_keep_pct": (
                float(np.mean(keep_values))
                if keep_values else 0.0
            ),
            "mean_tracks": (
                float(np.mean(track_values))
                if track_values else 0.0
            ),
            "resolutions": "/".join(
                str(v)
                for v in sorted(
                    set(resolutions)
                )
            ),
        }

        summary_rows.append(row)

        print(
            f"  DONE | "
            f"frames={row['frames']} "
            f"SCI={row['mean_sci']:.3f} "
            f"keep={row['mean_keep_pct']:.1f}% "
            f"tracks={row['mean_tracks']:.1f} "
            f"res={row['resolutions']}"
        )

    summary_path = (
        output_root / "summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "frames",
                "mean_sci",
                "mean_keep_pct",
                "mean_tracks",
                "resolutions",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 72)
    print("FULL VALIDATION COMPLETE")
    print("=" * 72)
    print("Sequences :", len(summary_rows))
    print("Tracks    :", tracks_root)
    print("Summary   :", summary_path)


if __name__ == "__main__":
    main()
