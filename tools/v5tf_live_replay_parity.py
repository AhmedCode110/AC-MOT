#!/usr/bin/env python3
"""Verify the chosen V5-TF F3 live motion path equals cached-cue replay.

This is a behaviour/fidelity check on development data only. It computes no
GT-based quality metric. The deterministic subset is the first two sequences
of TRAIN_SPLIT_V5 development-40 and the first 20 frames of each, for both
development detectors. The output is new and refuses to overwrite.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from adapters.trackers.bytetrack import ByteTrackAdapter
from run_universal_acmot import build_config
from tools.run_policy_validation import CachedDetector, make_tracker
from tools.v5tf_dev import FAMILIES
from universal_acmot import UniversalACMOT
from universal_policy_pipeline import POLICIES, UniversalPolicyPipeline, replace


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(
    "/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@gmail.com/"
    "My Drive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-train"
)
CACHE = ROOT / "outputs" / "det_cache_train_res"
OUT = ROOT / "outputs" / "v5tf_dev" / "live_replay_parity_v1.json"
DETECTORS = ("yolov8", "rtdetr")
SEQUENCE_COUNT = 2
FRAME_COUNT = 20


def track_rows(tracks):
    return [
        [t.track_id, t.x1, t.y1, t.x2, t.y2, t.confidence, t.class_id]
        for t in tracks
    ]


def rows_equal(left, right) -> bool:
    if len(left) != len(right):
        return False
    if not left:
        return True
    return bool(np.allclose(np.asarray(left, dtype=float),
                            np.asarray(right, dtype=float),
                            rtol=0.0, atol=1e-9, equal_nan=True))


def audit_equal(left: dict, right: dict) -> tuple[bool, list[str]]:
    keys = (
        "resolution", "raw_count", "accepted_after_topk", "tracks",
        "low_threshold", "high_threshold", "new_track_threshold",
        "tf_motion_ratio", "tf_otsu_t1", "tf_otsu_t2", "tf_otsu_eta",
    )
    mismatches = []
    for key in keys:
        lv, rv = left.get(key), right.get(key)
        if isinstance(lv, (float, np.floating)) or isinstance(rv, (float, np.floating)):
            if not np.isclose(lv, rv, rtol=0.0, atol=1e-12, equal_nan=True):
                mismatches.append(key)
        elif lv != rv:
            mismatches.append(key)
    return not mismatches, mismatches


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing parity artifact: {OUT}")

    split = json.loads((ROOT / "research" / "TRAIN_SPLIT_V5.json").read_text())
    sequences = split["development"][:SEQUENCE_COUNT]
    policy = replace(POLICIES["V1"], **FAMILIES["F3"])
    cfg = build_config()
    cases = []
    all_pass = True

    for detector in DETECTORS:
        for sequence in sequences:
            cache_path = CACHE / detector / f"{sequence}.npz"
            live_detector = CachedDetector(cache_path)
            replay_detector = CachedDetector(cache_path)
            live = UniversalACMOT(
                live_detector, ByteTrackAdapter(), policy=policy,
                density_kwargs={},
            )
            replay = UniversalPolicyPipeline(
                cfg, replay_detector, make_tracker(cfg, policy), policy,
                density_kwargs={},
            )
            frame_paths = sorted((DATASET / "sequences" / sequence).glob("*.jpg"))
            tested = min(FRAME_COUNT, len(frame_paths), live_detector.frames)
            case_mismatches = []

            for frame_number, frame_path in enumerate(frame_paths[:tested], start=1):
                image = cv2.imread(str(frame_path))
                if image is None:
                    raise RuntimeError(f"Could not read {frame_path}")

                live_detector.frame = frame_number
                live_tracks = live(image)
                live_result = live.last

                replay_detector.frame = frame_number
                replay_result = replay.process(
                    frame_number, image, replay_detector.visual_dict(frame_number)
                )

                live_rows = track_rows(live_tracks)
                replay_rows = track_rows(replay_result["tracks"])
                same_tracks = rows_equal(live_rows, replay_rows)
                same_audit, audit_mismatches = audit_equal(
                    live_result["audit"], replay_result["audit"]
                )
                if not same_tracks or not same_audit:
                    case_mismatches.append({
                        "frame": frame_number,
                        "tracks_equal": same_tracks,
                        "audit_mismatches": audit_mismatches,
                        "live_track_count": len(live_rows),
                        "replay_track_count": len(replay_rows),
                    })

            passed = not case_mismatches
            all_pass = all_pass and passed
            cases.append({
                "detector": detector,
                "sequence": sequence,
                "frames": tested,
                "passed": passed,
                "mismatches": case_mismatches,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "check": "V5-TF F3 live image_stats motion vs cached visual-cue replay",
        "data_role": "VisDrone2019-MOT-train development-40 only",
        "quality_metrics_computed": False,
        "policy_family": "F3",
        "sequences": sequences,
        "frames_per_sequence": FRAME_COUNT,
        "detectors": list(DETECTORS),
        "passed": all_pass,
        "cases": cases,
    }, indent=2) + "\n")
    print(json.dumps({"passed": all_pass, "artifact": str(OUT), "cases": cases}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
