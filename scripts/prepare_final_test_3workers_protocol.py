"""Prepare the immutable three-worker held-out final-test protocol.

This script does NOT access the test set. Run it only after:
1) Worker 1/2/3 validation stages are complete,
2) the frozen optimized configuration exists, and
3) the new A0-A3 validation component ablation is complete.

It freezes the exact three systems that will later be evaluated on test-dev:
- Baseline_Default
- Old_ACMOT_Frozen
- New_ACMOT_Frozen
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("ACMOT_REPO", "/content/AC-MOT"))
DRIVE = Path(os.environ.get("ACMOT_DRIVE", "/content/drive/MyDrive"))
RESULT_ROOT = Path(
    os.environ.get(
        "ACMOT_RESULT_ROOT",
        str(DRIVE / "AC-MOT-shared" / "defensible_acmot_3workers"),
    )
)

FROZEN = RESULT_ROOT / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"
CALIBRATION = RESULT_ROOT / "DETECTOR_DERIVED_CUE_CALIBRATION.json"
ABLATION_DONE = RESULT_ROOT / "NEW_ACMOT_COMPONENT_ABLATION_DONE.json"
FINAL_LOCK = RESULT_ROOT / "FINAL_TEST_DONE.json"
PLAN = RESULT_ROOT / "FINAL_TEST_3WORKER_PROTOCOL.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if FINAL_LOCK.exists():
    raise RuntimeError("FINAL_TEST_DONE.json already exists; held-out test has already been exposed.")
for p in [FROZEN, CALIBRATION, ABLATION_DONE]:
    if not p.exists():
        raise RuntimeError(f"Missing required validation artifact: {p}")

if PLAN.exists():
    print("FINAL TEST PROTOCOL ALREADY FROZEN")
    print(PLAN.read_text())
    raise SystemExit(0)

try:
    repo_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    repo_commit = "unknown"

frozen = json.loads(FROZEN.read_text())
plan = {
    "protocol_version": "acmot_final_test_3worker_v1",
    "purpose": "One locked held-out comparison of project baseline, historical AC-MOT, and new frozen AC-MOT.",
    "test_access_during_preparation": False,
    "selection_or_tuning_on_test": False,
    "official_visdrone": False,
    "evaluation_protocol": "custom class-agnostic AC-MOT research protocol",
    "repository_commit_at_freeze": repo_commit,
    "artifact_hashes": {
        "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json": sha256(FROZEN),
        "DETECTOR_DERIVED_CUE_CALIBRATION.json": sha256(CALIBRATION),
        "NEW_ACMOT_COMPONENT_ABLATION_DONE.json": sha256(ABLATION_DONE),
    },
    "ground_truth_filter": {
        "categories": [1, 4, 5, 6, 9],
        "score": 1,
        "occlusion_lt": 2,
        "truncation_lt": 2,
    },
    "minimum_processing_fps": 25.0,
    "required_gpu_class": "NVIDIA T4",
    "pinned_trackeval_commit": "12c8791b303e0a0b50f753af204249e622d0281a",
    "systems": [
        {
            "worker_id": 1,
            "name": "Baseline_Default",
            "description": "Static project baseline: conf=0.25, NMS IoU=0.45, imgsz=640, default ByteTrack profile.",
            "tracker_profile": "default",
            "controller": "static_project_baseline",
        },
        {
            "worker_id": 2,
            "name": "Old_ACMOT_Frozen",
            "description": "Historical heuristic AC-MOT controller with tuned ByteTrack, W=7, stride=10, original adaptive confidence/NMS/resolution mapping.",
            "tracker_profile": "tuned",
            "controller": "core_v17.PresentationController",
            "smoothing_window": 7,
            "analysis_stride": 10,
            "resolution_levels": [640, 736, 832],
        },
        {
            "worker_id": 3,
            "name": "New_ACMOT_Frozen",
            "description": "Validation-selected frozen empirical SCI controller with tuned ByteTrack; no test-time tuning.",
            "tracker_profile": "tuned",
            "controller": "FrozenSCIController",
            "smoothing_window": int(frozen["smoothing_window"]),
            "analysis_stride": int(frozen["analysis_stride"]),
            "resolution_levels": [int(x) for x in frozen["resolution_levels"]],
        },
    ],
    "merge_rule": "Merge all three predeclared systems; do not select or retune based on partial test results.",
    "warning": "After any worker accesses test-dev, do not change parameters and later call another run an unbiased held-out final test.",
}

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
PLAN.write_text(json.dumps(plan, indent=2))
print("FINAL TEST 3-WORKER PROTOCOL FROZEN")
print("Plan:", PLAN)
print(json.dumps(plan, indent=2))
print("Test-dev has NOT been accessed by this preparation script.")
