"""Merge/status checker for the 3-worker shared AC-MOT workflow.

This script does not recompute metrics. The three workers already write their
stage outputs into the same shared result folder. This checker verifies that the
expected artifacts are present, records a compact merged manifest, and makes it
obvious whether the distributed validation workflow is complete.

Run on any one worker/runtime after the three-worker workflow finishes.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

DRIVE = Path(os.environ.get("ACMOT_DRIVE", "/content/drive/MyDrive"))
SHARED_ROOT = Path(
    os.environ.get(
        "ACMOT_SHARED_RESULT_ROOT",
        str(DRIVE / "AC-MOT-shared" / "defensible_acmot_3workers"),
    )
)
SMOKE = os.environ.get("ACMOT_SMOKE_TEST", "0") == "1"

if not SHARED_ROOT.is_dir():
    raise RuntimeError(f"Shared result folder not found: {SHARED_ROOT}")

if SMOKE:
    expected = {
        "operating_search_space": SHARED_ROOT / "SCIENTIFIC_SEARCH_SPACE_SMOKE.json",
        "operating_marker": SHARED_ROOT / "WORKER1_OPERATING_SMOKE_DONE.json",
        "temporal_csv": SHARED_ROOT / "TEMPORAL_ABLATION_SMOKE.csv",
        "temporal_marker": SHARED_ROOT / "WORKER2_TEMPORAL_SMOKE_DONE.json",
        "joint_trials": SHARED_ROOT / "EMPIRICAL_OPTUNA_TRIALS_SMOKE.csv",
        "joint_marker": SHARED_ROOT / "WORKER3_JOINT_SMOKE_DONE.json",
    }
else:
    expected = {
        "resolution_sweep": SHARED_ROOT / "OPERATING_RESOLUTION_SWEEP.csv",
        "confidence_sweep": SHARED_ROOT / "OPERATING_CONFIDENCE_SWEEP.csv",
        "nms_sweep": SHARED_ROOT / "OPERATING_NMS_SWEEP.csv",
        "operating_search_space": SHARED_ROOT / "SCIENTIFIC_SEARCH_SPACE.json",
        "operating_report": SHARED_ROOT / "OPERATING_ABLATION_REPORT.json",
        "operating_marker": SHARED_ROOT / "WORKER1_OPERATING_DONE.json",
        "temporal_csv": SHARED_ROOT / "TEMPORAL_ABLATION_FULL.csv",
        "temporal_config": SHARED_ROOT / "FROZEN_TEMPORAL_CONFIG.json",
        "temporal_marker": SHARED_ROOT / "WORKER2_TEMPORAL_DONE.json",
        "joint_trials": SHARED_ROOT / "EMPIRICAL_OPTUNA_TRIALS.csv",
        "joint_db": SHARED_ROOT / "EMPIRICAL_OPTUNA.db",
        "cue_calibration": SHARED_ROOT / "DETECTOR_DERIVED_CUE_CALIBRATION.json",
        "frozen_config": SHARED_ROOT / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json",
        "joint_marker": SHARED_ROOT / "WORKER3_JOINT_DONE.json",
    }

status = {name: path.exists() for name, path in expected.items()}
missing = [name for name, ok in status.items() if not ok]

manifest = {
    "mode": "smoke" if SMOKE else "full_validation",
    "shared_result_root": str(SHARED_ROOT),
    "timestamp_unix": time.time(),
    "complete": not missing,
    "status": status,
    "files": {name: str(path) for name, path in expected.items()},
    "missing": missing,
    "test_dev_used": False,
    "note": (
        "This manifest combines stage completion from the 3-worker validation workflow. "
        "It does not authorize or run the held-out final test."
    ),
}

out = SHARED_ROOT / ("MERGED_3WORKER_SMOKE_STATUS.json" if SMOKE else "MERGED_3WORKER_FULL_STATUS.json")
out.write_text(json.dumps(manifest, indent=2))

print("=" * 100)
print("AC-MOT 3-WORKER MERGE / STATUS")
print("Shared root:", SHARED_ROOT)
print("Mode       :", manifest["mode"])
print("Complete   :", manifest["complete"])
if missing:
    print("Missing    :", ", ".join(missing))
else:
    print("All expected validation artifacts are present.")
print("Manifest   :", out)
print("Test-dev   : NOT ACCESSED")
print("=" * 100)
