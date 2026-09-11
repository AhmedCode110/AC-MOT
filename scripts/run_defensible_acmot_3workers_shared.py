"""Distributed 3-worker launcher for the defensible AC-MOT pipeline.

IMPORTANT
---------
This file does NOT replace or modify the original one-command full pipeline.
It orchestrates the validation stages across three independent workers/runtimes
that can see the same shared result folder.

Worker assignment
-----------------
Worker 1 -> Stage 1: resolution / confidence / NMS operating-point ablation
Worker 2 -> Stage 2: temporal ablation
Worker 3 -> Stage 3: empirical joint Optuna, after Stage 1 and Stage 2 finish

Workers 1 and 2 can run in parallel because Stage 2 does not consume Stage 1.
Worker 3 waits for the prerequisite output files in the shared folder, then runs
exactly the existing joint optimizer.

Scientific behavior is preserved:
- fixed pretrained YOLOv8n
- fixed tuned ByteTrack during SCI study
- validation only for stages 1-3
- test-dev is NOT accessed here
- full mode keeps the existing single-study 50-trial Optuna behavior
- Stage 1 uses the timing-fair launcher: exact tested resolution warm-up and
  unused visual SceneAnalyzer computation excluded from static-screen timing

Provider note
-------------
Use multiple runtimes/accounts only where permitted by the compute provider's
terms. This script is a generic multi-worker orchestration layer, not a quota-
circumvention mechanism.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("ACMOT_REPO", "/content/AC-MOT"))
DRIVE = Path(os.environ.get("ACMOT_DRIVE", "/content/drive/MyDrive"))
WORKER_ID = int(os.environ.get("ACMOT_WORKER_ID", "0"))
SMOKE = os.environ.get("ACMOT_SMOKE_TEST", "0") == "1"
TRIALS = int(os.environ.get("ACMOT_OPTUNA_TRIALS", "50"))
POLL_SECONDS = int(os.environ.get("ACMOT_POLL_SECONDS", "60"))

SHARED_ROOT = Path(
    os.environ.get(
        "ACMOT_SHARED_RESULT_ROOT",
        str(DRIVE / "AC-MOT-shared" / "defensible_acmot_3workers"),
    )
)

# Worker 1 intentionally uses the timing-fair Stage-1 launcher.  The launcher
# executes the original scientific Stage-1 sweep unchanged while fixing only
# FPS isolation: exact tested-shape warm-up and no unused visual SCI analysis.
OPERATING = ROOT / "scripts" / "scientific_operating_ablation_fair_timing.py"
TEMPORAL = ROOT / "scripts" / "temporal_ablation_portable_colab.py"
JOINT = ROOT / "scripts" / "optuna_sci_empirical_portable_colab.py"

if WORKER_ID not in {1, 2, 3}:
    raise RuntimeError("Set ACMOT_WORKER_ID to 1, 2, or 3 before running this script.")
if not DRIVE.is_dir():
    raise RuntimeError("Google Drive is not mounted. Run drive.mount('/content/drive') first.")
if not ROOT.is_dir():
    raise RuntimeError(f"Repository not found at {ROOT}. Clone/pull AhmedCode110/AC-MOT first.")
for p in (OPERATING, TEMPORAL, JOINT):
    if not p.exists():
        raise RuntimeError(f"Missing required script: {p}")

SHARED_ROOT.mkdir(parents=True, exist_ok=True)


def run_script(script: Path, env: dict):
    print("=" * 110, flush=True)
    print("RUNNING", script.name, flush=True)
    print("Shared result root:", SHARED_ROOT, flush=True)
    print("=" * 110, flush=True)
    subprocess.run([sys.executable, "-u", str(script)], cwd=str(ROOT), env=env, check=True)


def write_marker(name: str, payload: dict):
    path = SHARED_ROOT / name
    payload = dict(payload)
    payload["worker_id"] = WORKER_ID
    payload["smoke"] = SMOKE
    payload["timestamp_unix"] = time.time()
    path.write_text(json.dumps(payload, indent=2))
    print("[MARKER]", path, flush=True)


def wait_for(paths: list[Path]):
    print("[WAIT] Worker 3 prerequisites:", flush=True)
    for p in paths:
        print("  -", p, flush=True)
    while True:
        missing = [p for p in paths if not p.exists()]
        if not missing:
            print("[WAIT] All prerequisites are available.", flush=True)
            return
        print("[WAIT] Still missing:", ", ".join(p.name for p in missing), flush=True)
        time.sleep(POLL_SECONDS)


def ensure_worker3_validation_dataset(env: dict):
    """Worker 3 has its own runtime/Drive, so ensure validation data is locally accessible."""
    data_root = Path(env.get("ACMOT_DATA_ROOT", str(DRIVE / "AC-MOT-data")))
    val_dir = Path(env.get("ACMOT_VAL_DIR", str(data_root / "VisDrone2019-MOT-val")))

    def valid(path: Path) -> bool:
        return path.is_dir() and (path / "sequences").is_dir() and (path / "annotations").is_dir()

    candidates = [
        val_dir,
        DRIVE / "VisDrone2019-MOT-val",
        DRIVE / "VisDrone2019-MOT" / "VisDrone2019-MOT-val",
        data_root / "VisDrone2019-MOT-val",
    ]
    for p in candidates:
        if valid(p):
            env["ACMOT_VAL_DIR"] = str(p)
            return

    print("[DATA] Worker 3 validation set missing; downloading portable validation copy...", flush=True)
    import urllib.request
    import zipfile

    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / "VisDrone2019-MOT-val.zip"
    url = "https://huggingface.co/datasets/vanthanh/VisDrone2019-MOT/resolve/main/VisDrone2019-MOT-val.zip"
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(data_root)
    archive.unlink(missing_ok=True)

    found = [
        p for p in data_root.rglob("*")
        if valid(p) and "VisDrone2019-MOT-val" in p.name
    ]
    if len(found) == 1:
        env["ACMOT_VAL_DIR"] = str(found[0])
    elif valid(data_root / "VisDrone2019-MOT-val"):
        env["ACMOT_VAL_DIR"] = str(data_root / "VisDrone2019-MOT-val")
    else:
        raise RuntimeError("Worker 3 could not identify the validation dataset after extraction.")


env = os.environ.copy()
env["ACMOT_RESULT_ROOT"] = str(SHARED_ROOT)
env.setdefault("ACMOT_PROGRESS_EVERY", "50")

print("\n" + "#" * 110, flush=True)
print("AC-MOT 3-WORKER SHARED-RESULT RUNNER", flush=True)
print("Worker          :", WORKER_ID, flush=True)
print("Mode            :", "SMOKE" if SMOKE else "FULL", flush=True)
print("Shared folder   :", SHARED_ROOT, flush=True)
print("Test-dev        : NOT ACCESSED", flush=True)
if WORKER_ID == 1:
    print("Stage-1 timing  : FAIR (exact-shape warm-up; unused visual analysis excluded)", flush=True)
print("#" * 110 + "\n", flush=True)

if WORKER_ID == 1:
    env["ACMOT_CALIBRATION_SMOKE"] = "1" if SMOKE else "0"
    run_script(OPERATING, env)
    expected = SHARED_ROOT / ("SCIENTIFIC_SEARCH_SPACE_SMOKE.json" if SMOKE else "SCIENTIFIC_SEARCH_SPACE.json")
    if not expected.exists():
        raise RuntimeError(f"Worker 1 finished but expected output is missing: {expected}")
    write_marker(
        "WORKER1_OPERATING_SMOKE_DONE.json" if SMOKE else "WORKER1_OPERATING_DONE.json",
        {"stage": "operating_ablation", "expected_output": str(expected), "timing_fair": True},
    )

elif WORKER_ID == 2:
    env["ACMOT_TEMPORAL_SMOKE"] = "1" if SMOKE else "0"
    run_script(TEMPORAL, env)
    if SMOKE:
        expected = SHARED_ROOT / "TEMPORAL_ABLATION_SMOKE.csv"
        marker = "WORKER2_TEMPORAL_SMOKE_DONE.json"
    else:
        expected = SHARED_ROOT / "FROZEN_TEMPORAL_CONFIG.json"
        marker = "WORKER2_TEMPORAL_DONE.json"
    if not expected.exists():
        raise RuntimeError(f"Worker 2 finished but expected output is missing: {expected}")
    write_marker(marker, {"stage": "temporal_ablation", "expected_output": str(expected)})

else:
    if SMOKE:
        prereqs = [
            SHARED_ROOT / "SCIENTIFIC_SEARCH_SPACE_SMOKE.json",
            SHARED_ROOT / "WORKER2_TEMPORAL_SMOKE_DONE.json",
        ]
    else:
        prereqs = [
            SHARED_ROOT / "SCIENTIFIC_SEARCH_SPACE.json",
            SHARED_ROOT / "FROZEN_TEMPORAL_CONFIG.json",
        ]
    wait_for(prereqs)
    ensure_worker3_validation_dataset(env)
    env["ACMOT_SMOKE_TEST"] = "1" if SMOKE else "0"
    env["ACMOT_OPTUNA_TRIALS"] = str(TRIALS)
    run_script(JOINT, env)

    if SMOKE:
        expected = SHARED_ROOT / "EMPIRICAL_OPTUNA_TRIALS_SMOKE.csv"
        marker = "WORKER3_JOINT_SMOKE_DONE.json"
    else:
        expected = SHARED_ROOT / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"
        marker = "WORKER3_JOINT_DONE.json"
    if not expected.exists():
        raise RuntimeError(f"Worker 3 finished but expected output is missing: {expected}")
    write_marker(marker, {"stage": "joint_optuna", "expected_output": str(expected), "target_trials": TRIALS})

print("\n[DONE] Worker", WORKER_ID, "completed successfully.", flush=True)
print("Shared results:", SHARED_ROOT, flush=True)
