"""AC-MOT V2 Colab launcher using the canonical shared Drive layout.

This wrapper is intended for any Colab account where the same Drive shortcut
layout is present under MyDrive/AC-MOT-shared.

Project storage policy:
- search/read project inputs under AC-MOT-shared only
- treat defensible_acmot_3workers as frozen V1 read-only evidence
- write new V2 artifacts under AC-MOT-shared/V2_MULTI_OBJECTIVE_MOTA_IDS only
- never select test-dev for V2 optimization or smoke testing

Runtime safety policy:
- the V1-compatible Python packages must be installed BEFORE launching this file
- after changing NumPy/SciPy/Ultralytics in Colab, restart the runtime first
- verify the loaded NumPy ufunc ABI before Optuna starts
- execute one real YOLOv8n FP16 validation-frame inference before Optuna starts
- stop the Optuna study immediately if any trial records an internal exception
"""

from __future__ import annotations

import importlib.metadata as metadata
import os
import runpy
import traceback
from pathlib import Path

SHARED_ROOT = Path("/content/drive/MyDrive/AC-MOT-shared")
DATA_ROOT = SHARED_ROOT / "AC-MOT-data"
VAL_DIR = DATA_ROOT / "VisDrone2019-MOT-val"
V1_ROOT = SHARED_ROOT / "defensible_acmot_3workers"
DEFAULT_V2_ROOT = SHARED_ROOT / "V2_MULTI_OBJECTIVE_MOTA_IDS"
WEIGHTS = Path("/content/weights/yolov8n.pt")

os.environ.setdefault("ACMOT_REPO", "/content/AC-MOT")
os.environ.setdefault("ACMOT_DATA_ROOT", str(DATA_ROOT))
os.environ.setdefault("ACMOT_VAL_DIR", str(VAL_DIR))
os.environ.setdefault("ACMOT_V1_RESULT_ROOT", str(V1_ROOT))
os.environ.setdefault("ACMOT_V2_RESULT_ROOT", str(DEFAULT_V2_ROOT))

required = {
    "shared project root": SHARED_ROOT,
    "validation dataset": VAL_DIR,
    "V1 result root": V1_ROOT,
    "scientific search space": V1_ROOT / "SCIENTIFIC_SEARCH_SPACE.json",
    "frozen temporal config": V1_ROOT / "FROZEN_TEMPORAL_CONFIG.json",
    "detector-derived cue calibration": V1_ROOT / "DETECTOR_DERIVED_CUE_CALIBRATION.json",
    "frozen V1 config": V1_ROOT / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json",
}
missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
if missing:
    raise RuntimeError(
        "Canonical AC-MOT shared Drive layout is incomplete in this account:\n- "
        + "\n- ".join(missing)
    )

if not (VAL_DIR / "sequences").is_dir() or not (VAL_DIR / "annotations").is_dir():
    raise RuntimeError(f"Invalid VisDrone validation dataset structure: {VAL_DIR}")

shared_resolved = SHARED_ROOT.resolve()
v1_resolved = V1_ROOT.resolve()
v2_resolved = Path(os.environ["ACMOT_V2_RESULT_ROOT"]).resolve()
if shared_resolved != v2_resolved and shared_resolved not in v2_resolved.parents:
    raise RuntimeError(
        f"Unsafe V2 output root: {v2_resolved}. All AC-MOT writes must stay under {shared_resolved}."
    )
if v2_resolved == v1_resolved or v1_resolved in v2_resolved.parents:
    raise RuntimeError(
        f"Unsafe V2 output root: {v2_resolved}. Frozen V1 root {v1_resolved} is read-only."
    )

print("[AC-MOT V2] Canonical shared Drive layout detected")
print("[AC-MOT V2] SHARED_ROOT =", SHARED_ROOT)
print("[AC-MOT V2] DATA_ROOT   =", DATA_ROOT)
print("[AC-MOT V2] VAL_DIR     =", VAL_DIR)
print("[AC-MOT V2] V1_ROOT     =", V1_ROOT, "(READ ONLY)")
print("[AC-MOT V2] V2_ROOT     =", os.environ["ACMOT_V2_RESULT_ROOT"])
print("[AC-MOT V2] SEARCH      = AC-MOT-shared ONLY")
print("[AC-MOT V2] WRITES      = AC-MOT-shared ONLY")
print("[AC-MOT V2] TEST-DEV    = NOT SELECTED")

# ---------------------------------------------------------------------------
# Runtime preflight.  The original V1 optimizer pinned these versions.  Colab
# kernels can keep an already-imported NumPy object alive after pip replaces the
# package on disk, so package installation and the scientific run must not be
# performed across an unrestarted ABI-changing kernel session.
# ---------------------------------------------------------------------------
EXPECTED_EXACT = {
    "ultralytics": "8.3.200",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
}

installed = {}
missing_packages = []
for package, expected in EXPECTED_EXACT.items():
    try:
        installed[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        installed[package] = "NOT INSTALLED"
        missing_packages.append(package)

try:
    installed["optuna"] = metadata.version("optuna")
except metadata.PackageNotFoundError:
    installed["optuna"] = "NOT INSTALLED"
    missing_packages.append("optuna")

print("[PREFLIGHT] Package versions:")
for package, version in installed.items():
    print(f"[PREFLIGHT]   {package}={version}")

wrong_exact = {
    package: (installed[package], expected)
    for package, expected in EXPECTED_EXACT.items()
    if installed[package] != expected
}
wrong_optuna = installed["optuna"] == "NOT INSTALLED" or not installed["optuna"].startswith("4.")
if missing_packages or wrong_exact or wrong_optuna:
    raise RuntimeError(
        "V2 runtime packages are not the frozen V1-compatible versions. STOP before Optuna.\n"
        "Run this in a separate Colab cell:\n"
        "  !pip install -q ultralytics==8.3.200 numpy==2.2.6 scipy==1.15.3 "
        "lap opencv-python-headless 'optuna>=4,<5' pandas matplotlib\n"
        "Then use Runtime -> Restart session, remount Drive, and run the launcher again.\n"
        f"Observed: {installed}"
    )

import cv2
import numpy as np
import optuna
import torch
from ultralytics import YOLO

print("[PREFLIGHT] Loaded NumPy =", np.__version__)
print("[PREFLIGHT] Python NumPy path =", np.__file__)

# NumPy 2.2 ufuncs support an instance __dict__.  If this is false while pip
# reports NumPy 2.2.6, the live Colab kernel is stale and must be restarted.
if not hasattr(np.add, "__dict__"):
    raise RuntimeError(
        "STALE COLAB NUMPY KERNEL DETECTED: pip reports NumPy 2.2.6, but the loaded "
        "numpy.ufunc ABI has no __dict__. Restart the Colab session before running V2."
    )
try:
    np.add.__dict__["_acmot_v2_preflight"] = True
    del np.add.__dict__["_acmot_v2_preflight"]
except Exception as exc:
    raise RuntimeError(
        "Loaded NumPy ufunc objects are not writable as required by the pinned runtime. "
        "Restart the Colab session after installing the pinned packages."
    ) from exc

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required for V2.")
gpu_name = torch.cuda.get_device_name(0)
print("[PREFLIGHT] GPU =", gpu_name)
if "T4" not in gpu_name:
    raise RuntimeError(f"V2 controlled FPS comparison requires NVIDIA T4; got {gpu_name}")

# One real inference before creating/resuming any Optuna study.  This catches
# dependency/ABI failures before 50 trial records can be polluted.
WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
if not WEIGHTS.exists():
    old_cwd = os.getcwd()
    os.chdir(WEIGHTS.parent)
    try:
        YOLO("yolov8n.pt")
    finally:
        os.chdir(old_cwd)
if not WEIGHTS.exists():
    raise RuntimeError(f"Could not obtain fixed YOLOv8n weights: {WEIGHTS}")

sequence_dirs = sorted(p for p in (VAL_DIR / "sequences").iterdir() if p.is_dir())
if not sequence_dirs:
    raise RuntimeError("No validation sequences found for V2 preflight.")
first_frames = sorted(sequence_dirs[0].glob("*.jpg"))
if not first_frames:
    raise RuntimeError(f"No JPEG frames found in {sequence_dirs[0]}")
preflight_frame = first_frames[0]
img = cv2.imread(str(preflight_frame))
if img is None:
    raise RuntimeError(f"Could not decode V2 preflight frame: {preflight_frame}")

print("[PREFLIGHT] Running one YOLOv8n FP16 inference before Optuna...")
try:
    model = YOLO(str(WEIGHTS))
    result = model.predict(
        source=img,
        conf=0.25,
        iou=0.35,
        imgsz=512,
        classes=[0, 2, 5, 7],
        max_det=1000,
        device=0,
        half=True,
        verbose=False,
    )[0]
    print(f"[PREFLIGHT] YOLO inference PASS | detections={len(result.boxes)}")
except Exception:
    print("[PREFLIGHT] YOLO inference FAILED. Full traceback follows:")
    traceback.print_exc()
    raise RuntimeError("V2 preflight inference failed; Optuna was NOT started.")
finally:
    if "model" in locals():
        del model
    torch.cuda.empty_cache()

# Import the frozen tracker path as an additional dependency check before Optuna.
try:
    from types import SimpleNamespace
    from ultralytics.trackers.byte_tracker import BYTETracker

    repo_root = Path(os.environ["ACMOT_REPO"])
    if str(repo_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(repo_root))
    from core_v17 import tracker_settings

    tracker_cfg = tracker_settings("tuned")
    _tracker = BYTETracker(
        SimpleNamespace(
            track_high_thresh=tracker_cfg["high"],
            track_low_thresh=tracker_cfg["low"],
            new_track_thresh=tracker_cfg["new"],
            track_buffer=tracker_cfg["buffer"],
            match_thresh=tracker_cfg["match"],
            fuse_score=tracker_cfg["fuse"],
        ),
        frame_rate=30,
    )
    del _tracker
    print("[PREFLIGHT] ByteTrack import/init PASS")
except Exception:
    print("[PREFLIGHT] ByteTrack preflight FAILED. Full traceback follows:")
    traceback.print_exc()
    raise RuntimeError("V2 tracker preflight failed; Optuna was NOT started.")

# The optimizer records an internal exception in trial.user_attrs['error'] and
# returns sentinel objective values.  Add a fail-fast callback so one failed
# trial stops the study instead of silently consuming the whole 50-trial budget.
_original_optimize = optuna.study.Study.optimize


def _optimize_fail_fast(self, func, *args, callbacks=None, **kwargs):
    callback_list = list(callbacks or [])

    def _stop_on_internal_error(study, trial):
        error = trial.user_attrs.get("error")
        if error:
            print(f"[FAIL-FAST] Trial {trial.number} recorded an internal exception:")
            print(error)
            raise RuntimeError(
                f"V2 trial {trial.number} failed internally. Study stopped immediately; "
                "do not treat sentinel objective values as scientific results."
            )

    callback_list.append(_stop_on_internal_error)
    return _original_optimize(self, func, *args, callbacks=callback_list, **kwargs)


optuna.study.Study.optimize = _optimize_fail_fast

print("[PREFLIGHT] ALL CHECKS PASSED")
print("[PREFLIGHT] Optuna may start safely")

runpy.run_path(
    "/content/AC-MOT/scripts/optuna_sci_v2_multiobjective_validation.py",
    run_name="__main__",
)
