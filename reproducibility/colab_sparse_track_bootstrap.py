#!/usr/bin/env python3
"""Strict AC-MOT + SparseTrack Colab bootstrap.

Fresh Colab usage:

!curl -fsSL \
  https://raw.githubusercontent.com/AhmedCode110/AC-MOT/main/reproducibility/colab_sparse_track_bootstrap.py \
  -o /content/colab_sparse_track_bootstrap.py

%run /content/colab_sparse_track_bootstrap.py

This script restores and verifies the saved environment. It does NOT run
tracking, training, validation tuning, or MOT17 test evaluation.
"""

from pathlib import Path
from datetime import datetime, timezone
from packaging.version import Version
import importlib.metadata as md
import subprocess
import runpy
import hashlib
import json
import os
import sys

EXPECTED = {
    "python": "3.13.15",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "cuda": "12.8",
    "gpu_contains": "Tesla T4",
    "numpy": "2.1.3",
    "cv2": "5.0.0",
    "lap": "0.5.13",
    "sparse_track_commit": "499844f32c5bb2332f9811f26cd70cf4e517d4e7",
    "detectron2_commit": "a2f4a8771ab77e8411c26b27f24f9489a28a2453",
    "freeze_manifest_sha256": "5356d8270c2616bb12af52af56c56392062daf644cd51de8c4c200a9209083d7",
}

CRITICAL_PACKAGES = {
    "cloudpickle": "3.1.2",
    "cython-bbox": "0.1.5",
    "Cython": "3.0.12",
    "filterpy": "1.4.5",
    "fvcore": "0.1.5.post20221221",
    "h5py": "3.16.0",
    "hydra-core": "1.3.7",
    "iopath": "0.1.9",
    "lap": "0.5.13",
    "motmetrics": "1.4.0",
    "ninja": "1.13.2",
    "numpy": "2.1.3",
    "omegaconf": "2.3.1",
    "Pillow": "11.3.0",
    "portalocker": "4.3.2",
    "pycocotools": "2.0.11",
    "scikit-image": "0.25.2",
    "scipy": "1.16.3",
    "tabulate": "0.9.0",
    "termcolor": "3.3.0",
    "thop": "0.1.1.post2209072238",
    "tqdm": "4.67.3",
    "xmltodict": "1.0.4",
    "yacs": "0.1.8",
}


def fail(message):
    raise RuntimeError("\nSTRICT REPRODUCIBILITY FAILURE:\n" + message)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


print("=" * 96)
print("AC-MOT + SparseTrack — STRICT COLAB BOOTSTRAP")
print("=" * 96)

# 1) Mount Drive if required.
DRIVE = Path("/content/drive")
if not (DRIVE / "MyDrive").is_dir():
    try:
        from google.colab import drive
    except Exception as exc:
        fail(f"Google Drive is not mounted and google.colab.drive is unavailable: {exc}")
    print("\nMounting Google Drive...")
    drive.mount("/content/drive")

PROJECT = Path("/content/drive/MyDrive/AC-MOT-SparseTrack-NEW")
if not PROJECT.is_dir():
    fail(
        "Project is missing at /content/drive/MyDrive/AC-MOT-SparseTrack-NEW. "
        "Add the shared project as a MyDrive shortcut or copy it into MyDrive."
    )

PROJECT_REAL = PROJECT.resolve()
print("\nProject logical path:", PROJECT)
print("Project resolved path:", PROJECT_REAL)

# 2) Base runtime must match BEFORE restore.
os.environ["PYTHONHASHSEED"] = "0"  # for child processes
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torchvision

py = sys.version.split()[0]
if py != EXPECTED["python"]:
    fail(f"Python mismatch: {py} != {EXPECTED['python']}")
if torch.__version__ != EXPECTED["torch"]:
    fail(f"Torch mismatch: {torch.__version__} != {EXPECTED['torch']}")
if torchvision.__version__ != EXPECTED["torchvision"]:
    fail(f"torchvision mismatch: {torchvision.__version__} != {EXPECTED['torchvision']}")
if torch.version.cuda != EXPECTED["cuda"]:
    fail(f"CUDA mismatch: {torch.version.cuda} != {EXPECTED['cuda']}")
if not torch.cuda.is_available():
    fail("CUDA is unavailable.")

gpu = torch.cuda.get_device_name(0)
if EXPECTED["gpu_contains"].lower() not in gpu.lower():
    fail(f"Expected Tesla T4, got {gpu!r}")

print("\nBase runtime: PASS")
print(" Python     :", py)
print(" Torch      :", torch.__version__)
print(" torchvision:", torchvision.__version__)
print(" CUDA       :", torch.version.cuda)
print(" GPU        :", gpu)

# 3) Restore exact saved environment from Drive.
RESTORE = PROJECT / "runtime" / "restore_latest_env.py"
POINTER = PROJECT / "runtime" / "LATEST_ENV_BUNDLE.txt"
if not RESTORE.is_file():
    fail(f"Missing restore entrypoint: {RESTORE}")
if not POINTER.is_file():
    fail(f"Missing environment pointer: {POINTER}")

bundle_path = Path(POINTER.read_text().strip())
if not bundle_path.is_dir():
    fail(f"Environment bundle is missing: {bundle_path}")

print("\nEnvironment bundle:", bundle_path)
print("Running authoritative Drive restore...")
runpy.run_path(str(RESTORE), run_name="__main__")

# 4) Verify critical versions after restore.
import numpy as np
import cv2
import lap

if np.__version__ != EXPECTED["numpy"]:
    fail(f"NumPy mismatch: {np.__version__} != {EXPECTED['numpy']}")
if cv2.__version__ != EXPECTED["cv2"]:
    fail(f"cv2 mismatch: {cv2.__version__} != {EXPECTED['cv2']}")
if lap.__version__ != EXPECTED["lap"]:
    fail(f"lap mismatch: {lap.__version__} != {EXPECTED['lap']}")

package_report = {}
for package, expected in CRITICAL_PACKAGES.items():
    actual = md.version(package)
    package_report[package] = actual
    if package == "thop":
        ok = Version(actual) == Version(expected)
    else:
        ok = actual == expected
    if not ok:
        fail(f"Package mismatch: {package} {actual} != {expected}")

print("\nCritical Python packages: PASS")

# 5) Runtime paths used by scientific child processes.
RUNTIME = PROJECT / "runtime"
REPO = PROJECT / "SparseTrack"
D2 = Path("/content/detectron2_acmot_pinned")
OVERLAY = RUNTIME / "overlay"
OVERLAY_TRACKER = OVERLAY / "tracker"
BOOST_LIB = RUNTIME / "boost_python" / "lib"

for p in [REPO, D2, OVERLAY, OVERLAY_TRACKER, BOOST_LIB]:
    if not p.exists():
        fail(f"Required restored path is missing: {p}")

python_paths = [str(OVERLAY_TRACKER), str(OVERLAY), str(REPO), str(D2)]
os.environ["PYTHONPATH"] = ":".join(python_paths + [os.environ.get("PYTHONPATH", "")])
os.environ["LD_LIBRARY_PATH"] = ":".join([str(BOOST_LIB), os.environ.get("LD_LIBRARY_PATH", "")])

for p in reversed(python_paths):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

# 6) Source integrity.
st_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
st_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
d2_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=D2, text=True).strip()

if st_head != EXPECTED["sparse_track_commit"]:
    fail(f"SparseTrack HEAD mismatch: {st_head}")
if st_status:
    fail("SparseTrack pinned source tree is dirty.")
if d2_head != EXPECTED["detectron2_commit"]:
    fail(f"Detectron2 HEAD mismatch: {d2_head}")

print("\nSource integrity: PASS")
print(" SparseTrack:", st_head)
print(" Tree       : CLEAN")
print(" Detectron2 :", d2_head)

# 7) Import gate, including SparseTrack datasets shadowing fix.
import detectron2
import detectron2._C
import pbcvt
from tracker.sparse_tracker import SparseTracker
from datasets import builder as sparse_builder
from datasets.mot_mapper import MOTtestMapper

builder_file = Path(sparse_builder.__file__).resolve()
mapper_file = Path(sys.modules[MOTtestMapper.__module__].__file__).resolve()

if PROJECT_REAL not in builder_file.parents:
    fail(f"datasets.builder resolved outside SparseTrack: {builder_file}")
if PROJECT_REAL not in mapper_file.parents:
    fail(f"datasets.mot_mapper resolved outside SparseTrack: {mapper_file}")
if not hasattr(pbcvt, "GMC"):
    fail("pbcvt imported but GMC is missing.")

print("\nImport gate: PASS")
print(" Detectron2      :", detectron2.__file__)
print(" pbcvt           :", pbcvt.__file__)
print(" SparseTracker   : PASS")
print(" datasets.builder:", builder_file)
print(" datasets.mapper :", mapper_file)

# 8) Current-process determinism flags.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

if torch.backends.cudnn.benchmark:
    fail("cudnn.benchmark unexpectedly True")
if not torch.backends.cudnn.deterministic:
    fail("cudnn.deterministic unexpectedly False")
if not torch.are_deterministic_algorithms_enabled():
    fail("Deterministic algorithms not enabled")

print("\nDeterminism flags: PASS")

# 9) Verify exact frozen controller archive.
FROZEN = PROJECT / "adaptive" / "FROZEN_ADAPTIVE_EDGE_V1_20260917T095152Z"
MANIFEST = FROZEN / "FROZEN_CONTROLLER_MANIFEST.json"
LOCK = FROZEN / "FROZEN.lock"

if not FROZEN.is_dir() or not MANIFEST.is_file() or not LOCK.is_file():
    fail(f"Frozen controller archive is missing or incomplete: {FROZEN}")

manifest_sha = sha256_file(MANIFEST)
if manifest_sha != EXPECTED["freeze_manifest_sha256"]:
    fail(
        "Frozen manifest SHA mismatch: "
        f"{manifest_sha} != {EXPECTED['freeze_manifest_sha256']}"
    )

print("\nFrozen controller: PASS")
print(" Path        :", FROZEN)
print(" Manifest SHA:", manifest_sha)

# 10) Persistent audit record.
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
AUDIT = PROJECT / "reproducibility" / "bootstrap_logs" / stamp
AUDIT.mkdir(parents=True, exist_ok=False)

report = {
    "status": "PASS",
    "timestamp_utc": stamp,
    "project_logical_path": str(PROJECT),
    "project_resolved_path": str(PROJECT_REAL),
    "environment_bundle": str(bundle_path),
    "python": py,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda": torch.version.cuda,
    "gpu": gpu,
    "numpy": np.__version__,
    "opencv_python": cv2.__version__,
    "lap": lap.__version__,
    "critical_packages": package_report,
    "sparse_track_commit": st_head,
    "sparse_track_clean": True,
    "detectron2_commit": d2_head,
    "frozen_controller": str(FROZEN),
    "frozen_manifest_sha256": manifest_sha,
    "PYTHONHASHSEED_for_children": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "cudnn_deterministic": torch.backends.cudnn.deterministic,
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "rules": {
        "training": False,
        "mot17_test_tuning_allowed": False,
        "validation_retuning_after_freeze_allowed": False,
        "wall_clock_diagnostic_fps_is_paper_fps": False,
    },
}

(AUDIT / "environment_verification.json").write_text(json.dumps(report, indent=2))
(AUDIT / "BOOTSTRAP_PASS.txt").write_text("AC-MOT + SparseTrack bootstrap: PASS\n")
(AUDIT / "critical_packages_snapshot.txt").write_text(
    "\n".join(f"{k}=={v}" for k, v in sorted(package_report.items())) + "\n"
)

print("\n" + "=" * 96)
print("AC-MOT + SparseTrack bootstrap: PASS")
print("=" * 96)
print("Audit record:", AUDIT)
print("Frozen controller:", FROZEN)
print()
print("Scientific child-process environment:")
print(" PYTHONHASHSEED=0")
print(" CUBLAS_WORKSPACE_CONFIG=:4096:8")
print(" PYTHONPATH=", os.environ["PYTHONPATH"])
print(" LD_LIBRARY_PATH starts with:", BOOST_LIB)
print()
print("Validation tuning is CLOSED.")
print("Do not tune on MOT17 test.")
print("Do not report wall-clock diagnostic FPS as paper FPS.")
print("Use a unique Drive output path for every new scientific run.")
