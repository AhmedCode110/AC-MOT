#!/usr/bin/env python3
"""One-shot Colab bootstrap for the frozen V2 Trial22 test-dev technical rerun.

The launcher prepares only the execution environment:
- installs the pinned Python runtime dependencies;
- recreates /content/AC-MOT at the frozen evaluation commit;
- recreates /content/TrackEval at the pinned TrackEval commit;
- downloads the immutable technical-rerun script and executes it.

It does not tune, reselect, or alter any AC-MOT parameter or test protocol.
Google Drive must already be mounted by the notebook.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path("/content/AC-MOT")
TRACKEVAL = Path("/content/TrackEval")
DRIVE = Path("/content/drive/MyDrive")

REPO_URL = "https://github.com/AhmedCode110/AC-MOT.git"
REPO_COMMIT = "5becc52a569f271ee8b73ce47c67d495de0d64a5"
TRACKEVAL_URL = "https://github.com/JonathonLuiten/TrackEval.git"
TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"

RUNNER_COMMIT = "e4104ee3f2ee066a6d4e3a827e0ac9c902de528d"
RUNNER_URL = (
    "https://raw.githubusercontent.com/AhmedCode110/AC-MOT/"
    f"{RUNNER_COMMIT}/scripts/run_v2_trial22_testdev_technical_rerun.py"
)
RUNNER_PATH = Path("/content/run_v2_trial22_testdev_technical_rerun.py")

PINNED_PACKAGES = [
    "ultralytics==8.3.200",
    "numpy==2.2.6",
    "scipy==1.15.3",
    "lap",
    "opencv-python-headless",
    "optuna>=4,<5",
    "pandas",
    "matplotlib",
]


def sh(args):
    subprocess.run(args, check=True)


def fresh_checkout(url: str, path: Path, commit: str) -> None:
    if path.exists():
        shutil.rmtree(path)
    sh(["git", "clone", "-q", url, str(path)])
    sh(["git", "-C", str(path), "checkout", "-q", commit])
    head = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != commit:
        raise RuntimeError(f"Checkout mismatch for {path}: {head} != {commit}")


print("=" * 88, flush=True)
print("AC-MOT V2 TRIAL22 COLAB BOOTSTRAP", flush=True)
print("=" * 88, flush=True)

if not DRIVE.is_dir():
    raise RuntimeError(
        "Google Drive is not mounted. Run drive.mount('/content/drive') in Colab first."
    )

print("[1/4] Installing pinned runtime dependencies...", flush=True)
sh([sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES])

print("[2/4] Preparing frozen AC-MOT repository...", flush=True)
fresh_checkout(REPO_URL, REPO, REPO_COMMIT)
print("      AC-MOT HEAD:", REPO_COMMIT, flush=True)

print("[3/4] Preparing pinned TrackEval...", flush=True)
fresh_checkout(TRACKEVAL_URL, TRACKEVAL, TRACKEVAL_COMMIT)
print("      TrackEval HEAD:", TRACKEVAL_COMMIT, flush=True)

print("[4/4] Downloading immutable technical-rerun script...", flush=True)
urllib.request.urlretrieve(RUNNER_URL, RUNNER_PATH)
print("      Runner commit:", RUNNER_COMMIT, flush=True)
print("      Runner path  :", RUNNER_PATH, flush=True)

print("=" * 88, flush=True)
print("Launching frozen V2 Trial22 technical rerun...", flush=True)
print("=" * 88, flush=True)

sh([sys.executable, str(RUNNER_PATH)])
