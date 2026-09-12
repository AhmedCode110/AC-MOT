#!/usr/bin/env python3
"""Colab bootstrap launcher for the frozen V2 Trial22 test-dev technical rerun.

This launcher only prepares the pinned runtime dependencies, downloads the already
frozen technical-rerun script from its immutable commit, and runs it. It does not
change any AC-MOT parameters, candidate selection, or test protocol.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

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

print("=" * 88, flush=True)
print("AC-MOT V2 TRIAL22 COLAB BOOTSTRAP", flush=True)
print("=" * 88, flush=True)
print("Installing pinned runtime dependencies...", flush=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES],
    check=True,
)

print("Downloading frozen technical-rerun script...", flush=True)
urllib.request.urlretrieve(RUNNER_URL, RUNNER_PATH)

print("Runner commit:", RUNNER_COMMIT, flush=True)
print("Runner path  :", RUNNER_PATH, flush=True)
print("Launching frozen V2 Trial22 technical rerun...", flush=True)
print("=" * 88, flush=True)

subprocess.run([sys.executable, str(RUNNER_PATH)], check=True)
