"""One-command portable runner for the defensible AC-MOT validation workflow.

Order
-----
1) Scientific operating-point ablations
   - resolution screen
   - confidence screen
   - NMS IoU screen
2) Temporal ablation
   - smoothing window
   - analysis stride
3) Empirical joint Optuna
   - SCI weights
   - direction/magnitude of confidence adaptation
   - direction/magnitude of NMS adaptation
   - SCI resolution switching thresholds

Everything above uses validation only. Test-dev is deliberately NOT accessed.
The final held-out test is a separate explicit action after configuration freeze.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("ACMOT_REPO", "/content/AC-MOT"))
DRIVE = Path(os.environ.get("ACMOT_DRIVE", "/content/drive/MyDrive"))
RESULT_ROOT = Path(
    os.environ.get("ACMOT_RESULT_ROOT", str(DRIVE / "AC-MOT-results" / "defensible_acmot"))
)
SMOKE = os.environ.get("ACMOT_SMOKE_TEST", "0") == "1"
TRIALS = int(os.environ.get("ACMOT_OPTUNA_TRIALS", "50"))

OPERATING = ROOT / "scripts" / "scientific_operating_ablation_portable_colab.py"
TEMPORAL = ROOT / "scripts" / "temporal_ablation_portable_colab.py"
JOINT = ROOT / "scripts" / "optuna_sci_empirical_portable_colab.py"

if not DRIVE.is_dir():
    raise RuntimeError(
        "Google Drive is not mounted. Run in Colab first:\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')"
    )
for p in (OPERATING, TEMPORAL, JOINT):
    if not p.exists():
        raise RuntimeError(f"Missing {p}. Update repo: git -C /content/AC-MOT pull --ff-only")

RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def run_stage(number: int, total: int, title: str, script: Path, env: dict):
    print("\n" + "=" * 110, flush=True)
    print(f"STAGE {number}/{total} — {title}", flush=True)
    print("=" * 110, flush=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, "-u", str(script)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    elapsed = (time.perf_counter() - started) / 60.0
    if code != 0:
        raise RuntimeError(f"Stage {number} failed with return code {code}")
    print(f"[STAGE {number} COMPLETE] elapsed={elapsed:.1f} min", flush=True)


env = os.environ.copy()
env["ACMOT_RESULT_ROOT"] = str(RESULT_ROOT)
env.setdefault("ACMOT_PROGRESS_EVERY", "50")

env1 = env.copy()
env1["ACMOT_CALIBRATION_SMOKE"] = "1" if SMOKE else "0"
run_stage(1, 3, "SCIENTIFIC RESOLUTION / CONFIDENCE / NMS ABLATIONS", OPERATING, env1)

env2 = env.copy()
env2["ACMOT_TEMPORAL_SMOKE"] = "1" if SMOKE else "0"
run_stage(2, 3, "TEMPORAL ABLATION — SMOOTHING WINDOW + ANALYSIS STRIDE", TEMPORAL, env2)

env3 = env.copy()
env3["ACMOT_SMOKE_TEST"] = "1" if SMOKE else "0"
env3["ACMOT_OPTUNA_TRIALS"] = str(TRIALS)
run_stage(3, 3, "EMPIRICAL JOINT SCI OPTUNA", JOINT, env3)

print("\n" + "=" * 110, flush=True)
print("AC-MOT DEFENSIBLE VALIDATION PIPELINE FINISHED", flush=True)
print("Results folder :", RESULT_ROOT, flush=True)
print("Validation     : USED", flush=True)
print("Test-dev       : NOT ACCESSED", flush=True)
if SMOKE:
    print("Mode           : SMOKE — no final scientific configuration was frozen", flush=True)
else:
    print("Frozen config  :", RESULT_ROOT / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json", flush=True)
    print("NEXT STEP      : review the ablation tables + frozen config before running the held-out test once.", flush=True)
print("=" * 110, flush=True)
