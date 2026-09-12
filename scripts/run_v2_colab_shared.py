"""AC-MOT V2 Colab launcher using the canonical shared Drive layout.

This wrapper is intended for any Colab account where the same Drive shortcut
layout is present under MyDrive/AC-MOT-shared.

It never selects test-dev. The validation-only V2 optimizer remains responsible
for its own additional held-out-path rejection checks.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path

SHARED_ROOT = Path("/content/drive/MyDrive/AC-MOT-shared")
DATA_ROOT = SHARED_ROOT / "AC-MOT-data"
VAL_DIR = DATA_ROOT / "VisDrone2019-MOT-val"
V1_ROOT = SHARED_ROOT / "defensible_acmot_3workers"
DEFAULT_V2_ROOT = Path("/content/drive/MyDrive/AC-MOT-results/V2_MULTI_OBJECTIVE_MOTA_IDS")

os.environ.setdefault("ACMOT_REPO", "/content/AC-MOT")
os.environ.setdefault("ACMOT_DATA_ROOT", str(DATA_ROOT))
os.environ.setdefault("ACMOT_VAL_DIR", str(VAL_DIR))
os.environ.setdefault("ACMOT_V1_RESULT_ROOT", str(V1_ROOT))
os.environ.setdefault("ACMOT_V2_RESULT_ROOT", str(DEFAULT_V2_ROOT))

required = {
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

print("[AC-MOT V2] Canonical shared Drive layout detected")
print("[AC-MOT V2] DATA_ROOT =", DATA_ROOT)
print("[AC-MOT V2] VAL_DIR   =", VAL_DIR)
print("[AC-MOT V2] V1_ROOT   =", V1_ROOT)
print("[AC-MOT V2] V2_ROOT   =", os.environ["ACMOT_V2_RESULT_ROOT"])
print("[AC-MOT V2] TEST-DEV  = NOT SELECTED")

runpy.run_path(
    "/content/AC-MOT/scripts/optuna_sci_v2_multiobjective_validation.py",
    run_name="__main__",
)
