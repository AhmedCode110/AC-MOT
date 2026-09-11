"""Portable temporal ablation for AC-MOT SCI smoothing window and analysis stride.

Scientific purpose
------------------
This script evaluates the temporal design independently before joint Optuna tuning.
It answers two supervisor questions with validation evidence rather than manual choice:

1) Why analyze one frame every N frames?
2) Why smooth SCI over W measurements?

The detector, old SCI formulation/mapping, and tuned ByteTrack stay fixed. Only:
- smoothing_window
- analysis_stride

are varied on VisDrone2019-MOT-val. The old presentation setting W=7, stride=10
is included as the reference. A feasible candidate must satisfy:
- FPS >= ACMOT_MIN_FPS (default 25)
- IDS <= the W=7, stride=10 reference IDS

Among feasible candidates, selection is:
1. highest MOTA
2. lower IDS
3. higher HOTA
4. higher IDF1
5. higher FPS

The full grid is deliberately discrete and interpretable:
- smoothing windows: 1, 3, 5, 7, 9
- analysis strides: 1, 5, 10, 15, 20 frames

At the repository's nominal 30-FPS tracker clock, those strides correspond to
approximately 0.033, 0.167, 0.333, 0.500, and 0.667 seconds between scene analyses.
The window grid spans no smoothing through progressively longer causal memory.
No single value is assumed optimal a priori; the validation ablation determines it.

Smoke mode evaluates only four combinations and never writes a frozen selection.
This script never accesses test-dev.

Important protocol note
-----------------------
Evaluation remains the repository's custom class-agnostic AC-MOT TrackEval protocol,
not the official VisDrone leaderboard preprocessing protocol.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def sh(args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


# ---------------------------------------------------------------------------
# Portable paths / controls
# ---------------------------------------------------------------------------
ROOT = Path(os.environ.get("ACMOT_REPO", "/content/AC-MOT"))
DRIVE = Path(os.environ.get("ACMOT_DRIVE", "/content/drive/MyDrive"))
DATA_ROOT = Path(os.environ.get("ACMOT_DATA_ROOT", str(DRIVE / "AC-MOT-data")))
RESULT_ROOT = Path(
    os.environ.get("ACMOT_RESULT_ROOT", str(DRIVE / "AC-MOT-results" / "optuna_sci_joint"))
)
VAL_DIR = Path(os.environ.get("ACMOT_VAL_DIR", str(DATA_ROOT / "VisDrone2019-MOT-val")))
WEIGHTS = Path(os.environ.get("ACMOT_WEIGHTS", "/content/weights/yolov8n.pt"))
TRACKEVAL_DIR = Path(os.environ.get("ACMOT_TRACKEVAL", "/content/TrackEval"))

MIN_FPS = float(os.environ.get("ACMOT_MIN_FPS", "25"))
PROGRESS_EVERY = int(os.environ.get("ACMOT_PROGRESS_EVERY", "50"))
TEMPORAL_SMOKE = os.environ.get("ACMOT_TEMPORAL_SMOKE", "0") == "1"

FULL_WINDOWS = [1, 3, 5, 7, 9]
FULL_STRIDES = [1, 5, 10, 15, 20]
SMOKE_WINDOWS = [1, 7]
SMOKE_STRIDES = [1, 10]
REFERENCE_WINDOW = 7
REFERENCE_STRIDE = 10
NOMINAL_FPS = 30.0

if not DRIVE.is_dir():
    raise RuntimeError(
        "Google Drive is not mounted. Run drive.mount('/content/drive') in the parent Colab first."
    )
if not ROOT.is_dir():
    raise RuntimeError(
        f"Repository not found at {ROOT}. Clone or pull AhmedCode110/AC-MOT first."
    )

DATA_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

print("[TEMPORAL SETUP] Installing pinned runtime dependencies...", flush=True)
sh([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics==8.3.200", "numpy==2.2.6", "scipy==1.15.3",
    "lap", "opencv-python-headless", "pandas",
])


def valid_visdrone(path: Path) -> bool:
    return path.is_dir() and (path / "sequences").is_dir() and (path / "annotations").is_dir()


if not valid_visdrone(VAL_DIR):
    candidates = [
        DRIVE / "VisDrone2019-MOT-val",
        DRIVE / "VisDrone2019-MOT" / "VisDrone2019-MOT-val",
        DATA_ROOT / "VisDrone2019-MOT-val",
    ]
    for p in candidates:
        if valid_visdrone(p):
            VAL_DIR = p
            break

if not valid_visdrone(VAL_DIR):
    print("[TEMPORAL DATA] Validation set not found; downloading portable validation copy...", flush=True)
    import urllib.request
    import zipfile

    archive = DATA_ROOT / "VisDrone2019-MOT-val.zip"
    url = (
        "https://huggingface.co/datasets/vanthanh/VisDrone2019-MOT/resolve/main/"
        "VisDrone2019-MOT-val.zip"
    )
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(DATA_ROOT)
    archive.unlink(missing_ok=True)
    found = [
        p for p in DATA_ROOT.rglob("*")
        if valid_visdrone(p) and "VisDrone2019-MOT-val" in p.name
    ]
    if len(found) == 1:
        VAL_DIR = found[0]
    elif valid_visdrone(DATA_ROOT / "VisDrone2019-MOT-val"):
        VAL_DIR = DATA_ROOT / "VisDrone2019-MOT-val"
    else:
        raise RuntimeError("Could not identify extracted VisDrone validation folder.")

if not WEIGHTS.exists():
    print("[TEMPORAL SETUP] Downloading fixed YOLOv8n weights...", flush=True)
    from ultralytics import YOLO
    old = os.getcwd()
    os.chdir(WEIGHTS.parent)
    try:
        YOLO("yolov8n.pt")
    finally:
        os.chdir(old)
if not WEIGHTS.exists():
    raise RuntimeError("Could not obtain fixed yolov8n.pt")

PINNED_TRACKEVAL = "12c8791b303e0a0b50f753af204249e622d0281a"
need_clone = True
if TRACKEVAL_DIR.exists():
    try:
        rev = subprocess.check_output(
            ["git", "-C", str(TRACKEVAL_DIR), "rev-parse", "HEAD"], text=True
        ).strip()
        need_clone = rev != PINNED_TRACKEVAL
    except Exception:
        need_clone = True
if need_clone:
    if TRACKEVAL_DIR.exists():
        shutil.rmtree(TRACKEVAL_DIR)
    sh(["git", "clone", "-q", "https://github.com/JonathonLuiten/TrackEval.git", str(TRACKEVAL_DIR)])
    sh(["git", "-C", str(TRACKEVAL_DIR), "checkout", "-q", PINNED_TRACKEVAL])

import numpy as np
import pandas as pd
import torch

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_v17 import PresentationController as OldPresentationController
from experiment import dataset_manifest
import scripts.paper_eval_v17 as pe

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required. Select a GPU runtime in Colab.")
print("[GPU]", torch.cuda.get_device_name(0), flush=True)
if "T4" not in torch.cuda.get_device_name(0):
    print("[WARNING] GPU is not Tesla T4; FPS is not directly comparable with T4 reference.", flush=True)

val_names = sorted(p.name for p in (VAL_DIR / "sequences").iterdir() if p.is_dir())
VAL_MANIFEST = dataset_manifest(VAL_DIR, val_names)

print("\n" + "=" * 96, flush=True)
print("AC-MOT TEMPORAL ABLATION", flush=True)
print("=" * 96, flush=True)
print("Validation sequences :", len(VAL_MANIFEST), flush=True)
print("Detector             : FIXED YOLOv8n", flush=True)
print("Tracker              : FIXED tuned ByteTrack", flush=True)
print("SCI/mapping           : OLD FIXED presentation controller", flush=True)
print("Reference             : smoothing=7, stride=10", flush=True)
print("FPS gate              : >=", MIN_FPS, flush=True)
print("Smoke mode            :", TEMPORAL_SMOKE, flush=True)
print("Test-dev              : NOT ACCESSED", flush=True)
print("=" * 96, flush=True)

GT_FILTER = dict(categories=[1, 4, 5, 6, 9], score=1, occlusion_lt=2, truncation_lt=2)


def write_meta(root: Path, manifest, system):
    (root / "configuration.json").write_text(json.dumps({
        "version": "temporal_ablation",
        "systems": [system],
        "ground_truth_filter": GT_FILTER,
        "official_visdrone": False,
        "purpose": "validation-only temporal ablation of smoothing_window and analysis_stride",
    }, indent=2))
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_metrics(path: Path, name: str):
    row = next(r for r in csv.DictReader(path.open()) if r["system"] == name)
    return dict(
        HOTA=float(row["HOTA"]),
        DetA=float(row["DetA"]),
        AssA=float(row["AssA"]),
        MOTA=float(row["MOTA"]),
        IDF1=float(row["IDF1"]),
        IDS=int(float(row["IDS"])),
        FN=int(float(row["FN"])),
        FP=int(float(row["FP"])),
    )


def run_combo(window: int, stride: int):
    name = f"TEMP_W{window}_S{stride}"
    system = dict(
        name=name,
        tracker_profile="tuned",
        adaptive_threshold=True,
        adaptive_resolution=True,
        smoothing_window=int(window),
        analysis_stride=int(stride),
    )

    work_root = Path("/content/acmot_temporal_ablation") / name
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=False)
    write_meta(work_root, VAL_MANIFEST, system)

    timing = pe.run_system(
        system,
        VAL_DIR,
        VAL_MANIFEST,
        WEIGHTS,
        Path("/content/weights/unused.engine"),
        MIN_FPS,
        PROGRESS_EVERY,
        "pytorch",
        16,
        work_root,
        1,
        1,
    )

    eval_out = work_root / "trackeval"
    sh([
        sys.executable,
        str(ROOT / "evaluate.py"),
        str(work_root),
        "--dataset", str(VAL_DIR),
        "--trackeval", str(TRACKEVAL_DIR),
        "--output", str(eval_out),
    ], cwd=ROOT)

    result = parse_metrics(eval_out / "summary.csv", name)
    result.update(
        smoothing_window=int(window),
        analysis_stride=int(stride),
        FPS=float(timing["processing_fps"]),
        p95_ms=float(timing["processing_p95_ms"]),
        mean_imgsz=float(timing["mean_imgsz"]),
        mean_conf=float(timing["mean_conf"]),
        mean_nms_iou=float(timing["mean_nms_iou"]),
        analysis_interval_seconds=float(stride / NOMINAL_FPS),
        causal_history_span_seconds=float(max(window - 1, 0) * stride / NOMINAL_FPS),
    )

    compact = RESULT_ROOT / f"TEMPORAL_W{window}_S{stride}.json"
    compact.write_text(json.dumps(result, indent=2))
    shutil.rmtree(work_root, ignore_errors=True)
    torch.cuda.empty_cache()
    return result


windows = SMOKE_WINDOWS if TEMPORAL_SMOKE else FULL_WINDOWS
strides = SMOKE_STRIDES if TEMPORAL_SMOKE else FULL_STRIDES

# Reference first, then the remaining grid. This gives an early IDS gate value.
combos = [(REFERENCE_WINDOW, REFERENCE_STRIDE)]
combos += [
    (w, s) for w in windows for s in strides
    if (w, s) != (REFERENCE_WINDOW, REFERENCE_STRIDE)
]

csv_path = RESULT_ROOT / (
    "TEMPORAL_ABLATION_SMOKE.csv" if TEMPORAL_SMOKE else "TEMPORAL_ABLATION_FULL.csv"
)
rows = []
if csv_path.exists():
    try:
        rows = pd.read_csv(csv_path).to_dict("records")
    except Exception:
        rows = []

existing = {
    (int(r["smoothing_window"]), int(r["analysis_stride"])): r
    for r in rows
    if "smoothing_window" in r and "analysis_stride" in r
}

started = time.perf_counter()
for i, (window, stride) in enumerate(combos, 1):
    if (window, stride) in existing:
        print(
            f"[TEMPORAL {i}/{len(combos)}] reuse W={window} stride={stride}",
            flush=True,
        )
        continue

    print("\n" + "#" * 96, flush=True)
    print(
        f"TEMPORAL COMBINATION {i}/{len(combos)} | smoothing={window} | stride={stride} | "
        f"analysis_interval={stride / NOMINAL_FPS:.3f}s | "
        f"history_span={max(window - 1, 0) * stride / NOMINAL_FPS:.3f}s",
        flush=True,
    )
    print("#" * 96, flush=True)

    result = run_combo(window, stride)
    rows.append(result)
    existing[(window, stride)] = result
    pd.DataFrame(rows).sort_values(["smoothing_window", "analysis_stride"]).to_csv(csv_path, index=False)

    elapsed = time.perf_counter() - started
    completed = len(existing)
    avg = elapsed / max(i, 1)
    eta = avg * max(len(combos) - i, 0)
    print(
        f"[TEMPORAL RESULT] W={window} stride={stride} | "
        f"MOTA={100*result['MOTA']:.3f}% HOTA={100*result['HOTA']:.3f}% "
        f"IDF1={100*result['IDF1']:.3f}% IDS={result['IDS']} FPS={result['FPS']:.2f} | "
        f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m",
        flush=True,
    )

# Work only with the requested grid for this mode.
mode_rows = [existing[(w, s)] for (w, s) in combos if (w, s) in existing]
reference = existing.get((REFERENCE_WINDOW, REFERENCE_STRIDE))
if reference is None:
    raise RuntimeError("Reference W=7, stride=10 was not evaluated.")

reference_ids = int(reference["IDS"])
for r in mode_rows:
    r["feasible"] = bool(float(r["FPS"]) >= MIN_FPS and int(r["IDS"]) <= reference_ids)

pd.DataFrame(mode_rows).sort_values(["smoothing_window", "analysis_stride"]).to_csv(csv_path, index=False)

print("\n" + "=" * 96, flush=True)
print("TEMPORAL ABLATION SUMMARY", flush=True)
print("=" * 96, flush=True)
print(
    f"Reference W=7 stride=10 | MOTA={100*float(reference['MOTA']):.3f}% | "
    f"IDS={int(reference['IDS'])} | FPS={float(reference['FPS']):.2f}",
    flush=True,
)

if TEMPORAL_SMOKE:
    print("SMOKE TEST COMPLETE: no temporal configuration was frozen.", flush=True)
    print("Run with ACMOT_TEMPORAL_SMOKE=0 for the full 25-combination ablation.", flush=True)
    print("Test-dev was not accessed.", flush=True)
    sys.exit(0)

feasible = [r for r in mode_rows if bool(r["feasible"])]
if not feasible:
    raise RuntimeError("No feasible temporal candidate found, including the reference. Check timing/protocol.")

best = max(
    feasible,
    key=lambda r: (
        float(r["MOTA"]),
        -int(r["IDS"]),
        float(r["HOTA"]),
        float(r["IDF1"]),
        float(r["FPS"]),
    ),
)

frozen = {
    "method": "validation-only temporal ablation before joint SCI optimization",
    "test_used": False,
    "official_visdrone": False,
    "reference": {
        "smoothing_window": REFERENCE_WINDOW,
        "analysis_stride": REFERENCE_STRIDE,
        "metrics": reference,
    },
    "candidate_grid": {
        "smoothing_windows": FULL_WINDOWS,
        "analysis_strides": FULL_STRIDES,
        "nominal_tracker_fps": NOMINAL_FPS,
        "stride_intervals_seconds": {str(s): s / NOMINAL_FPS for s in FULL_STRIDES},
        "rationale": (
            "The grid spans every-frame analysis to sub-second temporal subsampling and "
            "no smoothing to progressively longer causal averaging. The grid bounds are a "
            "declared validation search budget, not claimed optimal operating values."
        ),
    },
    "selection_rule": (
        "Validation only: FPS>=25 and IDS<=W7/S10 reference; then highest MOTA, "
        "lower IDS, higher HOTA, higher IDF1, higher FPS."
    ),
    "selected": {
        "smoothing_window": int(best["smoothing_window"]),
        "analysis_stride": int(best["analysis_stride"]),
        "analysis_interval_seconds": float(best["analysis_interval_seconds"]),
        "causal_history_span_seconds": float(best["causal_history_span_seconds"]),
        "validation_metrics": best,
    },
    "fixed_during_ablation": [
        "pretrained YOLOv8n detector",
        "old SCI cue formulation and detector-control mapping",
        "tuned ByteTrack configuration",
        "validation split",
        "T4 timing protocol when a T4 is used",
        "TrackEval revision",
    ],
}

frozen_path = RESULT_ROOT / "FROZEN_TEMPORAL_CONFIG.json"
frozen_path.write_text(json.dumps(frozen, indent=2))

# Also write an easy-to-present ranked CSV.
ranked = sorted(
    mode_rows,
    key=lambda r: (
        not bool(r["feasible"]),
        -float(r["MOTA"]),
        int(r["IDS"]),
        -float(r["HOTA"]),
        -float(r["IDF1"]),
        -float(r["FPS"]),
    ),
)
pd.DataFrame(ranked).to_csv(RESULT_ROOT / "TEMPORAL_ABLATION_RANKED.csv", index=False)

print(
    f"SELECTED | smoothing_window={int(best['smoothing_window'])} | "
    f"analysis_stride={int(best['analysis_stride'])} | "
    f"MOTA={100*float(best['MOTA']):.3f}% | IDS={int(best['IDS'])} | "
    f"HOTA={100*float(best['HOTA']):.3f}% | IDF1={100*float(best['IDF1']):.3f}% | "
    f"FPS={float(best['FPS']):.2f}",
    flush=True,
)
print("Frozen temporal config:", frozen_path, flush=True)
print("Test-dev was not accessed.", flush=True)
