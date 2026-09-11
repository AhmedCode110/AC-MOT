"""One-shot held-out final evaluation for the defensible AC-MOT configuration.

Run this ONLY after validation ablations and Optuna are complete and reviewed.
The script refuses to run a second time if FINAL_TEST_DONE.json already exists.
It evaluates, in the same locked run:
- Baseline_Default: stock project baseline
- Full_ACMOT_Frozen: frozen validation-selected AC-MOT

No tuning, searching, or parameter selection occurs here.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path


def sh(args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


ROOT = Path(os.environ.get("ACMOT_REPO", "/content/AC-MOT"))
DRIVE = Path(os.environ.get("ACMOT_DRIVE", "/content/drive/MyDrive"))
RESULT_ROOT = Path(
    os.environ.get("ACMOT_RESULT_ROOT", str(DRIVE / "AC-MOT-results" / "defensible_acmot"))
)
TEST_DIR = Path(os.environ.get("ACMOT_TEST_DIR", str(DRIVE / "VisDrone2019-MOT-test-dev")))
WEIGHTS = Path(os.environ.get("ACMOT_WEIGHTS", "/content/weights/yolov8n.pt"))
TRACKEVAL_DIR = Path(os.environ.get("ACMOT_TRACKEVAL", "/content/TrackEval"))
MIN_FPS = float(os.environ.get("ACMOT_MIN_FPS", "25"))
PROGRESS_EVERY = int(os.environ.get("ACMOT_PROGRESS_EVERY", "50"))
FROZEN_PATH = RESULT_ROOT / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"
CALIBRATION_PATH = RESULT_ROOT / "DETECTOR_DERIVED_CUE_CALIBRATION.json"
LOCK = RESULT_ROOT / "FINAL_TEST_DONE.json"

if LOCK.exists():
    saved = json.loads(LOCK.read_text())
    print("FINAL TEST ALREADY EXISTS — refusing to rerun.")
    print(json.dumps(saved, indent=2))
    sys.exit(0)
if not FROZEN_PATH.exists():
    raise RuntimeError(f"Missing frozen config: {FROZEN_PATH}")
if not CALIBRATION_PATH.exists():
    raise RuntimeError(f"Missing cue calibration: {CALIBRATION_PATH}")
if not TEST_DIR.is_dir() or not (TEST_DIR / "sequences").is_dir() or not (TEST_DIR / "annotations").is_dir():
    raise RuntimeError(f"Invalid held-out dataset path: {TEST_DIR}")

print("[SETUP] Installing pinned runtime dependencies...", flush=True)
sh([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics==8.3.200", "numpy==2.2.6", "scipy==1.15.3",
    "lap", "opencv-python-headless", "pandas",
])

PINNED_TRACKEVAL = "12c8791b303e0a0b50f753af204249e622d0281a"
need_clone = True
if TRACKEVAL_DIR.exists():
    try:
        rev = subprocess.check_output(["git", "-C", str(TRACKEVAL_DIR), "rev-parse", "HEAD"], text=True).strip()
        need_clone = rev != PINNED_TRACKEVAL
    except Exception:
        need_clone = True
if need_clone:
    if TRACKEVAL_DIR.exists():
        shutil.rmtree(TRACKEVAL_DIR)
    sh(["git", "clone", "-q", "https://github.com/JonathonLuiten/TrackEval.git", str(TRACKEVAL_DIR)])
    sh(["git", "-C", str(TRACKEVAL_DIR), "checkout", "-q", PINNED_TRACKEVAL])

import cv2
import numpy as np
import torch

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core import boxes
from core_v17 import PresentationController as OldPresentationController, PresentationSpec
from experiment import dataset_manifest
import scripts.paper_eval_v17 as pe

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")
print("[GPU]", torch.cuda.get_device_name(0), flush=True)
if "T4" not in torch.cuda.get_device_name(0):
    print("[WARNING] Non-T4 GPU: FPS is not directly comparable with the T4 thesis reference.", flush=True)

FROZEN = json.loads(FROZEN_PATH.read_text())
CALIBRATION = json.loads(CALIBRATION_PATH.read_text())
PARAMS = FROZEN["optimized_parameters"]
RESOLUTIONS = [int(x) for x in FROZEN["resolution_levels"]]
SMOOTHING_WINDOW = int(FROZEN["smoothing_window"])
ANALYSIS_STRIDE = int(FROZEN["analysis_stride"])


def robust_visual(img):
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    return {
        "brightness": float(np.mean(gray)),
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edges": float(np.mean(gradient)),
    }


def empirical_rank(value, sorted_values):
    arr = np.asarray(sorted_values, dtype=float)
    return float(np.searchsorted(arr, value, side="right") / len(arr)) if len(arr) else 0.0


class FrozenSCIController:
    def __init__(self, spec: PresentationSpec):
        self.spec = spec.validate()
        self.history = deque(maxlen=self.spec.smoothing_window)
        self.sci = 0.0
        self.tiny = 0.0
        keys = ["crowd", "tiny", "edge", "night", "blur"]
        raw = np.asarray([float(PARAMS[f"weight_{k}"]) for k in keys], dtype=float)
        raw /= raw.sum()
        self.weights = dict(zip(keys, raw.tolist()))

    def choose(self, frame, visual, previous):
        analyze = frame == 1 or (frame - 1) % self.spec.analysis_stride == 0
        if analyze:
            previous = boxes(previous)
            n = len(previous)
            if n:
                areas = (previous[:, 2] - previous[:, 0]) * (previous[:, 3] - previous[:, 1])
                self.tiny = float(np.mean(areas < 32 * 32))
            else:
                self.tiny = 0.0
            cues = {
                "crowd": empirical_rank(n, CALIBRATION["crowd_sorted"]),
                "tiny": self.tiny,
                "edge": empirical_rank(float(visual["edges"]), CALIBRATION["edge_sorted"]),
                "night": 1.0 - empirical_rank(float(visual["brightness"]), CALIBRATION["brightness_sorted"]),
                "blur": 1.0 - empirical_rank(float(visual["blur"]), CALIBRATION["blur_sorted"]),
            }
            raw_sci = float(np.clip(sum(self.weights[k] * cues[k] for k in self.weights), 0.0, 1.0))
            self.history.append(raw_sci)
            self.sci = float(np.mean(self.history))

        conf = float(PARAMS["conf_easy"] + self.sci * (PARAMS["conf_hard"] - PARAMS["conf_easy"]))
        nms = float(PARAMS["nms_easy"] + self.sci * (PARAMS["nms_hard"] - PARAMS["nms_easy"]))
        r0, r1, r2 = RESOLUTIONS
        if self.sci >= float(PARAMS["threshold_high"]):
            size = r2
        elif self.sci >= float(PARAMS["threshold_mid"]):
            size = r1
        else:
            size = r0
        return {"conf": conf, "nms": nms, "size": int(size), "sci": self.sci, "scene": "frozen_empirical_sci"}


names = sorted(p.name for p in (TEST_DIR / "sequences").iterdir() if p.is_dir())
manifest = dataset_manifest(TEST_DIR, names)
output = Path("/content/acmot_frozen_final_test")
if output.exists():
    shutil.rmtree(output)
output.mkdir(parents=True)

systems = [
    dict(
        name="Baseline_Default", tracker_profile="default",
        adaptive_threshold=False, adaptive_resolution=False,
        smoothing_window=1, analysis_stride=1,
    ),
    dict(
        name="Full_ACMOT_Frozen", tracker_profile="tuned",
        adaptive_threshold=True, adaptive_resolution=True,
        smoothing_window=SMOOTHING_WINDOW, analysis_stride=ANALYSIS_STRIDE,
    ),
]
GT_FILTER = dict(categories=[1, 4, 5, 6, 9], score=1, occlusion_lt=2, truncation_lt=2)
(output / "configuration.json").write_text(json.dumps({
    "version": "defensible_final_test_v1",
    "systems": systems,
    "ground_truth_filter": GT_FILTER,
    "official_visdrone": False,
    "frozen_config": FROZEN,
    "selection_or_tuning_on_test": False,
}, indent=2))
(output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))

all_timing = []
for i, system in enumerate(systems, 1):
    old_controller, old_visual = pe.PresentationController, pe.visual
    try:
        if system["name"] == "Full_ACMOT_Frozen":
            pe.PresentationController = FrozenSCIController
            pe.visual = robust_visual
        else:
            pe.PresentationController = OldPresentationController
        timing = pe.run_system(
            system, TEST_DIR, manifest, WEIGHTS, Path("/content/weights/unused.engine"),
            MIN_FPS, PROGRESS_EVERY, "pytorch", 16, output, i, len(systems),
        )
        all_timing.append(timing)
    finally:
        pe.PresentationController, pe.visual = old_controller, old_visual

trackeval_out = output / "trackeval"
sh([
    sys.executable, str(ROOT / "evaluate.py"), str(output),
    "--dataset", str(TEST_DIR), "--trackeval", str(TRACKEVAL_DIR),
    "--output", str(trackeval_out),
], cwd=ROOT)

rows = list(csv.DictReader((trackeval_out / "summary.csv").open()))
timing_map = {x["system"]: x for x in all_timing}
results = {}
for row in rows:
    name = row["system"]
    if name not in timing_map:
        continue
    results[name] = {
        "MOTA": float(row["MOTA"]), "HOTA": float(row["HOTA"]),
        "IDF1": float(row["IDF1"]), "IDS": int(float(row["IDS"])),
        "FN": int(float(row["FN"])), "FP": int(float(row["FP"])),
        "FPS": float(timing_map[name]["processing_fps"]),
    }

if set(results) != {"Baseline_Default", "Full_ACMOT_Frozen"}:
    raise RuntimeError(f"Unexpected final result systems: {sorted(results)}")

final = {
    "status": "FINAL_TEST_DONE",
    "test_dir": str(TEST_DIR),
    "sequence_count": len(manifest),
    "frozen_config": str(FROZEN_PATH),
    "results": results,
    "warning": "Held-out result has now been exposed. Do not retune and re-label a later run as an unbiased final test.",
    "protocol": "custom class-agnostic AC-MOT evaluation; not official VisDrone leaderboard",
}
LOCK.write_text(json.dumps(final, indent=2))
(RESULT_ROOT / "FINAL_TEST_RESULTS.json").write_text(json.dumps(final, indent=2))

print("\n" + "=" * 100, flush=True)
print("FINAL HELD-OUT TEST COMPLETE — LOCK CREATED", flush=True)
print(json.dumps(results, indent=2), flush=True)
print("Lock:", LOCK, flush=True)
print("DO NOT RETUNE ON THESE RESULTS.", flush=True)
print("=" * 100, flush=True)
