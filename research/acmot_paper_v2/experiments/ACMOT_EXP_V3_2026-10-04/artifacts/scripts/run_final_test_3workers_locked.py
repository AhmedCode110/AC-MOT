"""Run exactly one predeclared system in the locked three-worker held-out test.

Worker mapping is fixed by FINAL_TEST_3WORKER_PROTOCOL.json:
  1 -> Baseline_Default
  2 -> Old_ACMOT_Frozen
  3 -> New_ACMOT_Frozen

All selection and tuning must already be complete. This script never changes
parameters based on test results. Each worker writes one immutable result JSON
into the shared result folder. Use merge_final_test_3workers_locked.py only after
all three worker results exist.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path


def sh(args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ROOT = Path(os.environ.get("ACMOT_REPO", "/content/AC-MOT"))
DRIVE = Path(os.environ.get("ACMOT_DRIVE", "/content/drive/MyDrive"))
RESULT_ROOT = Path(
    os.environ.get(
        "ACMOT_RESULT_ROOT",
        str(DRIVE / "AC-MOT-shared" / "defensible_acmot_3workers"),
    )
)
TEST_DIR = Path(os.environ.get("ACMOT_TEST_DIR", str(DRIVE / "VisDrone2019-MOT-test-dev")))
WEIGHTS = Path(os.environ.get("ACMOT_WEIGHTS", "/content/weights/yolov8n.pt"))
TRACKEVAL_DIR = Path(os.environ.get("ACMOT_TRACKEVAL", "/content/TrackEval"))
WORKER_ID = int(os.environ.get("ACMOT_FINAL_WORKER_ID", "0"))
PROGRESS_EVERY = int(os.environ.get("ACMOT_PROGRESS_EVERY", "50"))
ALLOW_NON_T4 = os.environ.get("ACMOT_ALLOW_NON_T4", "0") == "1"

PLAN_PATH = RESULT_ROOT / "FINAL_TEST_3WORKER_PROTOCOL.json"
FROZEN_PATH = RESULT_ROOT / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"
CALIBRATION_PATH = RESULT_ROOT / "DETECTOR_DERIVED_CUE_CALIBRATION.json"
FINAL_LOCK = RESULT_ROOT / "FINAL_TEST_DONE.json"

if WORKER_ID not in {1, 2, 3}:
    raise RuntimeError("Set ACMOT_FINAL_WORKER_ID to 1, 2, or 3.")
if FINAL_LOCK.exists():
    raise RuntimeError("FINAL_TEST_DONE.json already exists; refusing another held-out run.")
for p in [PLAN_PATH, FROZEN_PATH, CALIBRATION_PATH]:
    if not p.exists():
        raise RuntimeError(f"Missing locked prerequisite: {p}")
if not TEST_DIR.is_dir() or not (TEST_DIR / "sequences").is_dir() or not (TEST_DIR / "annotations").is_dir():
    raise RuntimeError(f"Invalid held-out dataset path: {TEST_DIR}")

PLAN = json.loads(PLAN_PATH.read_text())
FROZEN = json.loads(FROZEN_PATH.read_text())
CALIBRATION = json.loads(CALIBRATION_PATH.read_text())

expected_hashes = PLAN["artifact_hashes"]
if sha256(FROZEN_PATH) != expected_hashes["FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"]:
    raise RuntimeError("Frozen optimized configuration changed after final-test protocol freeze.")
if sha256(CALIBRATION_PATH) != expected_hashes["DETECTOR_DERIVED_CUE_CALIBRATION.json"]:
    raise RuntimeError("Cue calibration changed after final-test protocol freeze.")

worker_spec = next(x for x in PLAN["systems"] if int(x["worker_id"]) == WORKER_ID)
SYSTEM_NAME = worker_spec["name"]
slug = SYSTEM_NAME.upper()
RESULT_JSON = RESULT_ROOT / f"FINAL_TEST_WORKER_{WORKER_ID}_{slug}.json"
if RESULT_JSON.exists():
    print("THIS FINAL-TEST WORKER RESULT ALREADY EXISTS — refusing to rerun.")
    print(RESULT_JSON.read_text())
    raise SystemExit(0)

print("#" * 110)
print("AC-MOT LOCKED 3-WORKER HELD-OUT FINAL TEST")
print("Worker          :", WORKER_ID)
print("System          :", SYSTEM_NAME)
print("Shared folder   :", RESULT_ROOT)
print("Test-dev        : WILL BE ACCESSED NOW")
print("Tuning/search   : DISABLED")
print("#" * 110, flush=True)

print("[SETUP] Installing pinned runtime dependencies...", flush=True)
sh([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics==8.3.200", "numpy==2.2.6", "scipy==1.15.3",
    "lap", "opencv-python-headless", "pandas",
])

PINNED_TRACKEVAL = PLAN["pinned_trackeval_commit"]
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
gpu_name = torch.cuda.get_device_name(0)
print("[GPU]", gpu_name, flush=True)
if "T4" not in gpu_name and not ALLOW_NON_T4:
    raise RuntimeError(
        "Final comparison requires the same T4 GPU class for defensible FPS comparison. "
        "Use a T4 runtime, or set ACMOT_ALLOW_NON_T4=1 only if you intentionally accept non-comparable FPS."
    )

PARAMS = FROZEN["optimized_parameters"]
RESOLUTIONS = [int(x) for x in FROZEN["resolution_levels"]]
SMOOTHING_WINDOW = int(FROZEN["smoothing_window"])
ANALYSIS_STRIDE = int(FROZEN["analysis_stride"])
MIN_FPS = float(PLAN["minimum_processing_fps"])


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


class StaticBaselineController:
    def __init__(self, spec: PresentationSpec):
        self.spec = spec.validate()

    def choose(self, frame, visual, previous):
        return {
            "conf": 0.25,
            "nms": 0.45,
            "size": 640,
            "sci": 0.0,
            "scene": "static_project_baseline",
        }


class FrozenSCIController:
    def __init__(self, spec: PresentationSpec):
        self.spec = spec.validate()
        self.history = deque(maxlen=self.spec.smoothing_window)
        self.sci = 0.0
        self.tiny = 0.0
        keys = ["crowd", "tiny", "edge", "night", "blur"]
        raw = np.asarray([float(PARAMS[f"weight_{k}"]) for k in keys], dtype=float)
        if raw.sum() <= 0:
            raise RuntimeError("Frozen SCI weights sum to zero.")
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
        return {
            "conf": conf,
            "nms": nms,
            "size": int(size),
            "sci": self.sci,
            "scene": "frozen_empirical_sci",
        }


if WORKER_ID == 1:
    system = dict(
        name=SYSTEM_NAME,
        tracker_profile="default",
        adaptive_threshold=False,
        adaptive_resolution=False,
        smoothing_window=1,
        analysis_stride=1,
    )
    controller_factory = StaticBaselineController
    visual_fn = lambda _img: {}
    warm_sizes = [640]
elif WORKER_ID == 2:
    system = dict(
        name=SYSTEM_NAME,
        tracker_profile="tuned",
        adaptive_threshold=True,
        adaptive_resolution=True,
        smoothing_window=7,
        analysis_stride=10,
    )
    controller_factory = OldPresentationController
    visual_fn = pe.visual
    warm_sizes = [640, 736, 832]
else:
    system = dict(
        name=SYSTEM_NAME,
        tracker_profile="tuned",
        adaptive_threshold=True,
        adaptive_resolution=True,
        smoothing_window=SMOOTHING_WINDOW,
        analysis_stride=ANALYSIS_STRIDE,
    )
    controller_factory = FrozenSCIController
    visual_fn = robust_visual
    warm_sizes = RESOLUTIONS

names = sorted(p.name for p in (TEST_DIR / "sequences").iterdir() if p.is_dir())
manifest = dataset_manifest(TEST_DIR, names)
output = Path(f"/content/acmot_final_test_worker_{WORKER_ID}")
if output.exists():
    shutil.rmtree(output)
output.mkdir(parents=True)

GT_FILTER = PLAN["ground_truth_filter"]
(output / "configuration.json").write_text(json.dumps({
    "version": PLAN["protocol_version"],
    "worker_id": WORKER_ID,
    "system": system,
    "ground_truth_filter": GT_FILTER,
    "official_visdrone": False,
    "selection_or_tuning_on_test": False,
    "protocol_sha256": sha256(PLAN_PATH),
}, indent=2))
(output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))

original_controller = pe.PresentationController
original_visual = pe.visual
original_detect = pe.detect_realtime

mapping = {640: warm_sizes[0]}
if len(warm_sizes) == 3:
    mapping = {640: warm_sizes[0], 736: warm_sizes[1], 832: warm_sizes[2]}


def detect_with_exact_warm(model, img, size, nms, conf, backend):
    # paper_eval_v17 warm-up calls use nms=.45/conf=.19. Replace only those
    # warm-up shapes with the exact sizes required by this locked system.
    if abs(float(nms) - 0.45) < 1e-12 and abs(float(conf) - 0.19) < 1e-12:
        size = mapping.get(int(size), warm_sizes[0])
    return original_detect(model, img, size, nms, conf, backend)

try:
    pe.PresentationController = controller_factory
    pe.visual = visual_fn
    pe.detect_realtime = detect_with_exact_warm
    timing = pe.run_system(
        system, TEST_DIR, manifest, WEIGHTS, Path("/content/weights/unused.engine"),
        MIN_FPS, PROGRESS_EVERY, "pytorch", 16, output, 1, 1,
    )
finally:
    pe.PresentationController = original_controller
    pe.visual = original_visual
    pe.detect_realtime = original_detect

trackeval_out = output / "trackeval"
sh([
    sys.executable, str(ROOT / "evaluate.py"), str(output),
    "--dataset", str(TEST_DIR), "--trackeval", str(TRACKEVAL_DIR),
    "--output", str(trackeval_out),
], cwd=ROOT)

rows = list(csv.DictReader((trackeval_out / "summary.csv").open()))
row = next((r for r in rows if r["system"] == SYSTEM_NAME), None)
if row is None:
    raise RuntimeError(f"Could not find {SYSTEM_NAME} in TrackEval summary.")

metrics = {
    "MOTA": float(row["MOTA"]),
    "HOTA": float(row["HOTA"]),
    "IDF1": float(row["IDF1"]),
    "IDS": int(float(row["IDS"])),
    "FN": int(float(row["FN"])),
    "FP": int(float(row["FP"])),
    "FPS": float(timing["processing_fps"]),
}

shared_worker_dir = RESULT_ROOT / "final_test_3workers" / f"worker_{WORKER_ID}_{SYSTEM_NAME}"
shared_worker_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(trackeval_out / "summary.csv", shared_worker_dir / "summary.csv")
shutil.copy2(output / "configuration.json", shared_worker_dir / "configuration.json")
shutil.copy2(output / "dataset_manifest.json", shared_worker_dir / "dataset_manifest.json")
(shared_worker_dir / "timing.json").write_text(json.dumps(timing, indent=2))

result = {
    "status": "FINAL_TEST_WORKER_COMPLETE",
    "protocol_version": PLAN["protocol_version"],
    "protocol_sha256": sha256(PLAN_PATH),
    "worker_id": WORKER_ID,
    "system": SYSTEM_NAME,
    "test_dir": str(TEST_DIR),
    "sequence_count": len(manifest),
    "gpu": gpu_name,
    "frozen_config_sha256": sha256(FROZEN_PATH),
    "calibration_sha256": sha256(CALIBRATION_PATH),
    "metrics": metrics,
    "selection_or_tuning_on_test": False,
    "warning": "Held-out test data has now been exposed for this locked system. Do not retune based on this result.",
}
RESULT_JSON.write_text(json.dumps(result, indent=2))

print("\n" + "=" * 110)
print("FINAL TEST WORKER COMPLETE")
print(json.dumps(result, indent=2))
print("Saved:", RESULT_JSON)
print("Do not modify the frozen protocol or parameters after seeing this result.")
print("=" * 110)
