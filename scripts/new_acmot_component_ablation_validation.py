"""Validation-only component ablation for the NEW optimized AC-MOT.

Run this AFTER Worker 3 has produced FROZEN_DEFENSIBLE_ACMOT_CONFIG.json and
BEFORE the one-shot held-out test.

Purpose
-------
Create a clean, nested A0->A3 ablation for the new controller while keeping
YOLOv8n and tuned ByteTrack fixed so the controller contribution is isolated.

A0 Study Baseline
    Static validation-selected anchor: fixed confidence, NMS IoU and resolution.

A1 + Adaptive Confidence
    New empirical SCI + frozen temporal design + optimized confidence mapping.
    NMS IoU and resolution remain fixed at the static anchor.

A2 + Adaptive NMS
    A1 + optimized NMS IoU mapping. Resolution remains fixed at the anchor.

A3 Full New AC-MOT
    A2 + adaptive resolution using the three validation-selected resolution
    levels and optimized SCI switching thresholds.

Historical comparator (outside A0-A3)
    HIST_OLD_ACMOT_W7_S10 uses the original heuristic PresentationController
    with tuned ByteTrack and the original W=7 / stride=10 temporal design.

Protocol
--------
- VisDrone validation only; test-dev is NEVER accessed.
- Fixed pretrained YOLOv8n.
- Fixed tuned ByteTrack for A0-A3 and historical comparator.
- Same custom class-agnostic AC-MOT TrackEval protocol as the rest of the repo.
- Exact tested resolution warm-up is used for fair FPS measurement.
- A0 disables unused visual SceneAnalyzer work because its controller is static.
- The script refuses to run after FINAL_TEST_DONE.json exists, protecting the
  held-out protocol from post-test component selection/retuning.
"""
from __future__ import annotations

import csv
import gc
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
DATA_ROOT = Path(os.environ.get("ACMOT_DATA_ROOT", str(DRIVE / "AC-MOT-data")))
RESULT_ROOT = Path(
    os.environ.get("ACMOT_RESULT_ROOT", str(DRIVE / "AC-MOT-results" / "defensible_acmot"))
)
VAL_DIR = Path(os.environ.get("ACMOT_VAL_DIR", str(DATA_ROOT / "VisDrone2019-MOT-val")))
WEIGHTS = Path(os.environ.get("ACMOT_WEIGHTS", "/content/weights/yolov8n.pt"))
TRACKEVAL_DIR = Path(os.environ.get("ACMOT_TRACKEVAL", "/content/TrackEval"))
MIN_FPS = float(os.environ.get("ACMOT_MIN_FPS", "25"))
PROGRESS_EVERY = int(os.environ.get("ACMOT_PROGRESS_EVERY", "50"))
FORCE = os.environ.get("ACMOT_FORCE_NEW_ABLATION", "0") == "1"
INCLUDE_HISTORICAL = os.environ.get("ACMOT_INCLUDE_HISTORICAL", "1") == "1"

FROZEN_PATH = RESULT_ROOT / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"
CALIBRATION_PATH = RESULT_ROOT / "DETECTOR_DERIVED_CUE_CALIBRATION.json"
FINAL_LOCK = RESULT_ROOT / "FINAL_TEST_DONE.json"
DONE_MARKER = RESULT_ROOT / "NEW_ACMOT_COMPONENT_ABLATION_DONE.json"

if not DRIVE.is_dir():
    raise RuntimeError("Mount Google Drive first.")
if not ROOT.is_dir():
    raise RuntimeError(f"Repository not found at {ROOT}")
if FINAL_LOCK.exists():
    raise RuntimeError(
        "FINAL_TEST_DONE.json already exists. This validation ablation is intentionally "
        "blocked after held-out test exposure to avoid post-test retuning/selection."
    )
if not FROZEN_PATH.exists():
    raise RuntimeError(
        f"Missing frozen optimized configuration: {FROZEN_PATH}. Finish Worker 3 first."
    )
if not CALIBRATION_PATH.exists():
    raise RuntimeError(
        f"Missing detector-derived cue calibration: {CALIBRATION_PATH}. Finish Worker 3 first."
    )
if DONE_MARKER.exists() and not FORCE:
    print("NEW AC-MOT COMPONENT ABLATION ALREADY COMPLETE")
    print(DONE_MARKER.read_text())
    print("Set ACMOT_FORCE_NEW_ABLATION=1 only if you intentionally want to rerun it.")
    sys.exit(0)

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

print("[SETUP] Installing pinned runtime dependencies...", flush=True)
sh([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics==8.3.200", "numpy==2.2.6", "scipy==1.15.3",
    "lap", "opencv-python-headless", "pandas",
])


def valid_visdrone(path: Path) -> bool:
    return path.is_dir() and (path / "sequences").is_dir() and (path / "annotations").is_dir()


if not valid_visdrone(VAL_DIR):
    for p in [
        DRIVE / "VisDrone2019-MOT-val",
        DRIVE / "VisDrone2019-MOT" / "VisDrone2019-MOT-val",
        DATA_ROOT / "VisDrone2019-MOT-val",
    ]:
        if valid_visdrone(p):
            VAL_DIR = p
            break
if not valid_visdrone(VAL_DIR):
    raise RuntimeError("VisDrone validation set not found. This script never downloads/uses test-dev.")

if not WEIGHTS.exists():
    from ultralytics import YOLO
    old = os.getcwd(); os.chdir(WEIGHTS.parent)
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

import cv2
import numpy as np
import pandas as pd
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
    print("[WARNING] Non-T4 GPU: FPS is not directly comparable with T4 thesis measurements.", flush=True)

FROZEN = json.loads(FROZEN_PATH.read_text())
CALIBRATION = json.loads(CALIBRATION_PATH.read_text())
PARAMS = FROZEN["optimized_parameters"]
RESOLUTIONS = [int(x) for x in FROZEN["resolution_levels"]]
SMOOTHING_WINDOW = int(FROZEN["smoothing_window"])
ANALYSIS_STRIDE = int(FROZEN["analysis_stride"])
SPACE = FROZEN["operating_ablation"]
ANCHOR = SPACE["cue_calibration_anchor"]
ANCHOR_RESOLUTION = int(ANCHOR["resolution"])
ANCHOR_CONF = float(ANCHOR["confidence"])
ANCHOR_NMS = float(ANCHOR["nms_iou"])

if len(RESOLUTIONS) != 3:
    raise RuntimeError(f"Expected exactly three frozen resolution levels, got {RESOLUTIONS}")

val_names = sorted(p.name for p in (VAL_DIR / "sequences").iterdir() if p.is_dir())
VAL_MANIFEST = dataset_manifest(VAL_DIR, val_names)
GT_FILTER = dict(categories=[1, 4, 5, 6, 9], score=1, occlusion_lt=2, truncation_lt=2)

print("\n" + "=" * 100, flush=True)
print("NEW OPTIMIZED AC-MOT COMPONENT ABLATION", flush=True)
print("Validation only      : YES", flush=True)
print("Test-dev             : NOT ACCESSED", flush=True)
print("Detector             : FIXED pretrained YOLOv8n", flush=True)
print("Tracker              : FIXED tuned ByteTrack", flush=True)
print("Frozen temporal      : W=", SMOOTHING_WINDOW, "stride=", ANALYSIS_STRIDE, flush=True)
print("Static anchor        : conf=", ANCHOR_CONF, "nms=", ANCHOR_NMS, "imgsz=", ANCHOR_RESOLUTION, flush=True)
print("Resolution levels    :", RESOLUTIONS, flush=True)
print("Historical comparator:", INCLUDE_HISTORICAL, flush=True)
print("=" * 100, flush=True)


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


class NewAblationController:
    """One frozen empirical SCI controller with component gates for A0-A3."""

    def __init__(self, spec: PresentationSpec, mode: str):
        self.spec = spec.validate()
        self.mode = mode
        self.history = deque(maxlen=self.spec.smoothing_window)
        self.sci = 0.0
        self.tiny = 0.0
        keys = ["crowd", "tiny", "edge", "night", "blur"]
        raw = np.asarray([float(PARAMS[f"weight_{k}"]) for k in keys], dtype=float)
        if raw.sum() <= 0:
            raw[:] = 1.0
        raw /= raw.sum()
        self.weights = dict(zip(keys, raw.tolist()))

    def choose(self, frame, visual, previous):
        if self.mode == "static":
            return {
                "conf": ANCHOR_CONF,
                "nms": ANCHOR_NMS,
                "size": ANCHOR_RESOLUTION,
                "sci": 0.0,
                "scene": "A0_static_anchor",
            }

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

        conf = ANCHOR_CONF
        nms = ANCHOR_NMS
        size = ANCHOR_RESOLUTION

        if self.mode in {"conf", "conf_nms", "full"}:
            conf = float(PARAMS["conf_easy"] + self.sci * (PARAMS["conf_hard"] - PARAMS["conf_easy"]))
        if self.mode in {"conf_nms", "full"}:
            nms = float(PARAMS["nms_easy"] + self.sci * (PARAMS["nms_hard"] - PARAMS["nms_easy"]))
        if self.mode == "full":
            r0, r1, r2 = RESOLUTIONS
            if self.sci >= float(PARAMS["threshold_high"]):
                size = r2
            elif self.sci >= float(PARAMS["threshold_mid"]):
                size = r1
            else:
                size = r0

        return {
            "conf": float(conf), "nms": float(nms), "size": int(size),
            "sci": float(self.sci), "scene": f"new_ablation_{self.mode}",
        }


def parse_metrics(path: Path, name: str):
    row = next(r for r in csv.DictReader(path.open()) if r["system"] == name)
    return {
        "HOTA": float(row["HOTA"]), "DetA": float(row["DetA"]), "AssA": float(row["AssA"]),
        "MOTA": float(row["MOTA"]), "IDF1": float(row["IDF1"]),
        "IDS": int(float(row["IDS"])), "FN": int(float(row["FN"])), "FP": int(float(row["FP"])),
    }


def write_meta(root: Path, system: dict, stage_label: str, mode: str):
    (root / "configuration.json").write_text(json.dumps({
        "version": "new_acmot_component_ablation_v1",
        "systems": [system],
        "ground_truth_filter": GT_FILTER,
        "evaluation_protocol": "custom class-agnostic AC-MOT research protocol",
        "official_visdrone": False,
        "validation_only": True,
        "test_used": False,
        "stage_label": stage_label,
        "component_mode": mode,
        "frozen_source": str(FROZEN_PATH),
        "static_anchor": ANCHOR,
    }, indent=2))
    (root / "dataset_manifest.json").write_text(json.dumps(VAL_MANIFEST, indent=2))


def fair_run_system(system: dict, output: Path, warm_sizes: list[int], controller_factory, visual_fn):
    """Use exact requested-shape warm-up while preserving measured inference calls."""
    original_controller = pe.PresentationController
    original_visual = pe.visual
    original_detect = pe.detect_realtime

    mapping = {640: warm_sizes[0]}
    if len(warm_sizes) == 3:
        mapping = {640: warm_sizes[0], 736: warm_sizes[1], 832: warm_sizes[2]}

    def detect_with_exact_warm(model, img, size, nms, conf, backend):
        if abs(float(nms) - 0.45) < 1e-12 and abs(float(conf) - 0.19) < 1e-12:
            size = mapping.get(int(size), warm_sizes[0])
        return original_detect(model, img, size, nms, conf, backend)

    try:
        pe.PresentationController = controller_factory
        if visual_fn is not None:
            pe.visual = visual_fn
        pe.detect_realtime = detect_with_exact_warm
        return pe.run_system(
            system, VAL_DIR, VAL_MANIFEST, WEIGHTS, Path("/content/weights/unused.engine"),
            MIN_FPS, PROGRESS_EVERY, "pytorch", 16, output, 1, 1,
        )
    finally:
        pe.PresentationController = original_controller
        pe.visual = original_visual
        pe.detect_realtime = original_detect


def run_stage(stage_label: str, name: str, mode: str, historical: bool = False):
    root = Path("/content/acmot_new_component_ablation") / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)

    if historical:
        system = dict(
            name=name, tracker_profile="tuned",
            adaptive_threshold=True, adaptive_resolution=True,
            smoothing_window=7, analysis_stride=10,
        )
        write_meta(root, system, stage_label, mode)
        timing = fair_run_system(
            system, root, [640, 736, 832], OldPresentationController, None,
        )
    else:
        adaptive_resolution = mode == "full"
        system = dict(
            name=name, tracker_profile="tuned",
            adaptive_threshold=(mode != "static"),
            adaptive_resolution=adaptive_resolution,
            smoothing_window=(SMOOTHING_WINDOW if mode != "static" else 1),
            analysis_stride=(ANALYSIS_STRIDE if mode != "static" else 1),
        )
        write_meta(root, system, stage_label, mode)
        controller_factory = lambda spec: NewAblationController(spec, mode)
        if mode == "static":
            # No scene analysis is required by a static controller.
            visual_fn = lambda _img: {}
            warm_sizes = [ANCHOR_RESOLUTION]
        elif mode in {"conf", "conf_nms"}:
            visual_fn = robust_visual
            warm_sizes = [ANCHOR_RESOLUTION]
        else:
            visual_fn = robust_visual
            warm_sizes = RESOLUTIONS
        timing = fair_run_system(system, root, warm_sizes, controller_factory, visual_fn)

    eval_out = root / "trackeval"
    sh([
        sys.executable, str(ROOT / "evaluate.py"), str(root),
        "--dataset", str(VAL_DIR), "--trackeval", str(TRACKEVAL_DIR),
        "--output", str(eval_out),
    ], cwd=ROOT)
    result = parse_metrics(eval_out / "summary.csv", name)
    result.update({
        "stage": stage_label,
        "system": name,
        "component_mode": mode,
        "FPS": float(timing["processing_fps"]),
        "p95_ms": float(timing["processing_p95_ms"]),
        "mean_imgsz": float(timing["mean_imgsz"]),
        "mean_conf": float(timing["mean_conf"]),
        "mean_nms_iou": float(timing["mean_nms_iou"]),
        "RT25": bool(float(timing["processing_fps"]) >= MIN_FPS),
    })
    (RESULT_ROOT / f"NEW_ABLATION_{stage_label}.json").write_text(json.dumps(result, indent=2))
    shutil.rmtree(root, ignore_errors=True)
    gc.collect(); torch.cuda.empty_cache()
    return result


stages = [
    ("A0", "A0_STATIC_ANCHOR", "static", False),
    ("A1", "A1_ADAPT_CONF", "conf", False),
    ("A2", "A2_ADAPT_CONF_NMS", "conf_nms", False),
    ("A3", "A3_FULL_NEW_ACMOT", "full", False),
]
if INCLUDE_HISTORICAL:
    stages.append(("HIST", "HIST_OLD_ACMOT_W7_S10", "historical_old", True))

csv_path = RESULT_ROOT / "NEW_ACMOT_COMPONENT_ABLATION.csv"
rows = []
if csv_path.exists() and not FORCE:
    try:
        rows = pd.read_csv(csv_path).to_dict("records")
    except Exception:
        rows = []
existing = {str(r.get("stage")): r for r in rows}

for idx, (stage_label, name, mode, historical) in enumerate(stages, 1):
    if stage_label in existing and not FORCE:
        print(f"[ABLATION {idx}/{len(stages)}] reuse {stage_label} {name}", flush=True)
        continue
    print("\n" + "#" * 100, flush=True)
    print(f"NEW ABLATION {idx}/{len(stages)} | {stage_label} | {name}", flush=True)
    print("#" * 100, flush=True)
    result = run_stage(stage_label, name, mode, historical)
    rows = [r for r in rows if str(r.get("stage")) != stage_label]
    rows.append(result)
    existing[stage_label] = result
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(
        f"[{stage_label}] MOTA={100*result['MOTA']:.3f}% HOTA={100*result['HOTA']:.3f}% "
        f"IDF1={100*result['IDF1']:.3f}% IDS={result['IDS']} FPS={result['FPS']:.2f}",
        flush=True,
    )

order = {"A0": 0, "A1": 1, "A2": 2, "A3": 3, "HIST": 4}
rows = sorted([existing[s] for s, *_ in stages if s in existing], key=lambda r: order[str(r["stage"])])
a0 = next(r for r in rows if r["stage"] == "A0")
for r in rows:
    r["delta_MOTA_vs_A0"] = float(r["MOTA"]) - float(a0["MOTA"])
    r["delta_HOTA_vs_A0"] = float(r["HOTA"]) - float(a0["HOTA"])
    r["delta_IDF1_vs_A0"] = float(r["IDF1"]) - float(a0["IDF1"])
    r["delta_IDS_vs_A0"] = int(r["IDS"]) - int(a0["IDS"])
    r["delta_FPS_vs_A0"] = float(r["FPS"]) - float(a0["FPS"])

pd.DataFrame(rows).to_csv(csv_path, index=False)

report = {
    "status": "NEW_COMPONENT_ABLATION_COMPLETE",
    "validation_only": True,
    "test_dev_accessed": False,
    "held_out_lock_present_at_run": False,
    "detector": "fixed pretrained YOLOv8n",
    "tracker": "fixed tuned ByteTrack across A0-A3",
    "frozen_config": str(FROZEN_PATH),
    "cue_calibration": str(CALIBRATION_PATH),
    "definitions": {
        "A0": "Static validation-selected anchor; no adaptive controller.",
        "A1": "A0 + new empirical SCI adaptive confidence; NMS/resolution fixed.",
        "A2": "A1 + adaptive NMS IoU; resolution fixed.",
        "A3": "A2 + adaptive resolution; full new optimized AC-MOT.",
        "HIST": "Original heuristic AC-MOT controller at W=7/S=10; contextual comparator outside A0-A3.",
    },
    "static_anchor": ANCHOR,
    "frozen_temporal": {"smoothing_window": SMOOTHING_WINDOW, "analysis_stride": ANALYSIS_STRIDE},
    "resolution_levels": RESOLUTIONS,
    "selection_note": (
        "This is a post-freeze validation ablation for attribution only. Do not use these results to retune "
        "the already frozen A3 before claiming the held-out final test as unbiased."
    ),
    "results": rows,
    "protocol": "custom class-agnostic AC-MOT evaluation; not official VisDrone leaderboard",
}
report_path = RESULT_ROOT / "NEW_ACMOT_COMPONENT_ABLATION_REPORT.json"
report_path.write_text(json.dumps(report, indent=2))

marker = {
    "status": "DONE",
    "csv": str(csv_path),
    "report": str(report_path),
    "systems": [r["system"] for r in rows],
    "validation_only": True,
    "test_dev_accessed": False,
}
DONE_MARKER.write_text(json.dumps(marker, indent=2))

print("\n" + "=" * 100, flush=True)
print("NEW AC-MOT COMPONENT ABLATION COMPLETE", flush=True)
for r in rows:
    print(
        f"{r['stage']:>4} | {r['system']:<26} | "
        f"MOTA={100*float(r['MOTA']):7.3f}% | HOTA={100*float(r['HOTA']):7.3f}% | "
        f"IDF1={100*float(r['IDF1']):7.3f}% | IDS={int(r['IDS']):4d} | FPS={float(r['FPS']):6.2f}",
        flush=True,
    )
print("CSV   :", csv_path, flush=True)
print("Report:", report_path, flush=True)
print("Marker:", DONE_MARKER, flush=True)
print("Test-dev: NOT ACCESSED", flush=True)
print("Next: review this validation-only attribution study, then run the one-shot held-out test without retuning.")
print("=" * 100, flush=True)
