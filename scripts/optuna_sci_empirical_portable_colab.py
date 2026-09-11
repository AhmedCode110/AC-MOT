"""Portable empirical AC-MOT SCI joint optimizer.

This is the thesis-defense version of the SCI optimization stage. It does not
hard-code the old 640/736/832, confidence, or NMS operating choices.

Inputs produced on validation only
----------------------------------
- SCIENTIFIC_SEARCH_SPACE.json from scientific_operating_ablation_portable_colab.py
- FROZEN_TEMPORAL_CONFIG.json from temporal_ablation_portable_colab.py

What Optuna learns
------------------
- five non-negative SCI weights (normalized to sum to one)
- confidence at SCI=0 and SCI=1, selected only from empirically supported
  confidence values found by the validation screen
- NMS IoU at SCI=0 and SCI=1, selected only from empirically supported values
- two SCI switching thresholds in the mathematical SCI domain [0,1]

The direction of confidence/NMS adaptation is therefore learned rather than
assumed. Three resolution levels come from the resolution ablation.

Scene-cue calibration
---------------------
Brightness, blur and edge scales are empirical validation distributions.
Crowding is calibrated from FIXED-DETECTOR output counts, not ground truth, so
the controller does not need GT at deployment. The 32x32 tiny-box boundary is
kept as an externally documented small-object proxy rather than a learned
number.

Protocol
--------
- validation only; test-dev is never accessed
- fixed pretrained YOLOv8n
- fixed tuned ByteTrack during SCI optimization
- feasibility: FPS >= ACMOT_MIN_FPS and IDS <= old-A3 validation reference
- selection: highest MOTA; tie lower IDS, then HOTA, IDF1, FPS
- local SQLite study is copied to Drive after every trial for cross-account resume
"""
from __future__ import annotations

import csv
import gc
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


ROOT = Path(os.environ.get("ACMOT_REPO", "/content/AC-MOT"))
DRIVE = Path(os.environ.get("ACMOT_DRIVE", "/content/drive/MyDrive"))
DATA_ROOT = Path(os.environ.get("ACMOT_DATA_ROOT", str(DRIVE / "AC-MOT-data")))
RESULT_ROOT = Path(
    os.environ.get("ACMOT_RESULT_ROOT", str(DRIVE / "AC-MOT-results" / "defensible_acmot"))
)
VAL_DIR = Path(os.environ.get("ACMOT_VAL_DIR", str(DATA_ROOT / "VisDrone2019-MOT-val")))
WEIGHTS = Path(os.environ.get("ACMOT_WEIGHTS", "/content/weights/yolov8n.pt"))
TRACKEVAL_DIR = Path(os.environ.get("ACMOT_TRACKEVAL", "/content/TrackEval"))
N_TRIALS = int(os.environ.get("ACMOT_OPTUNA_TRIALS", "50"))
SMOKE = os.environ.get("ACMOT_SMOKE_TEST", "0") == "1"
SEED = int(os.environ.get("ACMOT_SEED", "42"))
MIN_FPS = float(os.environ.get("ACMOT_MIN_FPS", "25"))
PROGRESS_EVERY = int(os.environ.get("ACMOT_PROGRESS_EVERY", "50"))
FORCE_RESET_STUDY = os.environ.get("ACMOT_FORCE_RESET_STUDY", "0") == "1"

SPACE_PATH = RESULT_ROOT / ("SCIENTIFIC_SEARCH_SPACE_SMOKE.json" if SMOKE else "SCIENTIFIC_SEARCH_SPACE.json")
TEMPORAL_PATH = RESULT_ROOT / "FROZEN_TEMPORAL_CONFIG.json"

if not DRIVE.is_dir():
    raise RuntimeError("Mount Google Drive first.")
if not ROOT.is_dir():
    raise RuntimeError(f"Repository not found at {ROOT}")
if not SPACE_PATH.exists():
    raise RuntimeError(
        f"Missing {SPACE_PATH}. Run scientific_operating_ablation_portable_colab.py first."
    )
if not TEMPORAL_PATH.exists() and not SMOKE:
    raise RuntimeError(
        f"Missing {TEMPORAL_PATH}. Run the full temporal ablation before full Optuna."
    )

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

print("[SETUP] Installing pinned runtime dependencies...", flush=True)
sh([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics==8.3.200", "numpy==2.2.6", "scipy==1.15.3",
    "lap", "opencv-python-headless", "optuna>=4,<5", "pandas", "matplotlib",
])


def valid_visdrone(path: Path) -> bool:
    return path.is_dir() and (path / "sequences").is_dir() and (path / "annotations").is_dir()


if not valid_visdrone(VAL_DIR):
    for p in [DRIVE / "VisDrone2019-MOT-val", DATA_ROOT / "VisDrone2019-MOT-val"]:
        if valid_visdrone(p):
            VAL_DIR = p
            break
if not valid_visdrone(VAL_DIR):
    raise RuntimeError("Validation dataset not found. Run the operating ablation setup first.")

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
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from ultralytics import YOLO

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

SPACE = json.loads(SPACE_PATH.read_text())
RESOLUTIONS = [int(x) for x in SPACE["selected_resolution_levels"]]
CONF_CHOICES = [float(x) for x in SPACE["supported_confidence_values"]]
NMS_CHOICES = [float(x) for x in SPACE["supported_nms_values"]]
ANCHOR = SPACE["cue_calibration_anchor"]
if len(RESOLUTIONS) != 3:
    raise RuntimeError(f"Expected exactly three empirical resolution levels, got {RESOLUTIONS}")
if not CONF_CHOICES or not NMS_CHOICES:
    raise RuntimeError("Empirical confidence/NMS choices are empty.")

if TEMPORAL_PATH.exists():
    temporal = json.loads(TEMPORAL_PATH.read_text())["selected"]
    SMOOTHING_WINDOW = int(temporal["smoothing_window"])
    ANALYSIS_STRIDE = int(temporal["analysis_stride"])
else:
    # Pipeline smoke only; never frozen or reported as a scientific selection.
    SMOOTHING_WINDOW = 7
    ANALYSIS_STRIDE = 10

val_names = sorted(p.name for p in (VAL_DIR / "sequences").iterdir() if p.is_dir())
VAL_MANIFEST = dataset_manifest(VAL_DIR, val_names)
GT_FILTER = dict(categories=[1, 4, 5, 6, 9], score=1, occlusion_lt=2, truncation_lt=2)

print("\n" + "=" * 100, flush=True)
print("AC-MOT EMPIRICAL JOINT OPTIMIZATION", flush=True)
print("Validation only   : YES", flush=True)
print("Test-dev          : NOT ACCESSED", flush=True)
print("Resolutions       :", RESOLUTIONS, flush=True)
print("Confidence choices:", CONF_CHOICES, flush=True)
print("NMS choices       :", NMS_CHOICES, flush=True)
print("Temporal          : W=", SMOOTHING_WINDOW, "stride=", ANALYSIS_STRIDE, flush=True)
print("Trials target     :", N_TRIALS, flush=True)
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


def summary(values):
    a = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(a)), "q25": float(np.percentile(a, 25)),
        "median": float(np.median(a)), "q75": float(np.percentile(a, 75)),
        "max": float(np.max(a)),
    }


def calibrate_cues_from_fixed_detector():
    cache = RESULT_ROOT / "DETECTOR_DERIVED_CUE_CALIBRATION.json"
    signature = {
        "analysis_stride": ANALYSIS_STRIDE,
        "anchor": ANCHOR,
        "classes": [0, 2, 5, 7],
        "source": "fixed YOLOv8n detector outputs; no GT counts",
    }
    if cache.exists():
        data = json.loads(cache.read_text())
        if data.get("signature") == signature:
            print("[CACHE] Reusing detector-derived cue calibration", flush=True)
            return data

    detector = YOLO(str(WEIGHTS))
    brightness, blur, edge, crowd = [], [], [], []
    samples = 0
    print("[CALIBRATION] Deriving cue scales from validation images + fixed detector outputs...", flush=True)
    for seq_i, seq in enumerate(VAL_MANIFEST, 1):
        folder = VAL_DIR / "sequences" / seq["sequence"]
        for frame in range(1, int(seq["frames"]) + 1, ANALYSIS_STRIDE):
            path = folder / f"{frame:07d}.jpg"
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"Unreadable image: {path}")
            v = robust_visual(img)
            pred = detector.predict(
                source=img,
                imgsz=int(ANCHOR["resolution"]),
                conf=float(ANCHOR["confidence"]),
                iou=float(ANCHOR["nms_iou"]),
                classes=[0, 2, 5, 7],
                device=0,
                half=True,
                verbose=False,
            )[0]
            brightness.append(v["brightness"])
            blur.append(v["blur"])
            edge.append(v["edges"])
            crowd.append(float(len(pred.boxes)))
            samples += 1
            if samples == 1 or samples % 50 == 0:
                print(f"[CALIBRATION] samples={samples} seq={seq_i}/{len(VAL_MANIFEST)} frame={frame}/{seq['frames']}", flush=True)
    del detector
    torch.cuda.empty_cache()
    out = {
        "signature": signature,
        "sample_count": samples,
        "brightness_sorted": sorted(map(float, brightness)),
        "blur_sorted": sorted(map(float, blur)),
        "edge_sorted": sorted(map(float, edge)),
        "crowd_sorted": sorted(map(float, crowd)),
        "summary": {
            "brightness": summary(brightness), "blur": summary(blur),
            "edge": summary(edge), "crowd_detector_count": summary(crowd),
        },
    }
    cache.write_text(json.dumps(out, indent=2))
    print("[OK] Saved", cache, flush=True)
    return out


CALIBRATION = calibrate_cues_from_fixed_detector()


def empirical_rank(value, sorted_values):
    arr = np.asarray(sorted_values, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.searchsorted(arr, value, side="right") / len(arr))


class EmpiricalSCIController:
    def __init__(self, spec: PresentationSpec, params: dict, calibration: dict):
        self.spec = spec.validate()
        self.params = params
        self.calibration = calibration
        self.history = deque(maxlen=self.spec.smoothing_window)
        self.sci = 0.0
        self.tiny = 0.0
        keys = ["crowd", "tiny", "edge", "night", "blur"]
        raw = np.asarray([float(params[f"weight_{k}"]) for k in keys], dtype=float)
        if raw.sum() <= 0:
            raw[:] = 1.0
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
                "crowd": empirical_rank(n, self.calibration["crowd_sorted"]),
                "tiny": self.tiny,
                "edge": empirical_rank(float(visual["edges"]), self.calibration["edge_sorted"]),
                "night": 1.0 - empirical_rank(float(visual["brightness"]), self.calibration["brightness_sorted"]),
                "blur": 1.0 - empirical_rank(float(visual["blur"]), self.calibration["blur_sorted"]),
            }
            raw_sci = float(np.clip(sum(self.weights[k] * cues[k] for k in self.weights), 0.0, 1.0))
            self.history.append(raw_sci)
            self.sci = float(np.mean(self.history))

        p = self.params
        conf = float(p["conf_easy"] + self.sci * (p["conf_hard"] - p["conf_easy"]))
        nms = float(p["nms_easy"] + self.sci * (p["nms_hard"] - p["nms_easy"]))
        r0, r1, r2 = RESOLUTIONS
        if self.sci >= p["threshold_high"]:
            size = r2
        elif self.sci >= p["threshold_mid"]:
            size = r1
        else:
            size = r0
        return {"conf": conf, "nms": nms, "size": int(size), "sci": float(self.sci), "scene": "empirical_sci"}


A3 = {
    "tracker_profile": "tuned",
    "adaptive_threshold": True,
    "adaptive_resolution": True,
    "smoothing_window": SMOOTHING_WINDOW,
    "analysis_stride": ANALYSIS_STRIDE,
}


def write_meta(root, manifest, systems):
    (root / "configuration.json").write_text(json.dumps({
        "version": "optuna_sci_empirical_portable_v1",
        "systems": systems,
        "ground_truth_filter": GT_FILTER,
        "evaluation_protocol": "custom class-agnostic AC-MOT research protocol",
        "official_visdrone": False,
        "test_used": False,
    }, indent=2))
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_metrics(path, name):
    row = next(r for r in csv.DictReader(path.open()) if r["system"] == name)
    return {
        "HOTA": float(row["HOTA"]), "DetA": float(row["DetA"]), "AssA": float(row["AssA"]),
        "MOTA": float(row["MOTA"]), "IDF1": float(row["IDF1"]),
        "IDS": int(float(row["IDS"])), "FN": int(float(row["FN"])), "FP": int(float(row["FP"])),
    }


def run_one(system, root, controller_factory, visual_fn=None):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    write_meta(root, VAL_MANIFEST, [system])
    old_controller, old_visual = pe.PresentationController, pe.visual
    try:
        pe.PresentationController = controller_factory
        if visual_fn is not None:
            pe.visual = visual_fn
        timing = pe.run_system(
            system, VAL_DIR, VAL_MANIFEST, WEIGHTS, Path("/content/weights/unused.engine"),
            MIN_FPS, PROGRESS_EVERY, "pytorch", 16, root, 1, 1,
        )
    finally:
        pe.PresentationController, pe.visual = old_controller, old_visual
    eval_out = root / "trackeval"
    sh([
        sys.executable, str(ROOT / "evaluate.py"), str(root),
        "--dataset", str(VAL_DIR), "--trackeval", str(TRACKEVAL_DIR),
        "--output", str(eval_out),
    ], cwd=ROOT)
    result = parse_metrics(eval_out / "summary.csv", system["name"])
    result.update(
        FPS=float(timing["processing_fps"]),
        mean_imgsz=float(timing["mean_imgsz"]),
        mean_conf=float(timing["mean_conf"]),
        mean_nms_iou=float(timing["mean_nms_iou"]),
    )
    gc.collect(); torch.cuda.empty_cache()
    return result


OLD_CACHE = RESULT_ROOT / f"OLD_A3_VALIDATION_W{SMOOTHING_WINDOW}_S{ANALYSIS_STRIDE}.json"
if OLD_CACHE.exists():
    old_val = json.loads(OLD_CACHE.read_text())
    print("[REFERENCE] Reusing old-A3 validation reference", flush=True)
else:
    old_system = dict(name="A3_OLD_SCI", **A3)
    old_val = run_one(old_system, Path("/content/old_a3_empirical_reference"), OldPresentationController)
    OLD_CACHE.write_text(json.dumps(old_val, indent=2))
BASELINE_IDS = int(old_val["IDS"])
print("[OLD A3]", json.dumps(old_val, indent=2), flush=True)

WEIGHT_NAMES = ["crowd", "tiny", "edge", "night", "blur"]


def trial_params(trial):
    raw = {k: trial.suggest_float(f"raw_{k}", 0.0, 1.0) for k in WEIGHT_NAMES}
    total = sum(raw.values())
    if total <= 1e-12:
        raw = {k: 1.0 for k in WEIGHT_NAMES}; total = 5.0
    params = {f"weight_{k}": raw[k] / total for k in WEIGHT_NAMES}
    params["conf_easy"] = trial.suggest_categorical("conf_easy", CONF_CHOICES)
    params["conf_hard"] = trial.suggest_categorical("conf_hard", CONF_CHOICES)
    params["nms_easy"] = trial.suggest_categorical("nms_easy", NMS_CHOICES)
    params["nms_hard"] = trial.suggest_categorical("nms_hard", NMS_CHOICES)
    a = trial.suggest_float("threshold_a", 0.0, 1.0)
    b = trial.suggest_float("threshold_b", 0.0, 1.0)
    params["threshold_mid"] = float(min(a, b))
    params["threshold_high"] = float(max(a, b))
    return params


TRIAL_ROOT = Path("/content/acmot_empirical_joint_trials")
TRIAL_ROOT.mkdir(parents=True, exist_ok=True)


def objective(trial):
    params = trial_params(trial)
    system = dict(name=f"EMPIRICAL_{trial.number:03d}", **A3)
    print("\n" + "#" * 100, flush=True)
    print(f"EMPIRICAL OPTUNA TRIAL {trial.number + 1} / target {N_TRIALS}", flush=True)
    print("#" * 100, flush=True)
    try:
        result = run_one(
            system,
            TRIAL_ROOT / system["name"],
            lambda spec: EmpiricalSCIController(spec, params, CALIBRATION),
            robust_visual,
        )
    except Exception as exc:
        trial.set_user_attr("error", repr(exc))
        return -1.0, 10**9, 0.0
    feasible = result["FPS"] >= MIN_FPS and result["IDS"] <= BASELINE_IDS
    trial.set_user_attr("empirical_params", params)
    trial.set_user_attr("feasible", bool(feasible))
    for k, v in result.items():
        if isinstance(v, (int, float, str, bool)):
            trial.set_user_attr(k, v)
    print(
        f"[TRIAL {trial.number:03d}] MOTA={100*result['MOTA']:.3f}% HOTA={100*result['HOTA']:.3f}% "
        f"IDF1={100*result['IDF1']:.3f}% IDS={result['IDS']}/{BASELINE_IDS} "
        f"FPS={result['FPS']:.2f} FEASIBLE={feasible}", flush=True,
    )
    return result["MOTA"], result["IDS"], result["FPS"]


signature_payload = {
    "space": SPACE,
    "temporal": {"smoothing_window": SMOOTHING_WINDOW, "analysis_stride": ANALYSIS_STRIDE},
    "seed": SEED,
    "min_fps": MIN_FPS,
    "algorithm": "empirical_joint_v1",
}
SIGNATURE = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
sig_path = RESULT_ROOT / ("EMPIRICAL_STUDY_SIGNATURE_SMOKE.json" if SMOKE else "EMPIRICAL_STUDY_SIGNATURE.json")
if sig_path.exists() and json.loads(sig_path.read_text()).get("sha256") != SIGNATURE and not FORCE_RESET_STUDY:
    raise RuntimeError("Study search space changed. Set ACMOT_FORCE_RESET_STUDY=1 to start a new study intentionally.")
sig_path.write_text(json.dumps({"sha256": SIGNATURE, "payload": signature_payload}, indent=2))

local_db = Path("/content/acmot_empirical_smoke.db" if SMOKE else "/content/acmot_empirical.db")
drive_db = RESULT_ROOT / ("EMPIRICAL_OPTUNA_SMOKE.db" if SMOKE else "EMPIRICAL_OPTUNA.db")
if FORCE_RESET_STUDY:
    local_db.unlink(missing_ok=True); drive_db.unlink(missing_ok=True)
elif drive_db.exists() and not local_db.exists():
    shutil.copy2(drive_db, local_db)

study_name = "acmot_empirical_smoke" if SMOKE else "acmot_empirical_validation"
study = optuna.create_study(
    directions=["maximize", "minimize", "maximize"],
    sampler=TPESampler(seed=SEED),
    study_name=study_name,
    storage=f"sqlite:///{local_db}",
    load_if_exists=True,
)


def save_progress(study, trial):
    study.trials_dataframe().to_csv(
        RESULT_ROOT / ("EMPIRICAL_OPTUNA_TRIALS_SMOKE.csv" if SMOKE else "EMPIRICAL_OPTUNA_TRIALS.csv"),
        index=False,
    )
    shutil.copy2(local_db, drive_db)


complete_before = [t for t in study.trials if t.state == TrialState.COMPLETE and t.values is not None]
remaining = max(0, N_TRIALS - len(complete_before))
print(f"[RESUME] completed={len(complete_before)} target={N_TRIALS} remaining={remaining}", flush=True)
if remaining:
    study.optimize(objective, n_trials=remaining, gc_after_trial=True, show_progress_bar=True, callbacks=[save_progress])
shutil.copy2(local_db, drive_db)

complete = [
    t for t in study.trials
    if t.state == TrialState.COMPLETE and t.values is not None and float(t.values[0]) >= 0
]
feasible = [
    t for t in complete
    if bool(t.user_attrs.get("feasible", False))
    and float(t.values[2]) >= MIN_FPS
    and int(t.values[1]) <= BASELINE_IDS
]

if SMOKE:
    print("\n" + "=" * 100, flush=True)
    print("EMPIRICAL SMOKE TEST COMPLETE", flush=True)
    print("Completed       :", len(complete), flush=True)
    print("Feasible        :", len(feasible), flush=True)
    print("Frozen config   : NOT CREATED", flush=True)
    print("Test-dev        : NOT ACCESSED", flush=True)
    print("=" * 100, flush=True)
    sys.exit(0)

if not feasible:
    raise RuntimeError("No feasible empirical joint candidate. Inspect EMPIRICAL_OPTUNA_TRIALS.csv before changing methodology.")

best = max(
    feasible,
    key=lambda t: (
        float(t.values[0]), -int(t.values[1]),
        float(t.user_attrs["HOTA"]), float(t.user_attrs["IDF1"]), float(t.values[2]),
    ),
)
BEST_PARAMS = best.user_attrs["empirical_params"]

frozen = {
    "method": "AC-MOT Defensible Empirical Calibration + Joint SCI Optimization",
    "search_split": "VisDrone2019-MOT-val",
    "test_used_during_search": False,
    "detector": "fixed pretrained YOLOv8n",
    "tracker": "fixed tuned ByteTrack",
    "tracker_attribution_note": "ByteTrack is frozen during SCI study; tracker tuning evidence is a separate experiment.",
    "smoothing_window": SMOOTHING_WINDOW,
    "analysis_stride": ANALYSIS_STRIDE,
    "resolution_levels": RESOLUTIONS,
    "supported_confidence_values": CONF_CHOICES,
    "supported_nms_values": NMS_CHOICES,
    "cue_calibration_source": "fixed detector outputs + validation image statistics; no GT crowd counts",
    "tiny_object_definition": "box area < 32x32 pixels; externally documented small-object proxy",
    "maximum_trial_budget": N_TRIALS,
    "selection_rule": "Validation only: FPS gate and IDS<=Old-A3, then highest MOTA; tie lower IDS, HOTA, IDF1, FPS.",
    "old_A3_reference": old_val,
    "selected_trial": int(best.number),
    "validation_result": {
        "MOTA": float(best.values[0]), "IDS": int(best.values[1]), "FPS": float(best.values[2]),
        "HOTA": float(best.user_attrs["HOTA"]), "IDF1": float(best.user_attrs["IDF1"]),
    },
    "optimized_parameters": BEST_PARAMS,
    "operating_ablation": SPACE,
    "cue_calibration_summary": CALIBRATION["summary"],
    "study_signature": SIGNATURE,
    "evaluation_protocol": "Custom class-agnostic AC-MOT TrackEval protocol; not official VisDrone leaderboard.",
}
FROZEN = RESULT_ROOT / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"
FROZEN.write_text(json.dumps(frozen, indent=2))

try:
    from optuna.importance import get_param_importances
    imp = get_param_importances(study, target=lambda t: t.values[0])
    (RESULT_ROOT / "EMPIRICAL_PARAMETER_IMPORTANCE_MOTA.json").write_text(json.dumps(imp, indent=2))
except Exception as exc:
    print("[INFO] Parameter importance skipped:", exc, flush=True)

print("\n" + "=" * 100, flush=True)
print("DEFENSIBLE EMPIRICAL JOINT OPTIMIZATION COMPLETE", flush=True)
print("Selected trial :", best.number, flush=True)
print("MOTA           :", 100 * float(best.values[0]), flush=True)
print("HOTA           :", 100 * float(best.user_attrs["HOTA"]), flush=True)
print("IDF1           :", 100 * float(best.user_attrs["IDF1"]), flush=True)
print("IDS            :", int(best.values[1]), flush=True)
print("FPS            :", float(best.values[2]), flush=True)
print("Frozen config  :", FROZEN, flush=True)
print("Test-dev       : NOT ACCESSED", flush=True)
print("=" * 100, flush=True)
