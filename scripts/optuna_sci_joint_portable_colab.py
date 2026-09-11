"""Portable AC-MOT joint SCI + detector-control Optuna search for Google Colab.

Design goals
------------
- Runs on any Colab account after Google Drive is mounted.
- Uses validation only for optimization; this script NEVER evaluates test-dev.
- Keeps pretrained YOLOv8n and tuned ByteTrack fixed during the search.
- Jointly optimizes:
    * 5 normalized SCI cue weights
    * confidence base + SCI slope
    * NMS IoU base + SCI delta (positive or negative)
    * SCI thresholds for 736 and 832 inference sizes
- Replaces old fixed cue constants (crowd/30, edge/0.14, brightness<80,
  blur<180) with empirical percentile calibration derived from validation.
- Requires FPS >= 25 and IDS <= old A3 validation reference.
- Selects highest validation MOTA among feasible trials; tie-breaks with lower
  IDS, then HOTA, IDF1, and FPS.
- Saves all trial data and convergence evidence for thesis defense.
- Smoke-test mode performs validation trials only and does not freeze a final
  configuration.

Important protocol note
-----------------------
The repository evaluator is a custom class-agnostic AC-MOT research protocol,
not the official VisDrone leaderboard preprocessing protocol.
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


# ---------------------------------------------------------------------------
# Portable paths
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

N_TRIALS = int(os.environ.get("ACMOT_OPTUNA_TRIALS", "50"))
SMOKE_TEST = os.environ.get("ACMOT_SMOKE_TEST", "0") == "1"
SEED = int(os.environ.get("ACMOT_SEED", "42"))
MIN_FPS = float(os.environ.get("ACMOT_MIN_FPS", "25"))
PROGRESS_EVERY = int(os.environ.get("ACMOT_PROGRESS_EVERY", "250"))
ANALYSIS_STRIDE = int(os.environ.get("ACMOT_ANALYSIS_STRIDE", "10"))
SMOOTHING_WINDOW = int(os.environ.get("ACMOT_SMOOTHING_WINDOW", "7"))

SEARCH_SPACE = {
    "raw_weights": (0.02, 1.00),
    "conf_base": (0.18, 0.32),
    "conf_slope": (0.00, 0.15),
    "nms_base": (0.35, 0.65),
    "nms_delta": (-0.15, 0.15),
    "threshold_736": (0.10, 0.60),
    "threshold_gap": (0.05, 0.35),
}
CONF_CLIP = (0.05, 0.50)
NMS_CLIP = (0.25, 0.80)
GT_CATEGORIES = [1, 4, 5, 6, 9]


# ---------------------------------------------------------------------------
# Fresh-Colab setup
# ---------------------------------------------------------------------------
if not DRIVE.is_dir():
    raise RuntimeError(
        "Google Drive is not mounted. In the Colab notebook run:\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "then rerun this script."
    )
if not ROOT.is_dir():
    raise RuntimeError(
        f"Repository not found at {ROOT}. Clone it first with:\n"
        f"!git clone https://github.com/AhmedCode110/AC-MOT.git {ROOT}"
    )

DATA_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

print("[SETUP] Installing pinned runtime dependencies...")
sh([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics==8.3.200", "numpy==2.2.6", "scipy==1.15.3",
    "lap", "opencv-python-headless", "optuna>=4,<5", "pandas",
    "matplotlib", "gdown",
])


def valid_visdrone(path: Path) -> bool:
    return path.is_dir() and (path / "sequences").is_dir() and (path / "annotations").is_dir()


# Search common Drive locations before downloading.
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

# Public mirror fallback so a new account is self-contained.
if not valid_visdrone(VAL_DIR):
    print("[DATA] Validation set not found; downloading portable validation copy...")
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
    found = [p for p in DATA_ROOT.rglob("*") if valid_visdrone(p) and "VisDrone2019-MOT-val" in p.name]
    if len(found) == 1:
        VAL_DIR = found[0]
    elif valid_visdrone(DATA_ROOT / "VisDrone2019-MOT-val"):
        VAL_DIR = DATA_ROOT / "VisDrone2019-MOT-val"
    else:
        raise RuntimeError("Downloaded validation archive but could not identify its extracted folder.")

# Fixed detector download.
if not WEIGHTS.exists():
    print("[SETUP] Downloading fixed YOLOv8n weights...")
    from ultralytics import YOLO
    old = os.getcwd()
    os.chdir(WEIGHTS.parent)
    try:
        YOLO("yolov8n.pt")
    finally:
        os.chdir(old)
if not WEIGHTS.exists():
    raise RuntimeError("Could not obtain fixed yolov8n.pt")

# Pinned TrackEval.
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

# Imports after installation.
import cv2
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import boxes
from core_v17 import PresentationController as OldPresentationController, PresentationSpec
from experiment import dataset_manifest
import scripts.paper_eval_v17 as pe

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required. Select a GPU runtime in Colab.")
print("[GPU]", torch.cuda.get_device_name(0))
if "T4" not in torch.cuda.get_device_name(0):
    print("[WARNING] GPU is not Tesla T4; FPS is not directly comparable with prior T4 runs.")

print("\n" + "=" * 88)
print("AC-MOT PORTABLE JOINT OPTIMIZATION")
print("=" * 88)
print("Validation :", VAL_DIR)
print("Results    :", RESULT_ROOT)
print("Detector   : FIXED YOLOv8n")
print("Tracker    : FIXED tuned ByteTrack")
print("Trials     :", N_TRIALS)
print("Smoke test :", SMOKE_TEST)
print("FPS gate   : >=", MIN_FPS)
print("Test-dev   : NOT ACCESSED BY THIS SCRIPT")
print("Protocol   : custom class-agnostic AC-MOT evaluation; NOT official VisDrone leaderboard")
print("=" * 88)

val_names = sorted(p.name for p in (VAL_DIR / "sequences").iterdir() if p.is_dir())
VAL_MANIFEST = dataset_manifest(VAL_DIR, val_names)
print("[OK] Validation sequences:", len(VAL_MANIFEST))


# ---------------------------------------------------------------------------
# Validation-derived continuous scene cues
# ---------------------------------------------------------------------------
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


def load_gt(sequence_name):
    ann = VAL_DIR / "annotations" / f"{sequence_name}.txt"
    gt = np.loadtxt(ann, delimiter=",", ndmin=2)
    keep = (
        np.isin(gt[:, 7], GT_CATEGORIES)
        & (gt[:, 6] == 1)
        & (gt[:, 8] < 2)
        & (gt[:, 9] < 2)
    )
    return gt[keep]


def calibrate_validation_cues():
    path = RESULT_ROOT / "VALIDATION_CUE_CALIBRATION.json"
    if path.exists():
        print("[CACHE] Loading validation cue calibration")
        return json.loads(path.read_text())

    brightness, blur, edge, crowd = [], [], [], []
    samples = 0
    print("\n[CALIBRATION] Deriving continuous cue distributions from validation only...")
    for seq_i, seq in enumerate(VAL_MANIFEST, 1):
        gt = load_gt(seq["sequence"])
        folder = VAL_DIR / "sequences" / seq["sequence"]
        for frame in range(1, seq["frames"] + 1, ANALYSIS_STRIDE):
            img = cv2.imread(str(folder / f"{frame:07d}.jpg"))
            if img is None:
                raise RuntimeError(f"Unreadable validation image: {folder / f'{frame:07d}.jpg'}")
            v = robust_visual(img)
            brightness.append(v["brightness"])
            blur.append(v["blur"])
            edge.append(v["edges"])
            crowd.append(float(np.sum(gt[:, 0] == frame)))
            samples += 1
            if samples == 1 or samples % 50 == 0:
                print(f"[CALIBRATION] samples={samples} seq={seq_i}/{len(VAL_MANIFEST)} frame={frame}/{seq['frames']}")

    def summary(values):
        a = np.asarray(values, dtype=float)
        return {
            "min": float(np.min(a)), "q25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "q75": float(np.percentile(a, 75)),
            "max": float(np.max(a)),
        }

    out = {
        "analysis_stride": ANALYSIS_STRIDE,
        "sample_count": samples,
        "brightness_sorted": sorted(map(float, brightness)),
        "blur_sorted": sorted(map(float, blur)),
        "edge_sorted": sorted(map(float, edge)),
        "crowd_sorted": sorted(map(float, crowd)),
        "summary": {
            "brightness": summary(brightness),
            "blur": summary(blur),
            "edge": summary(edge),
            "crowd_count": summary(crowd),
        },
    }
    path.write_text(json.dumps(out, indent=2))
    print("[OK] Saved cue calibration:", path)
    return out


CALIBRATION = calibrate_validation_cues()


def empirical_rank(value, sorted_values):
    arr = np.asarray(sorted_values, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.searchsorted(arr, value, side="right") / len(arr))


class JointSCIController:
    def __init__(self, spec: PresentationSpec, params: dict, calibration: dict):
        self.spec = spec.validate()
        self.params = params
        self.calibration = calibration
        self.history = deque(maxlen=self.spec.smoothing_window)
        self.sci = 0.0
        self.tiny = 0.0
        keys = ["crowd", "tiny", "edge", "night", "blur"]
        raw = np.asarray([params[f"weight_{k}"] for k in keys], dtype=float)
        if np.any(raw < 0) or raw.sum() <= 0:
            raise ValueError("Invalid SCI weights")
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
        conf = float(np.clip(p["conf_base"] - p["conf_slope"] * self.sci, *CONF_CLIP))
        nms = float(np.clip(p["nms_base"] + p["nms_delta"] * self.sci, *NMS_CLIP))
        size = 832 if self.sci >= p["threshold_832"] else 736 if self.sci >= p["threshold_736"] else 640
        return {"conf": conf, "nms": nms, "size": size, "sci": float(self.sci), "scene": "continuous_sci"}


A3 = {
    "tracker_profile": "tuned",
    "adaptive_threshold": True,
    "adaptive_resolution": True,
    "smoothing_window": SMOOTHING_WINDOW,
    "analysis_stride": ANALYSIS_STRIDE,
}
GT_FILTER = dict(categories=GT_CATEGORIES, score=1, occlusion_lt=2, truncation_lt=2)


def write_meta(root, manifest, systems):
    (root / "configuration.json").write_text(json.dumps({
        "version": "optuna_sci_joint_portable",
        "systems": systems,
        "ground_truth_filter": GT_FILTER,
        "evaluation_protocol": "custom class-agnostic AC-MOT research protocol",
        "official_visdrone": False,
    }, indent=2))
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_metrics(path, name):
    row = next(r for r in csv.DictReader(path.open()) if r["system"] == name)
    return {
        "HOTA": float(row["HOTA"]), "DetA": float(row["DetA"]), "AssA": float(row["AssA"]),
        "MOTA": float(row["MOTA"]), "IDF1": float(row["IDF1"]),
        "IDS": int(float(row["IDS"])), "FN": int(float(row["FN"])), "FP": int(float(row["FP"])),
    }


def run_one(dataset, manifest, system, root, controller_factory, visual_fn=None):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    write_meta(root, manifest, [system])
    old_controller, old_visual = pe.PresentationController, pe.visual
    try:
        pe.PresentationController = controller_factory
        if visual_fn is not None:
            pe.visual = visual_fn
        timing = pe.run_system(
            system, dataset, manifest, WEIGHTS, Path("/content/weights/unused.engine"),
            MIN_FPS, PROGRESS_EVERY, "pytorch", 16, root, 1, 1,
        )
    finally:
        pe.PresentationController, pe.visual = old_controller, old_visual

    eval_out = root / "trackeval"
    sh([
        sys.executable, str(ROOT / "evaluate.py"), str(root),
        "--dataset", str(dataset), "--trackeval", str(TRACKEVAL_DIR),
        "--output", str(eval_out),
    ], cwd=ROOT)
    result = parse_metrics(eval_out / "summary.csv", system["name"])
    result.update(
        FPS=float(timing["processing_fps"]),
        mean_imgsz=float(timing["mean_imgsz"]),
        mean_conf=float(timing["mean_conf"]),
        mean_nms_iou=float(timing["mean_nms_iou"]),
    )
    (root / "compact_result.json").write_text(json.dumps(result, indent=2))
    gc.collect(); torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Old A3 reference (validation only)
# ---------------------------------------------------------------------------
OLD_CACHE = RESULT_ROOT / "OLD_A3_VALIDATION.json"
if OLD_CACHE.exists():
    old_val = json.loads(OLD_CACHE.read_text())
    print("[REFERENCE] Reusing cached Old A3 validation")
else:
    old_system = dict(name="A3_OLD_SCI", **A3)
    old_val = run_one(
        VAL_DIR, VAL_MANIFEST, old_system, Path("/content/old_a3_joint_reference"),
        OldPresentationController,
    )
    OLD_CACHE.write_text(json.dumps(old_val, indent=2))
BASELINE_IDS = int(old_val["IDS"])
print("\n=== OLD A3 VALIDATION REFERENCE ===")
print(json.dumps(old_val, indent=2))


# ---------------------------------------------------------------------------
# Joint Optuna search
# ---------------------------------------------------------------------------
WEIGHT_NAMES = ["crowd", "tiny", "edge", "night", "blur"]


def trial_joint_params(trial):
    raw = {k: trial.suggest_float(f"raw_{k}", *SEARCH_SPACE["raw_weights"]) for k in WEIGHT_NAMES}
    total = sum(raw.values())
    params = {f"weight_{k}": raw[k] / total for k in WEIGHT_NAMES}
    params["conf_base"] = trial.suggest_float("conf_base", *SEARCH_SPACE["conf_base"])
    params["conf_slope"] = trial.suggest_float("conf_slope", *SEARCH_SPACE["conf_slope"])
    params["nms_base"] = trial.suggest_float("nms_base", *SEARCH_SPACE["nms_base"])
    params["nms_delta"] = trial.suggest_float("nms_delta", *SEARCH_SPACE["nms_delta"])
    t736 = trial.suggest_float("threshold_736", *SEARCH_SPACE["threshold_736"])
    gap = trial.suggest_float("threshold_gap", *SEARCH_SPACE["threshold_gap"])
    params["threshold_736"] = float(t736)
    params["threshold_832"] = float(min(0.95, t736 + gap))
    return params


TRIAL_ROOT = Path("/content/acmot_optuna_joint_trials")
if TRIAL_ROOT.exists():
    shutil.rmtree(TRIAL_ROOT)
TRIAL_ROOT.mkdir(parents=True)


def objective(trial):
    params = trial_joint_params(trial)
    system = dict(name=f"JOINT_{trial.number:03d}", **A3)
    print("\n" + "#" * 88)
    print(f"JOINT OPTUNA TRIAL {trial.number + 1}/{N_TRIALS}")
    print("#" * 88)
    try:
        result = run_one(
            VAL_DIR, VAL_MANIFEST, system, TRIAL_ROOT / system["name"],
            lambda spec: JointSCIController(spec, params, CALIBRATION),
            robust_visual,
        )
    except Exception as exc:
        trial.set_user_attr("error", repr(exc))
        return -1.0, 10**9, 0.0

    feasible = result["FPS"] >= MIN_FPS and result["IDS"] <= BASELINE_IDS
    trial.set_user_attr("joint_params", params)
    trial.set_user_attr("feasible", bool(feasible))
    for k, v in result.items():
        if isinstance(v, (int, float, str, bool)):
            trial.set_user_attr(k, v)
    print(
        f"[TRIAL {trial.number:03d}] MOTA={100*result['MOTA']:.3f} "
        f"HOTA={100*result['HOTA']:.3f} IDF1={100*result['IDF1']:.3f} "
        f"IDS={result['IDS']}/{BASELINE_IDS} FPS={result['FPS']:.2f} FEASIBLE={feasible}"
    )
    return result["MOTA"], result["IDS"], result["FPS"]


study = optuna.create_study(
    directions=["maximize", "minimize", "maximize"],
    sampler=TPESampler(seed=SEED),
    study_name="acmot_joint_validation",
)


def save_progress(study, trial):
    study.trials_dataframe().to_csv(RESULT_ROOT / "JOINT_OPTUNA_TRIALS.csv", index=False)
    rows, best_mota, best_trial = [], None, None
    for t in study.trials:
        if t.state != TrialState.COMPLETE or t.values is None or t.values[0] < 0:
            continue
        feasible = bool(t.user_attrs.get("feasible", False))
        if feasible and (best_mota is None or float(t.values[0]) > best_mota):
            best_mota, best_trial = float(t.values[0]), int(t.number)
        rows.append({
            "trial": int(t.number), "MOTA": float(t.values[0]), "IDS": int(t.values[1]),
            "FPS": float(t.values[2]), "feasible": feasible,
            "best_feasible_MOTA_so_far": best_mota,
            "best_feasible_trial_so_far": best_trial,
        })
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "CONVERGENCE.csv", index=False)


study.optimize(
    objective, n_trials=N_TRIALS, gc_after_trial=True,
    show_progress_bar=True, callbacks=[save_progress],
)

complete = [
    t for t in study.trials
    if t.state == TrialState.COMPLETE and t.values is not None and t.values[0] >= 0
]
feasible = [
    t for t in complete
    if float(t.values[2]) >= MIN_FPS and int(t.values[1]) <= BASELINE_IDS
]

if SMOKE_TEST:
    print("\n" + "=" * 88)
    print("SMOKE TEST COMPLETE")
    print("Trials completed :", len(complete))
    print("Feasible trials  :", len(feasible))
    print("Test-dev         : NOT ACCESSED")
    print("Frozen config    : NOT CREATED")
    print("=" * 88)
    sys.exit(0)

if not feasible:
    raise RuntimeError(
        "No feasible joint candidate found. Inspect JOINT_OPTUNA_TRIALS.csv before changing the search space."
    )

best = max(
    feasible,
    key=lambda t: (
        float(t.values[0]), -int(t.values[1]),
        float(t.user_attrs["HOTA"]), float(t.user_attrs["IDF1"]), float(t.values[2]),
    ),
)
BEST_PARAMS = best.user_attrs["joint_params"]

frozen = {
    "method": "AC-MOT Joint SCI Optimization",
    "search_split": "VisDrone2019-MOT-val",
    "test_used_during_search": False,
    "detector": "fixed YOLOv8n",
    "tracker": "fixed tuned ByteTrack",
    "analysis_stride": ANALYSIS_STRIDE,
    "smoothing_window": SMOOTHING_WINDOW,
    "maximum_trial_budget": N_TRIALS,
    "selection_rule": (
        "Validation only: FPS>=25 and IDS<=Old-A3; then highest MOTA; "
        "tie-break lower IDS, then HOTA, IDF1, FPS."
    ),
    "old_A3_reference": old_val,
    "selected_trial": int(best.number),
    "validation_result": {
        "MOTA": float(best.values[0]), "IDS": int(best.values[1]), "FPS": float(best.values[2]),
        "HOTA": float(best.user_attrs["HOTA"]), "IDF1": float(best.user_attrs["IDF1"]),
    },
    "optimized_parameters": BEST_PARAMS,
    "cue_calibration_summary": CALIBRATION["summary"],
    "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
    "evaluation_protocol": "Custom class-agnostic AC-MOT TrackEval protocol; not official VisDrone leaderboard.",
}
FROZEN_PATH = RESULT_ROOT / "FROZEN_VALIDATION_CONFIG.json"
FROZEN_PATH.write_text(json.dumps(frozen, indent=2))

# Parameter importance (best-effort for multi-objective study, MOTA target).
try:
    from optuna.importance import get_param_importances
    importance = get_param_importances(study, target=lambda t: t.values[0])
    (RESULT_ROOT / "PARAMETER_IMPORTANCE_MOTA.json").write_text(json.dumps(importance, indent=2))
except Exception as exc:
    print("[INFO] Parameter importance skipped:", exc)

# Convergence plot.
try:
    import matplotlib.pyplot as plt
    conv = pd.read_csv(RESULT_ROOT / "CONVERGENCE.csv")
    if len(conv):
        plt.figure(figsize=(8, 5))
        plt.plot(conv["trial"], conv["best_feasible_MOTA_so_far"], marker="o")
        plt.xlabel("Optuna Trial")
        plt.ylabel("Best Feasible Validation MOTA")
        plt.title("AC-MOT Joint Optimization Convergence")
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / "CONVERGENCE.png", dpi=200)
        plt.close()
except Exception as exc:
    print("[INFO] Convergence plot skipped:", exc)

METHOD_AUDIT = {
    "learned_from_validation": [
        "5 SCI weights", "confidence base", "confidence SCI slope",
        "NMS IoU base", "NMS IoU SCI direction/magnitude",
        "SCI threshold for 736", "SCI threshold for 832",
    ],
    "derived_from_validation_statistics": [
        "crowding percentile scale", "edge-complexity percentile scale",
        "brightness percentile scale", "blur percentile scale",
    ],
    "fixed_for_experimental_isolation": [
        "YOLOv8n detector", "YOLO weights", "tuned ByteTrack configuration",
        "candidate resolutions 640/736/832", "analysis stride", "smoothing window",
        "tiny-object area boundary 32x32", "TrackEval revision",
    ],
    "requires_separate_ablation_or documented_rationale": [
        "analysis stride", "smoothing window", "tuned ByteTrack parameters",
    ],
    "protocol_limitation": (
        "Custom class-agnostic research evaluation; not the official VisDrone leaderboard protocol."
    ),
}
(RESULT_ROOT / "METHOD_AUDIT.json").write_text(json.dumps(METHOD_AUDIT, indent=2))

print("\n" + "=" * 88)
print("JOINT OPTIMIZATION COMPLETE")
print("Selected trial :", best.number)
print("MOTA           :", 100 * float(best.values[0]))
print("IDS            :", int(best.values[1]))
print("FPS            :", float(best.values[2]))
print("HOTA           :", 100 * float(best.user_attrs["HOTA"]))
print("IDF1           :", 100 * float(best.user_attrs["IDF1"]))
print("Frozen config  :", FROZEN_PATH)
print("IMPORTANT      : Test-dev has NOT been evaluated by this script.")
print("=" * 88)
