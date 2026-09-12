"""AC-MOT V2 validation-only Pareto optimization: maximize MOTA, minimize IDS.

V2 deliberately reuses the frozen V1 validation design and changes the optimization
and selection formulation only:
- objectives: maximize MOTA, minimize IDS
- feasibility gate: processing FPS >= 25
- no IDS<=Old-A3 gate
- no held-out test-dev access
- frozen V1 operating search space, temporal design, and cue calibration are inputs
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


ROOT = Path(os.environ.get("ACMOT_REPO", "/content/AC-MOT"))
DRIVE = Path(os.environ.get("ACMOT_DRIVE", "/content/drive/MyDrive"))
DATA_ROOT = Path(os.environ.get("ACMOT_DATA_ROOT", str(DRIVE / "AC-MOT-data")))
V1_ROOT = Path(
    os.environ.get(
        "ACMOT_V1_RESULT_ROOT",
        str(DRIVE / "AC-MOT-shared" / "defensible_acmot_3workers"),
    )
)
V2_ROOT = Path(
    os.environ.get(
        "ACMOT_V2_RESULT_ROOT",
        str(DRIVE / "AC-MOT-results" / "V2_MULTI_OBJECTIVE_MOTA_IDS"),
    )
)
VAL_DIR = Path(os.environ.get("ACMOT_VAL_DIR", str(DATA_ROOT / "VisDrone2019-MOT-val")))
WEIGHTS = Path(os.environ.get("ACMOT_WEIGHTS", "/content/weights/yolov8n.pt"))
TRACKEVAL_DIR = Path(os.environ.get("ACMOT_TRACKEVAL", "/content/TrackEval"))

SMOKE = os.environ.get("ACMOT_V2_SMOKE_TEST", "0") == "1"
N_TRIALS = int(os.environ.get("ACMOT_V2_OPTUNA_TRIALS", "3" if SMOKE else "50"))
SEED = int(os.environ.get("ACMOT_V2_SEED", "42"))
MIN_FPS = float(os.environ.get("ACMOT_V2_MIN_FPS", "25"))
PROGRESS_EVERY = int(os.environ.get("ACMOT_PROGRESS_EVERY", "50"))
FORCE_RESET_STUDY = os.environ.get("ACMOT_V2_FORCE_RESET_STUDY", "0") == "1"

RUN_ROOT = V2_ROOT / "SMOKE" if SMOKE else V2_ROOT

SPACE_PATH = V1_ROOT / "SCIENTIFIC_SEARCH_SPACE.json"
TEMPORAL_PATH = V1_ROOT / "FROZEN_TEMPORAL_CONFIG.json"
CALIBRATION_PATH = V1_ROOT / "DETECTOR_DERIVED_CUE_CALIBRATION.json"
V1_FROZEN_PATH = V1_ROOT / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"
V1_TRIALS_PATH = V1_ROOT / "EMPIRICAL_OPTUNA_TRIALS.csv"

REQUIRED_INPUTS = {
    "SCIENTIFIC_SEARCH_SPACE.json": SPACE_PATH,
    "FROZEN_TEMPORAL_CONFIG.json": TEMPORAL_PATH,
    "DETECTOR_DERIVED_CUE_CALIBRATION.json": CALIBRATION_PATH,
    "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json": V1_FROZEN_PATH,
}

if not DRIVE.is_dir():
    raise RuntimeError("Mount Google Drive first.")
if not ROOT.is_dir():
    raise RuntimeError(f"Repository not found at {ROOT}")
for name, path in REQUIRED_INPUTS.items():
    if not path.is_file():
        raise RuntimeError(f"Missing frozen V1 input {name}: {path}")

v1_resolved = V1_ROOT.resolve()
v2_resolved = V2_ROOT.resolve()
if v2_resolved == v1_resolved or v1_resolved in v2_resolved.parents:
    raise RuntimeError(
        f"Unsafe V2 output root {V2_ROOT}: it must be separate from and outside V1 root {V1_ROOT}."
    )

RUN_ROOT.mkdir(parents=True, exist_ok=True)
(RUN_ROOT / "plots").mkdir(exist_ok=True)
(RUN_ROOT / "logs").mkdir(exist_ok=True)

if "test-dev" in str(VAL_DIR).lower():
    raise RuntimeError(f"V2 refuses held-out test-dev path: {VAL_DIR}")

print("[SETUP] Installing V1-compatible runtime dependencies...", flush=True)
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
    raise RuntimeError("VisDrone2019-MOT-val dataset not found.")
if "test-dev" in str(VAL_DIR.resolve()).lower():
    raise RuntimeError(f"V2 refuses held-out test-dev path: {VAL_DIR.resolve()}")

if not WEIGHTS.exists():
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

import cv2
import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from ultralytics import YOLO

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import boxes
from core_v17 import PresentationSpec
from experiment import dataset_manifest
import scripts.paper_eval_v17 as pe

print("[OPTUNA]", optuna.__version__, flush=True)
if not str(optuna.__version__).startswith("4."):
    raise RuntimeError(f"V2 expects Optuna 4.x for V1-compatible methodology, got {optuna.__version__}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")
GPU_NAME = torch.cuda.get_device_name(0)
print("[GPU]", GPU_NAME, flush=True)
if "T4" not in GPU_NAME:
    print("[WARNING] Non-T4 GPU: FPS is not directly comparable with the V1 T4 reference.", flush=True)

SPACE = json.loads(SPACE_PATH.read_text())
TEMPORAL = json.loads(TEMPORAL_PATH.read_text())
CALIBRATION = json.loads(CALIBRATION_PATH.read_text())
V1_FROZEN = json.loads(V1_FROZEN_PATH.read_text())

RESOLUTIONS = [int(x) for x in SPACE["selected_resolution_levels"]]
CONF_CHOICES = [float(x) for x in SPACE["supported_confidence_values"]]
NMS_CHOICES = [float(x) for x in SPACE["supported_nms_values"]]
if len(RESOLUTIONS) != 3:
    raise RuntimeError(f"Expected exactly three frozen V1 resolution levels, got {RESOLUTIONS}")
if not CONF_CHOICES or not NMS_CHOICES:
    raise RuntimeError("Frozen V1 confidence/NMS choices are empty.")

temporal = TEMPORAL["selected"]
SMOOTHING_WINDOW = int(temporal["smoothing_window"])
ANALYSIS_STRIDE = int(temporal["analysis_stride"])
if (SMOOTHING_WINDOW, ANALYSIS_STRIDE) != (7, 10):
    raise RuntimeError(
        f"V2 controlled comparison expects frozen W=7/S=10; got W={SMOOTHING_WINDOW}/S={ANALYSIS_STRIDE}"
    )

INPUT_HASHES = {name: sha256_file(path) for name, path in REQUIRED_INPUTS.items()}
if V1_TRIALS_PATH.is_file():
    INPUT_HASHES["EMPIRICAL_OPTUNA_TRIALS.csv"] = sha256_file(V1_TRIALS_PATH)

EXPECTED_CANONICAL_HASHES = {
    "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json": "8eeb7b916e7085b290313349cb0ebb95fa97f7f3956caefedd902fcf2890379c",
    "DETECTOR_DERIVED_CUE_CALIBRATION.json": "fd42b22987365236944e90d3ebd646a28a56e5b60986cf1ad9f05881e7958c78",
}
for name, expected in EXPECTED_CANONICAL_HASHES.items():
    observed = INPUT_HASHES[name]
    if observed != expected:
        raise RuntimeError(
            f"Frozen V1 input hash mismatch for {name}: expected {expected}, observed {observed}"
        )

(RUN_ROOT / "V2_INPUT_HASHES.json").write_text(json.dumps(INPUT_HASHES, indent=2))

SEARCH_SPACE_SNAPSHOT = {
    "source": "Frozen V1 validation artifacts",
    "resolution_levels": RESOLUTIONS,
    "supported_confidence_values": CONF_CHOICES,
    "supported_nms_values": NMS_CHOICES,
    "smoothing_window": SMOOTHING_WINDOW,
    "analysis_stride": ANALYSIS_STRIDE,
    "weight_parameterization": "five raw non-negative weights normalized to sum to one",
    "threshold_parameterization": "two floats in [0,1], sorted into threshold_mid/high",
}
(RUN_ROOT / "V2_SEARCH_SPACE.json").write_text(json.dumps(SEARCH_SPACE_SNAPSHOT, indent=2))

try:
    CODE_COMMIT = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    CODE_COMMIT = "UNKNOWN"

PROTOCOL = {
    "protocol_version": "acmot_v2_multiobjective_mota_ids_validation_v1",
    "status": "SMOKE" if SMOKE else "FULL_VALIDATION",
    "validation_only": True,
    "held_out_test_accessed": False,
    "dataset": "VisDrone2019-MOT-val",
    "detector": "fixed pretrained YOLOv8n",
    "tracker": "fixed tuned ByteTrack",
    "objectives": ["maximize MOTA", "minimize IDS"],
    "feasibility_gate": {"processing_fps_gte": MIN_FPS},
    "explicitly_removed_v1_gate": "IDS <= Old-A3 is NOT a V2 constraint",
    "sampler": "TPESampler",
    "seed": SEED,
    "trial_budget": N_TRIALS,
    "smoothing_window": SMOOTHING_WINDOW,
    "analysis_stride": ANALYSIS_STRIDE,
    "pinned_trackeval_commit": PINNED_TRACKEVAL,
    "required_reference_gpu": "NVIDIA T4",
    "runtime_gpu": GPU_NAME,
    "code_commit_at_run": CODE_COMMIT,
    "v1_frozen_reference_commit": "a6c1fa49fce1d402513c2df05b7d04b962a6e89e",
    "input_hashes": INPUT_HASHES,
    "evaluation_protocol": "Custom class-agnostic AC-MOT TrackEval protocol; not official VisDrone leaderboard.",
    "balanced_selection": {
        "scope": "feasible Pareto trials only",
        "score": "0.5*MOTA_norm + 0.5*IDS_good_norm",
        "tie_breakers": ["higher HOTA", "higher IDF1", "higher FPS", "lower trial number"],
    },
}
(RUN_ROOT / "V2_PROTOCOL.json").write_text(json.dumps(PROTOCOL, indent=2))

val_names = sorted(p.name for p in (VAL_DIR / "sequences").iterdir() if p.is_dir())
VAL_MANIFEST = dataset_manifest(VAL_DIR, val_names)
GT_FILTER = dict(categories=[1, 4, 5, 6, 9], score=1, occlusion_lt=2, truncation_lt=2)

print("\n" + "=" * 100, flush=True)
print("AC-MOT V2 MULTI-OBJECTIVE VALIDATION", flush=True)
print("V2 VALIDATION ONLY", flush=True)
print("TEST-DEV NOT ACCESSED", flush=True)
print("Objectives        : maximize MOTA, minimize IDS", flush=True)
print("Feasibility       : FPS >=", MIN_FPS, flush=True)
print("IDS<=271 gate     : REMOVED", flush=True)
print("Sampler / seed    : TPESampler /", SEED, flush=True)
print("Trials target     :", N_TRIALS, flush=True)
print("Temporal          : W=", SMOOTHING_WINDOW, "stride=", ANALYSIS_STRIDE, flush=True)
print("Resolutions       :", RESOLUTIONS, flush=True)
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
                "night": 1.0 - empirical_rank(
                    float(visual["brightness"]), self.calibration["brightness_sorted"]
                ),
                "blur": 1.0 - empirical_rank(
                    float(visual["blur"]), self.calibration["blur_sorted"]
                ),
            }
            raw_sci = float(
                np.clip(sum(self.weights[k] * cues[k] for k in self.weights), 0.0, 1.0)
            )
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
        return {
            "conf": conf,
            "nms": nms,
            "size": int(size),
            "sci": float(self.sci),
            "scene": "empirical_sci_v2",
        }


A3 = {
    "tracker_profile": "tuned",
    "adaptive_threshold": True,
    "adaptive_resolution": True,
    "smoothing_window": SMOOTHING_WINDOW,
    "analysis_stride": ANALYSIS_STRIDE,
}


def write_meta(root, manifest, systems):
    (root / "configuration.json").write_text(
        json.dumps(
            {
                "version": "acmot_v2_multiobjective_validation_v1",
                "systems": systems,
                "ground_truth_filter": GT_FILTER,
                "evaluation_protocol": "custom class-agnostic AC-MOT research protocol",
                "official_visdrone": False,
                "test_used": False,
                "objectives": ["maximize MOTA", "minimize IDS"],
                "feasibility": f"FPS >= {MIN_FPS}",
            },
            indent=2,
        )
    )
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_metrics(path, name):
    row = next(r for r in csv.DictReader(path.open()) if r["system"] == name)
    return {
        "HOTA": float(row["HOTA"]),
        "DetA": float(row["DetA"]),
        "AssA": float(row["AssA"]),
        "MOTA": float(row["MOTA"]),
        "IDF1": float(row["IDF1"]),
        "IDS": int(float(row["IDS"])),
        "FN": int(float(row["FN"])),
        "FP": int(float(row["FP"])),
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
            system,
            VAL_DIR,
            VAL_MANIFEST,
            WEIGHTS,
            Path("/content/weights/unused.engine"),
            MIN_FPS,
            PROGRESS_EVERY,
            "pytorch",
            16,
            root,
            1,
            1,
        )
    finally:
        pe.PresentationController, pe.visual = old_controller, old_visual

    eval_out = root / "trackeval"
    sh(
        [
            sys.executable,
            str(ROOT / "evaluate.py"),
            str(root),
            "--dataset",
            str(VAL_DIR),
            "--trackeval",
            str(TRACKEVAL_DIR),
            "--output",
            str(eval_out),
        ],
        cwd=ROOT,
    )
    result = parse_metrics(eval_out / "summary.csv", system["name"])
    result.update(
        FPS=float(timing["processing_fps"]),
        mean_imgsz=float(timing["mean_imgsz"]),
        mean_conf=float(timing["mean_conf"]),
        mean_nms_iou=float(timing["mean_nms_iou"]),
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


WEIGHT_NAMES = ["crowd", "tiny", "edge", "night", "blur"]


def trial_params(trial):
    raw = {k: trial.suggest_float(f"raw_{k}", 0.0, 1.0) for k in WEIGHT_NAMES}
    total = sum(raw.values())
    if total <= 1e-12:
        raw = {k: 1.0 for k in WEIGHT_NAMES}
        total = 5.0
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


TRIAL_ROOT = Path("/content/acmot_v2_smoke_trials" if SMOKE else "/content/acmot_v2_trials")
TRIAL_ROOT.mkdir(parents=True, exist_ok=True)


def objective(trial):
    params = trial_params(trial)
    system = dict(name=f"V2_{trial.number:03d}", **A3)
    print("\n" + "#" * 100, flush=True)
    print(f"V2 OPTUNA TRIAL {trial.number + 1} / target {N_TRIALS}", flush=True)
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
        trial.set_user_attr("feasible", False)
        return -1.0, 10**9

    feasible = result["FPS"] >= MIN_FPS
    trial.set_user_attr("empirical_params", params)
    trial.set_user_attr("feasible", bool(feasible))
    for k, v in result.items():
        if isinstance(v, (int, float, str, bool)):
            trial.set_user_attr(k, v)

    print(
        f"[TRIAL {trial.number:03d}] MOTA={100*result['MOTA']:.3f}% "
        f"HOTA={100*result['HOTA']:.3f}% IDF1={100*result['IDF1']:.3f}% "
        f"IDS={result['IDS']} FPS={result['FPS']:.2f} FEASIBLE={feasible}",
        flush=True,
    )
    return result["MOTA"], result["IDS"]


SIGNATURE_PAYLOAD = {
    "search_space": SEARCH_SPACE_SNAPSHOT,
    "input_hashes": INPUT_HASHES,
    "seed": SEED,
    "sampler": "TPESampler",
    "directions": ["maximize", "minimize"],
    "objectives": ["MOTA", "IDS"],
    "min_fps": MIN_FPS,
    "trial_budget": N_TRIALS,
    "algorithm": "acmot_v2_pareto_mota_ids_v1",
    "evaluation_protocol": PROTOCOL["evaluation_protocol"],
}
SIGNATURE = hashlib.sha256(
    json.dumps(SIGNATURE_PAYLOAD, sort_keys=True).encode()
).hexdigest()
SIG_PATH = RUN_ROOT / "V2_STUDY_SIGNATURE.json"
if SIG_PATH.exists():
    prior = json.loads(SIG_PATH.read_text())
    if prior.get("sha256") != SIGNATURE and not FORCE_RESET_STUDY:
        raise RuntimeError(
            "Existing V2 study signature is incompatible. "
            "Use ACMOT_V2_FORCE_RESET_STUDY=1 only for an intentional new study."
        )
SIG_PATH.write_text(json.dumps({"sha256": SIGNATURE, "payload": SIGNATURE_PAYLOAD}, indent=2))

local_db = Path("/content/acmot_v2_smoke.db" if SMOKE else "/content/acmot_v2_multiobjective.db")
drive_db = RUN_ROOT / "V2_MULTI_OBJECTIVE_OPTUNA.db"
if FORCE_RESET_STUDY:
    local_db.unlink(missing_ok=True)
    drive_db.unlink(missing_ok=True)
elif drive_db.exists() and not local_db.exists():
    shutil.copy2(drive_db, local_db)

study_name = "acmot_v2_smoke" if SMOKE else "acmot_v2_mota_ids_validation"
study = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=TPESampler(seed=SEED),
    study_name=study_name,
    storage=f"sqlite:///{local_db}",
    load_if_exists=True,
)


def trial_row(t):
    row = {
        "trial": int(t.number),
        "state": t.state.name,
        "MOTA": float(t.values[0]) if t.values is not None else "",
        "IDS": int(t.values[1]) if t.values is not None else "",
        "feasible": bool(t.user_attrs.get("feasible", False)),
        "HOTA": t.user_attrs.get("HOTA", ""),
        "IDF1": t.user_attrs.get("IDF1", ""),
        "DetA": t.user_attrs.get("DetA", ""),
        "AssA": t.user_attrs.get("AssA", ""),
        "FN": t.user_attrs.get("FN", ""),
        "FP": t.user_attrs.get("FP", ""),
        "FPS": t.user_attrs.get("FPS", ""),
        "mean_imgsz": t.user_attrs.get("mean_imgsz", ""),
        "mean_conf": t.user_attrs.get("mean_conf", ""),
        "mean_nms_iou": t.user_attrs.get("mean_nms_iou", ""),
    }
    for k, v in sorted(t.params.items()):
        row[f"optuna_{k}"] = v
    effective = t.user_attrs.get("empirical_params", {})
    if isinstance(effective, dict):
        for k, v in sorted(effective.items()):
            row[f"effective_{k}"] = v
    if "error" in t.user_attrs:
        row["error"] = t.user_attrs["error"]
    return row


def write_rows_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_progress(study, trial=None):
    write_rows_csv(RUN_ROOT / "V2_ALL_TRIALS.csv", [trial_row(t) for t in study.trials])
    shutil.copy2(local_db, drive_db)


complete_before = [
    t for t in study.trials
    if t.state == TrialState.COMPLETE and t.values is not None
]
remaining = max(0, N_TRIALS - len(complete_before))
print(f"[RESUME] completed={len(complete_before)} target={N_TRIALS} remaining={remaining}", flush=True)
if remaining:
    study.optimize(
        objective,
        n_trials=remaining,
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[save_progress],
    )
save_progress(study)

complete = [
    t for t in study.trials
    if t.state == TrialState.COMPLETE
    and t.values is not None
    and float(t.values[0]) >= 0
]
feasible = [
    t for t in complete
    if bool(t.user_attrs.get("feasible", False))
    and float(t.user_attrs.get("FPS", 0.0)) >= MIN_FPS
]
if not feasible:
    raise RuntimeError("No feasible V2 candidate with FPS >= minimum gate.")


def dominates(a, b):
    a_mota, a_ids = float(a.values[0]), int(a.values[1])
    b_mota, b_ids = float(b.values[0]), int(b.values[1])
    return (a_mota >= b_mota and a_ids <= b_ids) and (
        a_mota > b_mota or a_ids < b_ids
    )


pareto = [
    t for t in feasible
    if not any(dominates(other, t) for other in feasible if other.number != t.number)
]
pareto = sorted(pareto, key=lambda t: (-float(t.values[0]), int(t.values[1]), t.number))


def metrics_dict(t):
    return {
        "trial": int(t.number),
        "MOTA": float(t.values[0]),
        "HOTA": float(t.user_attrs["HOTA"]),
        "IDF1": float(t.user_attrs["IDF1"]),
        "DetA": float(t.user_attrs["DetA"]),
        "AssA": float(t.user_attrs["AssA"]),
        "IDS": int(t.values[1]),
        "FN": int(t.user_attrs["FN"]),
        "FP": int(t.user_attrs["FP"]),
        "FPS": float(t.user_attrs["FPS"]),
        "mean_imgsz": float(t.user_attrs["mean_imgsz"]),
        "mean_conf": float(t.user_attrs["mean_conf"]),
        "mean_nms_iou": float(t.user_attrs["mean_nms_iou"]),
        "optimized_parameters": t.user_attrs["empirical_params"],
    }


highest_mota = max(
    pareto,
    key=lambda t: (
        float(t.values[0]),
        -int(t.values[1]),
        float(t.user_attrs["HOTA"]),
        float(t.user_attrs["IDF1"]),
        float(t.user_attrs["FPS"]),
        -t.number,
    ),
)
lowest_ids = min(
    pareto,
    key=lambda t: (
        int(t.values[1]),
        -float(t.values[0]),
        -float(t.user_attrs["HOTA"]),
        -float(t.user_attrs["IDF1"]),
        -float(t.user_attrs["FPS"]),
        t.number,
    ),
)

motas = [float(t.values[0]) for t in pareto]
idss = [int(t.values[1]) for t in pareto]
mota_lo, mota_hi = min(motas), max(motas)
ids_lo, ids_hi = min(idss), max(idss)


def normalize_good(value, lo, hi, higher_is_better=True):
    if hi == lo:
        return 1.0
    if higher_is_better:
        return (value - lo) / (hi - lo)
    return (hi - value) / (hi - lo)


balanced_scored = []
for t in pareto:
    mota_norm = normalize_good(float(t.values[0]), mota_lo, mota_hi, True)
    ids_good_norm = normalize_good(int(t.values[1]), ids_lo, ids_hi, False)
    score = 0.5 * mota_norm + 0.5 * ids_good_norm
    balanced_scored.append((t, mota_norm, ids_good_norm, score))

balanced, _, _, balanced_score = max(
    balanced_scored,
    key=lambda item: (
        item[3],
        float(item[0].user_attrs["HOTA"]),
        float(item[0].user_attrs["IDF1"]),
        float(item[0].user_attrs["FPS"]),
        -item[0].number,
    ),
)

pareto_rows = []
score_map = {
    t.number: (mota_norm, ids_good_norm, score)
    for t, mota_norm, ids_good_norm, score in balanced_scored
}
for t in pareto:
    row = trial_row(t)
    row["pareto"] = True
    row["MOTA_norm"] = score_map[t.number][0]
    row["IDS_good_norm"] = score_map[t.number][1]
    row["BALANCED_SCORE"] = score_map[t.number][2]
    pareto_rows.append(row)
write_rows_csv(RUN_ROOT / "V2_PARETO_FRONT.csv", pareto_rows)


def write_candidate(filename, role, t, extra=None):
    payload = {
        "role": role,
        "selection_scope": "feasible Pareto front",
        "feasibility_gate": f"FPS >= {MIN_FPS}",
        "metrics": metrics_dict(t),
        "study_signature": SIGNATURE,
        "validation_only": True,
        "test_dev_accessed": False,
    }
    if extra:
        payload.update(extra)
    (RUN_ROOT / filename).write_text(json.dumps(payload, indent=2))
    return payload


highest_payload = write_candidate(
    "V2_HIGHEST_MOTA_CANDIDATE.json", "highest_mota_feasible_pareto", highest_mota
)
lowest_payload = write_candidate(
    "V2_LOWEST_IDS_CANDIDATE.json", "lowest_ids_feasible_pareto", lowest_ids
)
balanced_payload = write_candidate(
    "V2_SELECTED_CANDIDATE.json",
    "balanced_feasible_pareto",
    balanced,
    {
        "balanced_score": balanced_score,
        "balanced_formula": "0.5*MOTA_norm + 0.5*IDS_good_norm",
        "normalization": {
            "MOTA_min": mota_lo,
            "MOTA_max": mota_hi,
            "IDS_min": ids_lo,
            "IDS_max": ids_hi,
            "zero_denominator_rule": "normalized value = 1.0 when all Pareto values are identical",
        },
        "tie_breakers": ["higher HOTA", "higher IDF1", "higher FPS", "lower trial number"],
    },
)


def load_v1_reference():
    selected = int(V1_FROZEN["selected_trial"])
    ref = dict(V1_FROZEN["validation_result"])
    ref["trial"] = selected
    ref.setdefault("FN", "")
    ref.setdefault("FP", "")
    ref.setdefault("DetA", "")
    ref.setdefault("AssA", "")
    if V1_TRIALS_PATH.is_file():
        for row in csv.DictReader(V1_TRIALS_PATH.open()):
            if int(float(row["number"])) != selected:
                continue
            mapping = {
                "FN": "user_attrs_FN",
                "FP": "user_attrs_FP",
                "DetA": "user_attrs_DetA",
                "AssA": "user_attrs_AssA",
                "HOTA": "user_attrs_HOTA",
                "IDF1": "user_attrs_IDF1",
                "FPS": "user_attrs_FPS",
                "MOTA": "user_attrs_MOTA",
                "IDS": "user_attrs_IDS",
            }
            for out_key, csv_key in mapping.items():
                if row.get(csv_key, "") != "":
                    if out_key in {"FN", "FP", "IDS"}:
                        ref[out_key] = int(float(row[csv_key]))
                    else:
                        ref[out_key] = float(row[csv_key])
            break
    return ref


v1_ref = load_v1_reference()
candidate_roles = [
    ("V1_TRIAL24_REFERENCE", v1_ref),
    ("V2_HIGHEST_MOTA", metrics_dict(highest_mota)),
    ("V2_LOWEST_IDS", metrics_dict(lowest_ids)),
    ("V2_BALANCED", metrics_dict(balanced)),
]
comparison = []
for role, data in candidate_roles:
    row = {
        "system": role,
        "trial": data.get("trial", ""),
        "MOTA": data.get("MOTA", ""),
        "HOTA": data.get("HOTA", ""),
        "IDF1": data.get("IDF1", ""),
        "IDS": data.get("IDS", ""),
        "FN": data.get("FN", ""),
        "FP": data.get("FP", ""),
        "FPS": data.get("FPS", ""),
    }
    if role != "V1_TRIAL24_REFERENCE":
        for key in ["MOTA", "HOTA", "IDF1", "IDS", "FN", "FP", "FPS"]:
            if row[key] != "" and v1_ref.get(key, "") != "":
                row[f"delta_{key}_vs_V1"] = float(row[key]) - float(v1_ref[key])
    comparison.append(row)
write_rows_csv(RUN_ROOT / "V1_VS_V2_VALIDATION_COMPARISON.csv", comparison)

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.scatter(
        [int(t.values[1]) for t in feasible],
        [100 * float(t.values[0]) for t in feasible],
        alpha=0.45,
        label="Feasible trials",
    )
    plt.scatter(
        [int(t.values[1]) for t in pareto],
        [100 * float(t.values[0]) for t in pareto],
        label="Pareto front",
    )
    plt.scatter(
        [int(balanced.values[1])],
        [100 * float(balanced.values[0])],
        marker="*",
        s=180,
        label="Balanced candidate",
    )
    plt.xlabel("IDS (lower is better)")
    plt.ylabel("MOTA (%) (higher is better)")
    plt.title("AC-MOT V2 Validation Pareto Front")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_ROOT / "plots" / "V2_PARETO_MOTA_IDS.png", dpi=180)
    plt.close()
except Exception as exc:
    print("[INFO] Pareto plot skipped:", exc, flush=True)

run_summary = {
    "smoke": SMOKE,
    "completed_trials": len(complete),
    "feasible_trials": len(feasible),
    "pareto_trials": len(pareto),
    "highest_mota_trial": int(highest_mota.number),
    "lowest_ids_trial": int(lowest_ids.number),
    "balanced_trial": int(balanced.number),
    "study_signature": SIGNATURE,
    "validation_only": True,
    "test_dev_accessed": False,
}
(RUN_ROOT / "logs" / "V2_RUN_SUMMARY.json").write_text(json.dumps(run_summary, indent=2))

if not SMOKE:
    done = {
        "status": "V2_VALIDATION_DONE",
        **run_summary,
        "protocol": str(RUN_ROOT / "V2_PROTOCOL.json"),
        "selected_candidate": balanced_payload,
        "warning": "V2 is validation-only and does not replace the frozen V1 held-out final result.",
    }
    (RUN_ROOT / "V2_DONE.json").write_text(json.dumps(done, indent=2))

print("\n" + "=" * 100, flush=True)
print("V2 MULTI-OBJECTIVE VALIDATION COMPLETE" if not SMOKE else "V2 SMOKE TEST COMPLETE", flush=True)
print("Completed trials :", len(complete), flush=True)
print("Feasible trials  :", len(feasible), flush=True)
print("Pareto trials    :", len(pareto), flush=True)
print("Highest MOTA     : trial", highest_mota.number, flush=True)
print("Lowest IDS       : trial", lowest_ids.number, flush=True)
print("Balanced         : trial", balanced.number, flush=True)
print("V2_DONE          :", "CREATED" if not SMOKE else "NOT CREATED (SMOKE)", flush=True)
print("TEST-DEV         : NOT ACCESSED", flush=True)
print("=" * 100, flush=True)
