"""Portable validation-only operating-point ablations for AC-MOT.

Purpose
-------
Remove hand-picked detector-control numbers before the final SCI optimization.
This script keeps YOLOv8n and the tuned ByteTrack profile fixed and uses only
VisDrone validation data. It performs three independent, auditable sweeps:

1) Inference resolution screening.
2) Detector confidence screening.
3) NMS IoU screening.

The sweeps are deliberately broad and regular rather than centered on the old
AC-MOT hand-picked values. 640 is retained as the standard YOLO reference, and
confidence=0.25 / NMS IoU=0.70 are used only as the documented Ultralytics
static inference reference while the one-at-a-time screens are performed.

Outputs
-------
- OPERATING_RESOLUTION_SWEEP.csv
- OPERATING_CONFIDENCE_SWEEP.csv
- OPERATING_NMS_SWEEP.csv
- SCIENTIFIC_SEARCH_SPACE.json
- OPERATING_ABLATION_REPORT.json

SCIENTIFIC_SEARCH_SPACE.json is consumed by the empirical joint Optuna stage.
This script NEVER reads test-dev.

Protocol note
-------------
Evaluation is the repository's custom class-agnostic AC-MOT TrackEval protocol,
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
import time
from pathlib import Path


def sh(args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def float_grid(start: float, stop: float, step: float):
    n = int(round((stop - start) / step))
    return [round(start + i * step, 6) for i in range(n + 1)]


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
PROGRESS_EVERY = int(os.environ.get("ACMOT_PROGRESS_EVERY", "100"))
SMOKE = os.environ.get("ACMOT_CALIBRATION_SMOKE", "0") == "1"
FORCE = os.environ.get("ACMOT_FORCE_OPERATING_ABLATION", "0") == "1"

# Broad regular screens. They are configuration, are written into the audit,
# and can be overridden without editing source.
if SMOKE:
    RESOLUTIONS = [640, 736, 832]
    CONF_VALUES = [0.15, 0.25, 0.35]
    NMS_VALUES = [0.45, 0.60, 0.75]
else:
    rmin = int(os.environ.get("ACMOT_RESOLUTION_MIN", "512"))
    rmax = int(os.environ.get("ACMOT_RESOLUTION_MAX", "960"))
    rstep = int(os.environ.get("ACMOT_RESOLUTION_STEP", "32"))
    RESOLUTIONS = list(range(rmin, rmax + 1, rstep))
    CONF_VALUES = float_grid(
        float(os.environ.get("ACMOT_CONF_SCREEN_MIN", "0.05")),
        float(os.environ.get("ACMOT_CONF_SCREEN_MAX", "0.50")),
        float(os.environ.get("ACMOT_CONF_SCREEN_STEP", "0.05")),
    )
    NMS_VALUES = float_grid(
        float(os.environ.get("ACMOT_NMS_SCREEN_MIN", "0.30")),
        float(os.environ.get("ACMOT_NMS_SCREEN_MAX", "0.80")),
        float(os.environ.get("ACMOT_NMS_SCREEN_STEP", "0.05")),
    )

# External/static reference only. Final values are NOT forced to these values.
REFERENCE_CONF = 0.25
REFERENCE_NMS = 0.70
REFERENCE_RESOLUTION = 640

if not DRIVE.is_dir():
    raise RuntimeError("Mount Google Drive first: from google.colab import drive; drive.mount('/content/drive')")
if not ROOT.is_dir():
    raise RuntimeError(f"Repository not found at {ROOT}. Clone AhmedCode110/AC-MOT first.")

DATA_ROOT.mkdir(parents=True, exist_ok=True)
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
    print("[DATA] Validation set missing; downloading portable public mirror...", flush=True)
    import urllib.request
    import zipfile
    archive = DATA_ROOT / "VisDrone2019-MOT-val.zip"
    url = "https://huggingface.co/datasets/vanthanh/VisDrone2019-MOT/resolve/main/VisDrone2019-MOT-val.zip"
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
        raise RuntimeError("Could not identify extracted validation folder.")

if not WEIGHTS.exists():
    from ultralytics import YOLO
    old = os.getcwd(); os.chdir(WEIGHTS.parent)
    try:
        YOLO("yolov8n.pt")
    finally:
        os.chdir(old)
if not WEIGHTS.exists():
    raise RuntimeError("Could not obtain yolov8n.pt")

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

import pandas as pd
import torch

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment import dataset_manifest
import scripts.paper_eval_v17 as pe

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")
print("[GPU]", torch.cuda.get_device_name(0), flush=True)
if "T4" not in torch.cuda.get_device_name(0):
    print("[WARNING] Non-T4 GPU: FPS cannot be compared directly with prior T4 measurements.", flush=True)

val_names = sorted(p.name for p in (VAL_DIR / "sequences").iterdir() if p.is_dir())
VAL_MANIFEST = dataset_manifest(VAL_DIR, val_names)
GT_FILTER = dict(categories=[1, 4, 5, 6, 9], score=1, occlusion_lt=2, truncation_lt=2)


class FixedOperatingController:
    def __init__(self, spec, conf: float, nms: float, size: int):
        self.conf = float(conf)
        self.nms = float(nms)
        self.size = int(size)

    def choose(self, frame, visual, previous):
        return {
            "conf": self.conf,
            "nms": self.nms,
            "size": self.size,
            "sci": 0.0,
            "scene": "static_operating_screen",
        }


def parse_metrics(path: Path, name: str):
    row = next(r for r in csv.DictReader(path.open()) if r["system"] == name)
    return dict(
        HOTA=float(row["HOTA"]), DetA=float(row["DetA"]), AssA=float(row["AssA"]),
        MOTA=float(row["MOTA"]), IDF1=float(row["IDF1"]),
        IDS=int(float(row["IDS"])), FN=int(float(row["FN"])), FP=int(float(row["FP"])),
    )


def write_meta(root: Path, system: dict, purpose: str):
    (root / "configuration.json").write_text(json.dumps({
        "version": "scientific_operating_ablation_v1",
        "systems": [system],
        "ground_truth_filter": GT_FILTER,
        "official_visdrone": False,
        "purpose": purpose,
    }, indent=2))
    (root / "dataset_manifest.json").write_text(json.dumps(VAL_MANIFEST, indent=2))


def run_static(label: str, conf: float, nms: float, size: int, purpose: str):
    safe = label.replace(".", "p")
    root = Path("/content/acmot_operating_ablation") / safe
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    system = dict(
        name=label,
        tracker_profile="tuned",
        adaptive_threshold=False,
        # True warms the repository's standard multi-resolution shapes; the
        # custom controller still returns exactly the requested static size.
        adaptive_resolution=True,
        smoothing_window=1,
        analysis_stride=1,
    )
    write_meta(root, system, purpose)
    old_controller = pe.PresentationController
    try:
        pe.PresentationController = lambda spec: FixedOperatingController(spec, conf, nms, size)
        timing = pe.run_system(
            system, VAL_DIR, VAL_MANIFEST, WEIGHTS,
            Path("/content/weights/unused.engine"), MIN_FPS, PROGRESS_EVERY,
            "pytorch", 16, root, 1, 1,
        )
    finally:
        pe.PresentationController = old_controller

    out = root / "trackeval"
    sh([
        sys.executable, str(ROOT / "evaluate.py"), str(root),
        "--dataset", str(VAL_DIR), "--trackeval", str(TRACKEVAL_DIR),
        "--output", str(out),
    ], cwd=ROOT)
    result = parse_metrics(out / "summary.csv", label)
    result.update(
        conf=float(conf), nms=float(nms), resolution=int(size),
        FPS=float(timing["processing_fps"]),
        p95_ms=float(timing["processing_p95_ms"]),
    )
    shutil.rmtree(root, ignore_errors=True)
    gc.collect(); torch.cuda.empty_cache()
    return result


def load_csv(path: Path):
    if not path.exists() or FORCE:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


def save_rows(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def quality_key(r):
    return (float(r["MOTA"]), -int(r["IDS"]), float(r["HOTA"]), float(r["IDF1"]), float(r["FPS"]))


def supported_values(rows, field: str, reference_value: float, minimum_count: int = 3):
    ref = min(rows, key=lambda r: abs(float(r[field]) - reference_value))
    threshold = float(ref["MOTA"])
    supported = [r for r in rows if float(r["FPS"]) >= MIN_FPS and float(r["MOTA"]) >= threshold]
    if len(supported) < minimum_count:
        supported = sorted([r for r in rows if float(r["FPS"]) >= MIN_FPS], key=quality_key, reverse=True)[:minimum_count]
    values = sorted({round(float(r[field]), 6) for r in supported})
    return values, ref


def select_three_resolutions(rows):
    feasible = [r for r in rows if float(r["FPS"]) >= MIN_FPS]
    if len(feasible) < 3:
        feasible = list(rows)
    by_res = sorted(feasible, key=lambda r: int(r["resolution"]))
    low = by_res[0]
    high = max(feasible, key=quality_key)
    if int(high["resolution"]) == int(low["resolution"]):
        distinct = [r for r in sorted(feasible, key=quality_key, reverse=True) if int(r["resolution"]) != int(low["resolution"])]
        if distinct:
            high = distinct[0]
    lo, hi = sorted([int(low["resolution"]), int(high["resolution"])])
    between = [r for r in feasible if lo < int(r["resolution"]) < hi]
    if between:
        mid = max(between, key=quality_key)
    else:
        remaining = [r for r in sorted(feasible, key=quality_key, reverse=True)
                     if int(r["resolution"]) not in {int(low["resolution"]), int(high["resolution"])}]
        if not remaining:
            raise RuntimeError("Could not select three distinct resolution levels.")
        mid = remaining[0]
    selected = sorted({int(low["resolution"]), int(mid["resolution"]), int(high["resolution"])})
    if len(selected) < 3:
        for r in sorted(feasible, key=quality_key, reverse=True):
            selected.append(int(r["resolution"]))
            selected = sorted(set(selected))
            if len(selected) == 3:
                break
    if len(selected) != 3:
        raise RuntimeError("Resolution sweep did not provide three distinct levels.")
    return selected


print("\n" + "=" * 100, flush=True)
print("AC-MOT SCIENTIFIC OPERATING-POINT ABLATION", flush=True)
print("Validation only : YES", flush=True)
print("Test-dev        : NOT ACCESSED", flush=True)
print("Detector        : fixed pretrained YOLOv8n", flush=True)
print("Tracker         : fixed tuned ByteTrack (not retuned in this SCI study)", flush=True)
print("FPS gate        : >=", MIN_FPS, flush=True)
print("Smoke           :", SMOKE, flush=True)
print("=" * 100, flush=True)

# -------------------------------------------------------------------------
# 1) Resolution sweep using documented static YOLO reference settings.
# -------------------------------------------------------------------------
res_csv = RESULT_ROOT / ("OPERATING_RESOLUTION_SWEEP_SMOKE.csv" if SMOKE else "OPERATING_RESOLUTION_SWEEP.csv")
res_rows = load_csv(res_csv)
seen = {int(r["resolution"]) for r in res_rows} if res_rows else set()
started = time.perf_counter()
for i, size in enumerate(RESOLUTIONS, 1):
    if size in seen:
        print(f"[RES {i}/{len(RESOLUTIONS)}] reuse {size}", flush=True)
        continue
    print(f"\n[RES {i}/{len(RESOLUTIONS)}] imgsz={size}", flush=True)
    r = run_static(f"RES_{size}", REFERENCE_CONF, REFERENCE_NMS, size, "resolution screening")
    res_rows.append(r); seen.add(size); save_rows(res_csv, res_rows)
    print(f"[RES RESULT] {size} | MOTA={100*r['MOTA']:.3f}% HOTA={100*r['HOTA']:.3f}% IDF1={100*r['IDF1']:.3f}% IDS={r['IDS']} FPS={r['FPS']:.2f}", flush=True)

selected_resolutions = select_three_resolutions(res_rows)
anchor_resolution = max(
    [r for r in res_rows if int(r["resolution"]) in selected_resolutions and float(r["FPS"]) >= MIN_FPS] or res_rows,
    key=quality_key,
)["resolution"]
anchor_resolution = int(anchor_resolution)
print("[RESOLUTION SELECTED]", selected_resolutions, "anchor=", anchor_resolution, flush=True)

# -------------------------------------------------------------------------
# 2) Confidence sweep at selected static resolution; NMS uses reference.
# -------------------------------------------------------------------------
conf_csv = RESULT_ROOT / ("OPERATING_CONFIDENCE_SWEEP_SMOKE.csv" if SMOKE else "OPERATING_CONFIDENCE_SWEEP.csv")
conf_rows = load_csv(conf_csv)
seen = {round(float(r["conf"]), 6) for r in conf_rows} if conf_rows else set()
for i, conf in enumerate(CONF_VALUES, 1):
    if round(conf, 6) in seen:
        print(f"[CONF {i}/{len(CONF_VALUES)}] reuse {conf:.3f}", flush=True)
        continue
    print(f"\n[CONF {i}/{len(CONF_VALUES)}] conf={conf:.3f}", flush=True)
    r = run_static(f"CONF_{conf:.3f}", conf, REFERENCE_NMS, anchor_resolution, "confidence screening")
    conf_rows.append(r); seen.add(round(conf, 6)); save_rows(conf_csv, conf_rows)
    print(f"[CONF RESULT] {conf:.3f} | MOTA={100*r['MOTA']:.3f}% HOTA={100*r['HOTA']:.3f}% IDF1={100*r['IDF1']:.3f}% IDS={r['IDS']} FPS={r['FPS']:.2f}", flush=True)

conf_candidates, conf_reference = supported_values(conf_rows, "conf", REFERENCE_CONF)
anchor_conf_row = max([r for r in conf_rows if round(float(r["conf"]), 6) in set(conf_candidates)], key=quality_key)
anchor_conf = float(anchor_conf_row["conf"])
print("[CONF SUPPORTED]", conf_candidates, "anchor=", anchor_conf, flush=True)

# -------------------------------------------------------------------------
# 3) NMS IoU sweep using selected confidence + resolution.
# -------------------------------------------------------------------------
nms_csv = RESULT_ROOT / ("OPERATING_NMS_SWEEP_SMOKE.csv" if SMOKE else "OPERATING_NMS_SWEEP.csv")
nms_rows = load_csv(nms_csv)
seen = {round(float(r["nms"]), 6) for r in nms_rows} if nms_rows else set()
for i, nms in enumerate(NMS_VALUES, 1):
    if round(nms, 6) in seen:
        print(f"[NMS {i}/{len(NMS_VALUES)}] reuse {nms:.3f}", flush=True)
        continue
    print(f"\n[NMS {i}/{len(NMS_VALUES)}] iou={nms:.3f}", flush=True)
    r = run_static(f"NMS_{nms:.3f}", anchor_conf, nms, anchor_resolution, "NMS IoU screening")
    nms_rows.append(r); seen.add(round(nms, 6)); save_rows(nms_csv, nms_rows)
    print(f"[NMS RESULT] {nms:.3f} | MOTA={100*r['MOTA']:.3f}% HOTA={100*r['HOTA']:.3f}% IDF1={100*r['IDF1']:.3f}% IDS={r['IDS']} FPS={r['FPS']:.2f}", flush=True)

nms_candidates, nms_reference = supported_values(nms_rows, "nms", REFERENCE_NMS)
anchor_nms_row = max([r for r in nms_rows if round(float(r["nms"]), 6) in set(nms_candidates)], key=quality_key)
anchor_nms = float(anchor_nms_row["nms"])
print("[NMS SUPPORTED]", nms_candidates, "anchor=", anchor_nms, flush=True)

search_space = {
    "version": 1,
    "validation_only": True,
    "test_used": False,
    "detector": "fixed pretrained YOLOv8n",
    "tracker": "fixed tuned ByteTrack",
    "selection_principle": (
        "Broad one-at-a-time validation screens first; joint optimization then uses only empirically supported detector-control choices."
    ),
    "resolution_candidates_screened": RESOLUTIONS,
    "selected_resolution_levels": selected_resolutions,
    "confidence_candidates_screened": CONF_VALUES,
    "supported_confidence_values": conf_candidates,
    "nms_candidates_screened": NMS_VALUES,
    "supported_nms_values": nms_candidates,
    "cue_calibration_anchor": {
        "resolution": anchor_resolution,
        "confidence": anchor_conf,
        "nms_iou": anchor_nms,
    },
    "reference_static_settings": {
        "confidence": REFERENCE_CONF,
        "nms_iou": REFERENCE_NMS,
        "resolution": REFERENCE_RESOLUTION,
        "role": "static screening reference only; not forced as the final AC-MOT setting",
    },
    "fps_gate": MIN_FPS,
    "screening_rule_for_conf_nms": (
        "Keep values meeting the FPS gate and at least the static-reference MOTA; if fewer than 3 survive, retain the top 3 feasible values by MOTA, lower IDS, HOTA, IDF1, FPS."
    ),
    "resolution_selection_rule": (
        "Choose a low-compute feasible level, the highest-quality feasible level, and the highest-quality feasible intermediate/distinct level; then sort ascending."
    ),
    "protocol": "custom class-agnostic AC-MOT evaluation; not official VisDrone leaderboard",
}
space_path = RESULT_ROOT / ("SCIENTIFIC_SEARCH_SPACE_SMOKE.json" if SMOKE else "SCIENTIFIC_SEARCH_SPACE.json")
space_path.write_text(json.dumps(search_space, indent=2))

report = {
    "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    "search_space_file": str(space_path),
    "resolution_csv": str(res_csv),
    "confidence_csv": str(conf_csv),
    "nms_csv": str(nms_csv),
    "selected_resolution_levels": selected_resolutions,
    "supported_confidence_values": conf_candidates,
    "supported_nms_values": nms_candidates,
    "cue_calibration_anchor": search_space["cue_calibration_anchor"],
    "scientific_scope": (
        "These ablations justify detector-control operating choices. ByteTrack parameters are intentionally fixed so SCI contribution is not confounded; their evidence must come from the separate tracker-tuning experiment."
    ),
}
(RESULT_ROOT / ("OPERATING_ABLATION_REPORT_SMOKE.json" if SMOKE else "OPERATING_ABLATION_REPORT.json")).write_text(json.dumps(report, indent=2))

print("\n" + "=" * 100, flush=True)
print("OPERATING-POINT ABLATION COMPLETE", flush=True)
print("Resolution levels :", selected_resolutions, flush=True)
print("Confidence values :", conf_candidates, flush=True)
print("NMS IoU values    :", nms_candidates, flush=True)
print("Search-space file :", space_path, flush=True)
print("Test-dev          : NOT ACCESSED", flush=True)
print("=" * 100, flush=True)
