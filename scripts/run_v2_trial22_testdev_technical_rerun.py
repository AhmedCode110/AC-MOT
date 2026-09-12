#!/usr/bin/env python3
"""Frozen V2 Trial22 VisDrone test-dev technical rerun (attempt 2).

Purpose:
- Recover the post-selection test-dev evaluation after attempt 1 was interrupted
  before any persisted result.
- Run exactly the preselected V2 Trial22 configuration.
- Do not tune, reselect, or modify parameters based on test-dev.

Scientific lock:
- Evaluation repository commit: 5becc52a569f271ee8b73ce47c67d495de0d64a5
- TrackEval commit: 12c8791b303e0a0b50f753af204249e622d0281a
- Trial: 22
- VisDrone custom class-agnostic AC-MOT TrackEval protocol.
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
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch
import ultralytics
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# Locked paths and identities
# -----------------------------------------------------------------------------

REPO = Path(os.environ.get("ACMOT_REPO", "/content/AC-MOT"))
TRACKEVAL = Path(os.environ.get("ACMOT_TRACKEVAL", "/content/TrackEval"))

POST_ROOT = Path(
    os.environ.get(
        "ACMOT_V2_POST_ROOT",
        "/content/drive/MyDrive/AC-MOT-shared/V2_POST_SELECTION_TEST_2026-09-12",
    )
)
V1_ROOT = Path(
    os.environ.get(
        "ACMOT_V1_RESULT_ROOT",
        "/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers",
    )
)

FREEZE = POST_ROOT / "V2_TRIAL22_FINAL_FREEZE.json"
PROTOCOL = POST_ROOT / "V2_POST_SELECTION_TEST_PROTOCOL.json"
AUDIT = POST_ROOT / "V2_TRIAL22_TECHNICAL_RERUN_AUDIT.json"
ATTEMPT1 = POST_ROOT / "V2_TESTDEV_ACCESS_STARTED_ATTEMPT1_INTERRUPTED.json"

START = POST_ROOT / "V2_TESTDEV_ACCESS_STARTED.json"
DONE = POST_ROOT / "V2_TRIAL22_TESTDEV_DONE.json"
FAILED = POST_ROOT / "V2_TRIAL22_TESTDEV_ATTEMPT2_FAILED.json"

FINAL_OUT = POST_ROOT / "V2_TRIAL22_TESTDEV"
COPYING_OUT = POST_ROOT / "V2_TRIAL22_TESTDEV_ATTEMPT2_COPYING"
LOCAL_OUT = Path("/content/V2_TRIAL22_TESTDEV_ATTEMPT2")

REPO_COMMIT = "5becc52a569f271ee8b73ce47c67d495de0d64a5"
TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
EXPECTED_FREEZE_SHA = "ba922adac720ac822e80145aa697e24f86bd8bd9a71cb49b2736ba83195fcd72"
EXPECTED_PROTOCOL_SHA = "8a20dbc2117d779455fa495d5524afb7a4a760e57a6dce6519256e5bf1eace5e"
CALIBRATION_SHA = "fd42b22987365236944e90d3ebd646a28a56e5b60986cf1ad9f05881e7958c78"

WEIGHTS = Path(os.environ.get("ACMOT_WEIGHTS", "/content/weights/yolov8n.pt"))

GT_FILTER = dict(
    categories=[1, 4, 5, 6, 9],
    score=1,
    occlusion_lt=2,
    truncation_lt=2,
)

RESOLUTIONS = [512, 928, 960]
SMOOTHING_WINDOW = 7
ANALYSIS_STRIDE = 10
MIN_FPS = 25.0
PROGRESS_EVERY = int(os.environ.get("ACMOT_PROGRESS_EVERY", "50"))

EXPECTED_PARAMS = {
    "weight_crowd": 0.1646452656714585,
    "weight_tiny": 0.1746265279504544,
    "weight_edge": 0.5076112530333374,
    "weight_night": 0.1206956469372537,
    "weight_blur": 0.0324213064074956,
    "conf_easy": 0.40,
    "conf_hard": 0.40,
    "nms_easy": 0.60,
    "nms_hard": 0.35,
    "threshold_mid": 0.2927135841069045,
    "threshold_high": 0.6661671600900015,
}
PARAM_KEYS = set(EXPECTED_PARAMS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def git_tracked_changes(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()


def valid_visdrone(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "sequences").is_dir()
        and (path / "annotations").is_dir()
    )


def find_test_dir() -> Path:
    candidates = [
        Path(
            "/content/drive/MyDrive/AC-MOT-shared/"
            "AC-MOT-data/VisDrone2019-MOT-test-dev"
        ),
        Path("/content/drive/MyDrive/AC-MOT-data/VisDrone2019-MOT-test-dev"),
        Path("/content/drive/MyDrive/VisDrone2019-MOT-test-dev"),
    ]
    for candidate in candidates:
        if valid_visdrone(candidate):
            return candidate
    raise RuntimeError("VisDrone2019-MOT-test-dev not found in expected Drive locations.")


def find_trial(obj):
    if isinstance(obj, dict):
        for key in ("selected_trial", "trial"):
            if key in obj:
                try:
                    value = int(obj[key])
                    if value == 22:
                        return value
                except (TypeError, ValueError):
                    pass
        for value in obj.values():
            found = find_trial(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_trial(value)
            if found is not None:
                return found
    return None


def find_params(obj):
    if isinstance(obj, dict):
        if PARAM_KEYS.issubset(obj.keys()):
            return {key: float(obj[key]) for key in PARAM_KEYS}
        for value in obj.values():
            found = find_params(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_params(value)
            if found is not None:
                return found
    return None


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# -----------------------------------------------------------------------------
# Frozen V2 controller (copied from the pinned V2 validation implementation)
# -----------------------------------------------------------------------------

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


# Imports from the frozen evaluation repository are intentionally delayed until
# after the repository identity checks.
boxes = None
PresentationSpec = None
dataset_manifest = None
pe = None


class EmpiricalSCIController:
    def __init__(self, spec, params: dict, calibration: dict):
        self.spec = spec.validate()
        self.params = params
        self.calibration = calibration
        self.history = deque(maxlen=self.spec.smoothing_window)
        self.sci = 0.0
        self.tiny = 0.0

        keys = ["crowd", "tiny", "edge", "night", "blur"]
        raw = np.asarray(
            [float(params[f"weight_{key}"]) for key in keys], dtype=float
        )
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
                areas = (
                    (previous[:, 2] - previous[:, 0])
                    * (previous[:, 3] - previous[:, 1])
                )
                self.tiny = float(np.mean(areas < 32 * 32))
            else:
                self.tiny = 0.0

            cues = {
                "crowd": empirical_rank(n, self.calibration["crowd_sorted"]),
                "tiny": self.tiny,
                "edge": empirical_rank(
                    float(visual["edges"]), self.calibration["edge_sorted"]
                ),
                "night": 1.0
                - empirical_rank(
                    float(visual["brightness"]),
                    self.calibration["brightness_sorted"],
                ),
                "blur": 1.0
                - empirical_rank(
                    float(visual["blur"]), self.calibration["blur_sorted"]
                ),
            }

            raw_sci = float(
                np.clip(
                    sum(self.weights[key] * cues[key] for key in self.weights),
                    0.0,
                    1.0,
                )
            )
            self.history.append(raw_sci)
            self.sci = float(np.mean(self.history))

        p = self.params
        conf = float(
            p["conf_easy"] + self.sci * (p["conf_hard"] - p["conf_easy"])
        )
        nms = float(
            p["nms_easy"] + self.sci * (p["nms_hard"] - p["nms_easy"])
        )

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


def main():
    global boxes, PresentationSpec, dataset_manifest, pe

    print("=" * 90)
    print("V2 TRIAL22 FROZEN — TECHNICAL RERUN ATTEMPT 2")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Hard locks before held-out test access
    # -------------------------------------------------------------------------

    for path in (FREEZE, PROTOCOL, AUDIT, ATTEMPT1):
        if not path.exists():
            raise RuntimeError(f"Missing required audit/freeze artifact: {path}")

    if sha256_file(FREEZE) != EXPECTED_FREEZE_SHA:
        raise RuntimeError("FREEZE HASH CHANGED")
    if sha256_file(PROTOCOL) != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("PROTOCOL HASH CHANGED")

    if DONE.exists():
        raise RuntimeError("DONE marker already exists. DO NOT rerun.")
    if FINAL_OUT.exists():
        raise RuntimeError("Final output already exists. Inspect it instead of rerunning.")
    if START.exists():
        raise RuntimeError("Active START marker already exists.")
    if COPYING_OUT.exists():
        raise RuntimeError(
            "ATTEMPT2_COPYING output exists. Inspect/recover it before rerunning."
        )

    if not REPO.is_dir():
        raise RuntimeError(f"Frozen evaluation repository not found: {REPO}")
    if not TRACKEVAL.is_dir():
        raise RuntimeError(f"Pinned TrackEval checkout not found: {TRACKEVAL}")

    repo_head = git_head(REPO)
    te_head = git_head(TRACKEVAL)
    if repo_head != REPO_COMMIT:
        raise RuntimeError(
            f"Evaluation repo must remain at frozen commit {REPO_COMMIT}; got {repo_head}"
        )
    if te_head != TRACKEVAL_COMMIT:
        raise RuntimeError(
            f"TrackEval must remain at {TRACKEVAL_COMMIT}; got {te_head}"
        )
    if git_tracked_changes(REPO):
        raise RuntimeError("Frozen evaluation repository has modified tracked files.")
    if git_tracked_changes(TRACKEVAL):
        raise RuntimeError("TrackEval has modified tracked files.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name:
        raise RuntimeError(f"Expected Tesla T4; got {gpu_name}")

    if ultralytics.__version__ != "8.3.200":
        raise RuntimeError(
            f"Expected ultralytics 8.3.200; got {ultralytics.__version__}"
        )
    if np.__version__ != "2.2.6":
        raise RuntimeError(f"Expected numpy 2.2.6; got {np.__version__}")
    if scipy.__version__ != "1.15.3":
        raise RuntimeError(f"Expected scipy 1.15.3; got {scipy.__version__}")

    test_dir = find_test_dir()

    print("GPU        :", gpu_name)
    print("Repo       :", repo_head)
    print("TrackEval  :", te_head)
    print("Freeze     : LOCKED")
    print("Protocol   : LOCKED")
    print("Test-dev   :", test_dir)

    # -------------------------------------------------------------------------
    # Load frozen Trial22 and exact calibration
    # -------------------------------------------------------------------------

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    trial = find_trial(freeze)
    params = find_params(freeze)

    if trial != 22:
        raise RuntimeError(f"Expected frozen Trial22, got {trial}")
    if params is None:
        raise RuntimeError("Could not locate frozen Trial22 parameters in freeze JSON.")

    for key, expected in EXPECTED_PARAMS.items():
        observed = float(params[key])
        if abs(observed - expected) >= 1e-10:
            raise RuntimeError(
                f"Frozen parameter mismatch for {key}: {observed} != {expected}"
            )

    print("\nFROZEN TRIAL22 PARAMETERS: VERIFIED")
    for key in sorted(params):
        print(f"  {key}: {params[key]}")

    calibration_path = V1_ROOT / "DETECTOR_DERIVED_CUE_CALIBRATION.json"
    if not calibration_path.exists():
        raise RuntimeError(f"Missing calibration: {calibration_path}")
    if sha256_file(calibration_path) != CALIBRATION_SHA:
        raise RuntimeError("Cue calibration hash mismatch.")

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    print("\nCalibration: VERIFIED")

    # -------------------------------------------------------------------------
    # Fixed detector weights
    # -------------------------------------------------------------------------

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.exists():
        old_cwd = os.getcwd()
        try:
            os.chdir(WEIGHTS.parent)
            YOLO("yolov8n.pt")
        finally:
            os.chdir(old_cwd)

    if not WEIGHTS.exists():
        raise RuntimeError("Could not obtain fixed yolov8n.pt")
    print("YOLOv8n    : READY")

    # -------------------------------------------------------------------------
    # Import only from the frozen repository after identity verification
    # -------------------------------------------------------------------------

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    from core import boxes as frozen_boxes
    from core_v17 import PresentationSpec as FrozenPresentationSpec
    from experiment import dataset_manifest as frozen_dataset_manifest
    import scripts.paper_eval_v17 as frozen_pe

    boxes = frozen_boxes
    PresentationSpec = FrozenPresentationSpec
    dataset_manifest = frozen_dataset_manifest
    pe = frozen_pe

    system = {
        "name": "V2_Trial22_Frozen",
        "tracker_profile": "tuned",
        "adaptive_threshold": True,
        "adaptive_resolution": True,
        "smoothing_window": SMOOTHING_WINDOW,
        "analysis_stride": ANALYSIS_STRIDE,
    }

    # Remove only local output from an earlier pre-start local failure. Drive
    # evidence is never silently deleted.
    if LOCAL_OUT.exists():
        shutil.rmtree(LOCAL_OUT)
    FAILED.unlink(missing_ok=True)

    print("\n" + "=" * 90)
    print("FINAL LOCK CHECK")
    print("=" * 90)
    print("System       : V2_Trial22_Frozen")
    print("Trial        : 22")
    print("Resolutions  :", RESOLUTIONS)
    print("W / stride   :", SMOOTHING_WINDOW, "/", ANALYSIS_STRIDE)
    print("Tracker      : tuned ByteTrack")
    print("GPU          :", gpu_name)
    print("Tuning       : NO")
    print("Reselection  : NO")
    print("Param change : NO")

    # -------------------------------------------------------------------------
    # START marker. Held-out test is considered exposed after this write.
    # -------------------------------------------------------------------------

    start_payload = {
        "status": "TEST_DEV_ACCESS_STARTED",
        "attempt": 2,
        "technical_rerun": True,
        "system": "V2_Trial22_Frozen",
        "trial": 22,
        "repository_commit": REPO_COMMIT,
        "trackeval_commit": TRACKEVAL_COMMIT,
        "freeze_sha256": EXPECTED_FREEZE_SHA,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "gpu": gpu_name,
        "selection_or_tuning_on_test": False,
        "parameter_changes": False,
        "candidate_reselection": False,
        "reason": (
            "Technical rerun after attempt 1 was interrupted before any "
            "persisted evaluation result."
        ),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "Held-out test-dev is exposed. No tuning, parameter changes, "
            "or candidate reselection are permitted."
        ),
    }
    atomic_json(START, start_payload)

    print("\nSTART MARKER CREATED:")
    print(START)

    try:
        # Test split is enumerated only after the access marker exists.
        test_names = sorted(
            p.name for p in (test_dir / "sequences").iterdir() if p.is_dir()
        )
        test_manifest = dataset_manifest(test_dir, test_names)
        test_frames = sum(int(item["frames"]) for item in test_manifest)

        print("\nTest sequences :", len(test_manifest))
        print("Test frames    :", test_frames)

        if len(test_manifest) != 17:
            raise RuntimeError(
                f"Expected 17 test-dev sequences; got {len(test_manifest)}"
            )
        if test_frames != 6635:
            raise RuntimeError(
                f"Expected 6635 test-dev frames; got {test_frames}"
            )

        LOCAL_OUT.mkdir(parents=True, exist_ok=False)

        # evaluate.py expects cfg["systems"], not cfg["system"].
        configuration = {
            "version": "V2_Trial22_Frozen_testdev_technical_rerun_v1",
            "systems": [system],
            "trial": 22,
            "ground_truth_filter": GT_FILTER,
            "evaluation_protocol": "custom class-agnostic AC-MOT research protocol",
            "official_visdrone": False,
            "selection_or_tuning_on_test": False,
            "technical_rerun_attempt": 2,
            "resolutions": RESOLUTIONS,
            "optimized_parameters": params,
            "repository_commit": REPO_COMMIT,
            "trackeval_commit": TRACKEVAL_COMMIT,
        }
        (LOCAL_OUT / "configuration.json").write_text(
            json.dumps(configuration, indent=2), encoding="utf-8"
        )
        (LOCAL_OUT / "dataset_manifest.json").write_text(
            json.dumps(test_manifest, indent=2), encoding="utf-8"
        )

        # Patch exactly the two components replaced during V2 validation.
        old_controller = pe.PresentationController
        old_visual = pe.visual
        try:
            pe.PresentationController = (
                lambda spec: EmpiricalSCIController(spec, params, calibration)
            )
            pe.visual = robust_visual

            timing = pe.run_system(
                system,
                test_dir,
                test_manifest,
                WEIGHTS,
                Path("/content/weights/unused.engine"),
                MIN_FPS,
                PROGRESS_EVERY,
                "pytorch",
                16,
                LOCAL_OUT,
                1,
                1,
            )
        finally:
            pe.PresentationController = old_controller
            pe.visual = old_visual

        track_out = LOCAL_OUT / "trackeval"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "evaluate.py"),
                str(LOCAL_OUT),
                "--dataset",
                str(test_dir),
                "--trackeval",
                str(TRACKEVAL),
                "--output",
                str(track_out),
            ],
            cwd=REPO,
            check=True,
        )

        summary_csv = track_out / "summary.csv"
        if not summary_csv.exists():
            raise RuntimeError("TrackEval summary.csv missing.")

        rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
        row = next(r for r in rows if r["system"] == "V2_Trial22_Frozen")

        result = {
            "MOTA": float(row["MOTA"]),
            "HOTA": float(row["HOTA"]),
            "IDF1": float(row["IDF1"]),
            "DetA": float(row["DetA"]),
            "AssA": float(row["AssA"]),
            "IDS": int(float(row["IDS"])),
            "FN": int(float(row["FN"])),
            "FP": int(float(row["FP"])),
            "FPS": float(timing["processing_fps"]),
            "mean_imgsz": float(timing["mean_imgsz"]),
            "mean_conf": float(timing["mean_conf"]),
            "mean_nms_iou": float(timing["mean_nms_iou"]),
        }

        (LOCAL_OUT / "V2_TRIAL22_TESTDEV_RESULT.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

        # Persist complete output under a temporary Drive name, then rename.
        shutil.copytree(LOCAL_OUT, COPYING_OUT)
        copied_result = COPYING_OUT / "V2_TRIAL22_TESTDEV_RESULT.json"
        if not copied_result.exists():
            raise RuntimeError("Drive copy verification failed: result JSON missing.")
        COPYING_OUT.rename(FINAL_OUT)

        done_payload = {
            "status": "COMPLETE",
            "system": "V2_Trial22_Frozen",
            "trial": 22,
            "attempt": 2,
            "technical_rerun": True,
            "repository_commit": REPO_COMMIT,
            "trackeval_commit": TRACKEVAL_COMMIT,
            "freeze_sha256": EXPECTED_FREEZE_SHA,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA,
            "gpu": gpu_name,
            "sequences": len(test_manifest),
            "frames": test_frames,
            "selection_or_tuning_on_test": False,
            "parameter_changes": False,
            "candidate_reselection": False,
            "results": result,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "output": str(FINAL_OUT),
        }
        atomic_json(DONE, done_payload)

        print("\n" + "=" * 90)
        print("V2 TRIAL22 TEST-DEV COMPLETE")
        print("=" * 90)
        print(f"MOTA : {100 * result['MOTA']:.3f}%")
        print(f"HOTA : {100 * result['HOTA']:.3f}%")
        print(f"IDF1 : {100 * result['IDF1']:.3f}%")
        print(f"IDS  : {result['IDS']}")
        print(f"FN   : {result['FN']}")
        print(f"FP   : {result['FP']}")
        print(f"FPS  : {result['FPS']:.3f}")
        print("=" * 90)
        print("\nDONE marker:")
        print(DONE)
        print("\nFinal output:")
        print(FINAL_OUT)

        gc.collect()
        torch.cuda.empty_cache()

    except Exception as exc:
        failure = {
            "status": "TECHNICAL_RERUN_FAILED",
            "attempt": 2,
            "system": "V2_Trial22_Frozen",
            "trial": 22,
            "repository_commit": REPO_COMMIT,
            "trackeval_commit": TRACKEVAL_COMMIT,
            "freeze_sha256": EXPECTED_FREEZE_SHA,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA,
            "parameter_changes": False,
            "tuning": False,
            "candidate_reselection": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(FAILED, failure)

        print("\n" + "=" * 90)
        print("TECHNICAL RERUN FAILED")
        print("=" * 90)
        print("Failure marker:")
        print(FAILED)
        raise


if __name__ == "__main__":
    main()
