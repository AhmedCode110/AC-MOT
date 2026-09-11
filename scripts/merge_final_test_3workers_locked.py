"""Merge the three locked held-out final-test worker results.

This script performs no inference and no tuning. It verifies that all three
predeclared worker results were produced under the same frozen protocol, writes
a compact comparison CSV/JSON, and creates FINAL_TEST_DONE.json as the final
held-out lock.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

DRIVE = Path(os.environ.get("ACMOT_DRIVE", "/content/drive/MyDrive"))
RESULT_ROOT = Path(
    os.environ.get(
        "ACMOT_RESULT_ROOT",
        str(DRIVE / "AC-MOT-shared" / "defensible_acmot_3workers"),
    )
)
PLAN_PATH = RESULT_ROOT / "FINAL_TEST_3WORKER_PROTOCOL.json"
FROZEN_PATH = RESULT_ROOT / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"
CALIBRATION_PATH = RESULT_ROOT / "DETECTOR_DERIVED_CUE_CALIBRATION.json"
FINAL_LOCK = RESULT_ROOT / "FINAL_TEST_DONE.json"
FINAL_JSON = RESULT_ROOT / "FINAL_TEST_RESULTS_3WORKER.json"
FINAL_CSV = RESULT_ROOT / "FINAL_TEST_COMPARISON_3WORKER.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if FINAL_LOCK.exists():
    print("FINAL TEST ALREADY MERGED AND LOCKED")
    print(FINAL_LOCK.read_text())
    raise SystemExit(0)
for p in [PLAN_PATH, FROZEN_PATH, CALIBRATION_PATH]:
    if not p.exists():
        raise RuntimeError(f"Missing required artifact: {p}")

PLAN = json.loads(PLAN_PATH.read_text())
protocol_sha = sha256(PLAN_PATH)
frozen_sha = sha256(FROZEN_PATH)
calibration_sha = sha256(CALIBRATION_PATH)

worker_files = []
for spec in PLAN["systems"]:
    wid = int(spec["worker_id"])
    name = spec["name"]
    path = RESULT_ROOT / f"FINAL_TEST_WORKER_{wid}_{name.upper()}.json"
    worker_files.append((wid, name, path))

missing = [str(p) for _, _, p in worker_files if not p.exists()]
if missing:
    raise RuntimeError("Cannot merge; missing final-test worker results:\n- " + "\n- ".join(missing))

results = {}
gpus = {}
for wid, expected_name, path in worker_files:
    data = json.loads(path.read_text())
    if data.get("status") != "FINAL_TEST_WORKER_COMPLETE":
        raise RuntimeError(f"Worker {wid} result is not complete: {path}")
    if int(data.get("worker_id", -1)) != wid or data.get("system") != expected_name:
        raise RuntimeError(f"Worker/system mismatch in {path}")
    if data.get("protocol_sha256") != protocol_sha:
        raise RuntimeError(f"Protocol hash mismatch in {path}")
    if data.get("frozen_config_sha256") != frozen_sha:
        raise RuntimeError(f"Frozen config hash mismatch in {path}")
    if data.get("calibration_sha256") != calibration_sha:
        raise RuntimeError(f"Calibration hash mismatch in {path}")
    results[expected_name] = data["metrics"]
    gpus[expected_name] = data.get("gpu", "unknown")

ordered_names = [x["name"] for x in PLAN["systems"]]
with FINAL_CSV.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["system", "MOTA", "HOTA", "IDF1", "IDS", "FN", "FP", "FPS", "GPU"])
    for name in ordered_names:
        m = results[name]
        writer.writerow([
            name, m["MOTA"], m["HOTA"], m["IDF1"], m["IDS"],
            m["FN"], m["FP"], m["FPS"], gpus[name],
        ])

baseline = results["Baseline_Default"]
old = results["Old_ACMOT_Frozen"]
new = results["New_ACMOT_Frozen"]

def delta(a, b):
    return {
        "MOTA": a["MOTA"] - b["MOTA"],
        "HOTA": a["HOTA"] - b["HOTA"],
        "IDF1": a["IDF1"] - b["IDF1"],
        "IDS": a["IDS"] - b["IDS"],
        "FN": a["FN"] - b["FN"],
        "FP": a["FP"] - b["FP"],
        "FPS": a["FPS"] - b["FPS"],
    }

final = {
    "status": "FINAL_TEST_DONE",
    "protocol_version": PLAN["protocol_version"],
    "protocol_sha256": protocol_sha,
    "protocol": "custom class-agnostic AC-MOT evaluation; not official VisDrone leaderboard",
    "systems": ordered_names,
    "results": results,
    "gpu_by_system": gpus,
    "deltas": {
        "New_minus_Baseline": delta(new, baseline),
        "New_minus_Old_ACMOT": delta(new, old),
        "Old_ACMOT_minus_Baseline": delta(old, baseline),
    },
    "selection_or_tuning_on_test": False,
    "warning": "Held-out results are now exposed. Do not retune and later label another run as an unbiased final test.",
}
FINAL_JSON.write_text(json.dumps(final, indent=2))
FINAL_LOCK.write_text(json.dumps(final, indent=2))

print("\n" + "=" * 110)
print("THREE-WORKER HELD-OUT FINAL TEST MERGED — FINAL LOCK CREATED")
print(json.dumps(final, indent=2))
print("CSV :", FINAL_CSV)
print("JSON:", FINAL_JSON)
print("LOCK:", FINAL_LOCK)
print("DO NOT RETUNE ON THESE RESULTS.")
print("=" * 110)
