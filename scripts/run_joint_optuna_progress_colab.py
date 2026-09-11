"""Live-progress runner for the complete AC-MOT validation workflow in Colab.

Stages
------
1) Temporal ablation: choose SCI smoothing_window and analysis_stride on validation.
2) Freeze/reuse the temporal choice.
3) Joint Optuna: tune SCI weights + detector-control mapping on validation.

The final test set is never accessed by either optimization stage.

Smoke mode:
- runs a 4-combination temporal smoke test,
- does NOT freeze temporal settings,
- then runs the requested small joint-Optuna smoke test using W=7/S=10.

Full mode:
- runs/reuses the full 25-combination temporal ablation,
- exports the selected W/S to the joint Optuna stage,
- then runs the requested joint Optuna budget.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(os.environ.get("ACMOT_REPO", "/content/AC-MOT"))
JOINT_TARGET = ROOT / "scripts" / "optuna_sci_joint_portable_colab.py"
TEMPORAL_TARGET = ROOT / "scripts" / "temporal_ablation_portable_colab.py"
DRIVE = Path(os.environ.get("ACMOT_DRIVE", "/content/drive/MyDrive"))
RESULT_ROOT = Path(
    os.environ.get("ACMOT_RESULT_ROOT", str(DRIVE / "AC-MOT-results" / "optuna_sci_joint"))
)
TEMPORAL_FROZEN = RESULT_ROOT / "FROZEN_TEMPORAL_CONFIG.json"

for target in (JOINT_TARGET, TEMPORAL_TARGET):
    if not target.exists():
        raise RuntimeError(
            f"Required script not found: {target}\n"
            "Update the repository first with: git -C /content/AC-MOT pull --ff-only"
        )

TOTAL_TRIALS = int(os.environ.get("ACMOT_OPTUNA_TRIALS", "50"))
SMOKE_TEST = os.environ.get("ACMOT_SMOKE_TEST", "0") == "1"
HEARTBEAT_SECONDS = float(os.environ.get("ACMOT_HEARTBEAT_SECONDS", "5"))

# Dense frame-level progress from paper_eval_v17.
os.environ.setdefault("ACMOT_PROGRESS_EVERY", "50")


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(ratio * width)
    return "[" + "=" * filled + ">" * (filled < width) + "." * max(width - filled - 1, 0) + "]"


def stream_process(cmd, env, label, parse_line=None):
    """Stream a child process and emit heartbeats during silent phases."""
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    q: queue.Queue[str | None] = queue.Queue()

    def reader():
        try:
            for line in process.stdout:
                q.put(line)
        finally:
            q.put(None)

    threading.Thread(target=reader, daemon=True).start()

    start = time.perf_counter()
    last_output = start
    finished_stream = False

    while not finished_stream:
        try:
            item = q.get(timeout=1.0)
        except queue.Empty:
            item = "__NO_LINE__"

        now = time.perf_counter()
        if item is None:
            finished_stream = True
            continue

        if item == "__NO_LINE__":
            silent = now - last_output
            if silent >= HEARTBEAT_SECONDS:
                print(
                    f"[HEARTBEAT] stage={label} | process=RUNNING | "
                    f"silent={silent:.0f}s | total_elapsed={fmt_seconds(now-start)}",
                    flush=True,
                )
                last_output = now
            continue

        print(item, end="", flush=True)
        last_output = now
        if parse_line is not None:
            parse_line(item.rstrip("\n"))

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{label} failed with return code {return_code}")
    return return_code


# =============================================================================
# STAGE 1 — TEMPORAL ABLATION
# =============================================================================
print("=" * 100, flush=True)
print("AC-MOT COMPLETE VALIDATION WORKFLOW", flush=True)
print(f"Joint Optuna trials      : {TOTAL_TRIALS}", flush=True)
print(f"Frame progress every     : {os.environ['ACMOT_PROGRESS_EVERY']} frames", flush=True)
print(f"Smoke test               : {int(SMOKE_TEST)}", flush=True)
print("Validation only          : YES", flush=True)
print("Test-dev                 : NOT ACCESSED", flush=True)
print("=" * 100, flush=True)

child_env = os.environ.copy()

if SMOKE_TEST:
    run_temporal_smoke = os.environ.get("ACMOT_SKIP_TEMPORAL_SMOKE", "0") != "1"
    if run_temporal_smoke:
        print("\n[STAGE 1/2] Temporal ablation SMOKE TEST: 4 combinations, no freeze.\n", flush=True)
        temporal_env = child_env.copy()
        temporal_env["ACMOT_TEMPORAL_SMOKE"] = "1"
        stream_process(
            [sys.executable, "-u", str(TEMPORAL_TARGET)],
            temporal_env,
            "temporal smoke ablation",
        )
    else:
        print("[STAGE 1/2] Temporal smoke explicitly skipped.", flush=True)

    # Smoke mode validates the pipeline but does not make a scientific selection.
    # Keep the old W=7/S=10 only for this smoke run.
    child_env["ACMOT_SMOOTHING_WINDOW"] = "7"
    child_env["ACMOT_ANALYSIS_STRIDE"] = "10"
    print("[SMOKE] Joint stage uses temporary W=7/S=10; nothing is frozen.", flush=True)

else:
    if TEMPORAL_FROZEN.exists():
        frozen = json.loads(TEMPORAL_FROZEN.read_text())
        selected = frozen["selected"]
        print("\n[STAGE 1/2] Reusing frozen validation temporal selection.", flush=True)
    else:
        print("\n[STAGE 1/2] Full temporal ablation: 25 validation combinations.\n", flush=True)
        temporal_env = child_env.copy()
        temporal_env["ACMOT_TEMPORAL_SMOKE"] = "0"
        stream_process(
            [sys.executable, "-u", str(TEMPORAL_TARGET)],
            temporal_env,
            "full temporal ablation",
        )
        if not TEMPORAL_FROZEN.exists():
            raise RuntimeError("Temporal ablation finished but FROZEN_TEMPORAL_CONFIG.json was not created.")
        frozen = json.loads(TEMPORAL_FROZEN.read_text())
        selected = frozen["selected"]

    child_env["ACMOT_SMOOTHING_WINDOW"] = str(int(selected["smoothing_window"]))
    child_env["ACMOT_ANALYSIS_STRIDE"] = str(int(selected["analysis_stride"]))

    print(
        "[TEMPORAL SELECTED] "
        f"smoothing_window={child_env['ACMOT_SMOOTHING_WINDOW']} | "
        f"analysis_stride={child_env['ACMOT_ANALYSIS_STRIDE']} | "
        f"analysis_interval={float(selected['analysis_interval_seconds']):.3f}s | "
        f"history_span={float(selected['causal_history_span_seconds']):.3f}s",
        flush=True,
    )


# =============================================================================
# STAGE 2 — JOINT OPTUNA
# =============================================================================
print("\n" + "=" * 100, flush=True)
print("[STAGE 2/2] JOINT SCI + DETECTOR-CONTROL OPTUNA", flush=True)
print("=" * 100, flush=True)
print("Trials              :", TOTAL_TRIALS, flush=True)
print("Smoothing window    :", child_env["ACMOT_SMOOTHING_WINDOW"], flush=True)
print("Analysis stride     :", child_env["ACMOT_ANALYSIS_STRIDE"], flush=True)
print("Smoke test          :", int(SMOKE_TEST), flush=True)
print("FPS gate            : >=", child_env.get("ACMOT_MIN_FPS", "25"), flush=True)
print("Test-dev            : NOT ACCESSED", flush=True)
print("=" * 100, flush=True)

trial_started_at: float | None = None
trial_durations: list[float] = []
completed = 0
current_result: dict[str, float | int | bool] = {}
best: dict | None = None
joint_started = time.perf_counter()

trial_header_re = re.compile(r"JOINT OPTUNA TRIAL\s+(\d+)\s*/\s*(\d+)")
trial_result_re = re.compile(r"\[TRIAL\s+(\d+)\]")


def print_overall():
    pct = 100.0 * completed / TOTAL_TRIALS if TOTAL_TRIALS else 0.0
    elapsed = time.perf_counter() - joint_started
    eta = None
    if trial_durations and completed < TOTAL_TRIALS:
        eta = (sum(trial_durations) / len(trial_durations)) * (TOTAL_TRIALS - completed)

    print("\n" + "-" * 100, flush=True)
    print(
        f"[OVERALL OPTUNA] {bar(completed, TOTAL_TRIALS)} {pct:6.2f}% | "
        f"completed={completed}/{TOTAL_TRIALS} | elapsed={fmt_seconds(elapsed)} | ETA={fmt_seconds(eta)}",
        flush=True,
    )
    if best is None:
        print("[BEST FEASIBLE] none yet | requirement: FPS>=25 and IDS<=Old-A3", flush=True)
    else:
        print(
            f"[BEST FEASIBLE] trial={best['trial']:03d} | MOTA={best['MOTA']:.3f}% | "
            f"IDS={best['IDS']} | FPS={best['FPS']:.2f}",
            flush=True,
        )
    print("-" * 100 + "\n", flush=True)


def parse_joint_line(line: str):
    global trial_started_at, completed, current_result, best

    m = trial_header_re.search(line)
    if m:
        one_based = int(m.group(1))
        current_result = {"trial": one_based - 1}
        trial_started_at = time.perf_counter()
        print(
            f"[TRIAL START] {bar(one_based - 1, TOTAL_TRIALS)} trial={one_based}/{TOTAL_TRIALS}",
            flush=True,
        )
        return

    m = trial_result_re.search(line)
    if m:
        current_result["trial"] = int(m.group(1))
        return

    stripped = line.strip()
    try:
        if stripped.startswith("MOTA ="):
            current_result["MOTA"] = float(stripped.split("=", 1)[1].replace("%", "").strip())
        elif stripped.startswith("IDS ="):
            current_result["IDS"] = int(stripped.split("=", 1)[1].strip().split()[0])
        elif stripped.startswith("FPS ="):
            current_result["FPS"] = float(stripped.split("=", 1)[1].strip().split()[0])
        elif stripped.startswith("FEASIBLE ="):
            feasible = stripped.split("=", 1)[1].strip().lower() == "true"
            current_result["FEASIBLE"] = feasible

            if trial_started_at is not None:
                trial_durations.append(time.perf_counter() - trial_started_at)
                trial_started_at = None
            completed += 1

            if feasible and all(k in current_result for k in ("trial", "MOTA", "IDS", "FPS")):
                candidate = {
                    "trial": int(current_result["trial"]),
                    "MOTA": float(current_result["MOTA"]),
                    "IDS": int(current_result["IDS"]),
                    "FPS": float(current_result["FPS"]),
                }
                if best is None or (
                    candidate["MOTA"], -candidate["IDS"], candidate["FPS"]
                ) > (
                    best["MOTA"], -best["IDS"], best["FPS"]
                ):
                    best = candidate
            print_overall()
    except Exception:
        # Progress parsing must never interrupt the scientific run.
        pass


stream_process(
    [sys.executable, "-u", str(JOINT_TARGET)],
    child_env,
    "joint Optuna optimization",
    parse_line=parse_joint_line,
)

print("\n" + "=" * 100, flush=True)
print("AC-MOT VALIDATION WORKFLOW FINISHED SUCCESSFULLY", flush=True)
print_overall()
print("Test-dev was NOT accessed by this workflow.", flush=True)
print("=" * 100, flush=True)
