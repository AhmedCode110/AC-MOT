"""Live-progress runner for AC-MOT portable joint Optuna search in Google Colab.

This wrapper does not change the scientific method. It only improves runtime
visibility while delegating all optimization/evaluation work to
scripts/optuna_sci_joint_portable_colab.py.

It provides:
- denser per-frame progress by default (every 50 measured frames),
- current trial / total trials,
- overall Optuna percentage,
- completed-trial ETA based on measured trial durations,
- best feasible MOTA / IDS / FPS seen so far,
- live streaming of the underlying sequence/frame SCI progress.

Usage in Colab:
    from google.colab import drive
    drive.mount('/content/drive')

    # clone/pull repo first, then:
    import os
    os.environ['ACMOT_OPTUNA_TRIALS'] = '2'   # use 50 for full search
    os.environ['ACMOT_SMOKE_TEST'] = '1'      # use 0 for full search
    %run /content/AC-MOT/scripts/run_joint_optuna_progress_colab.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("ACMOT_REPO", "/content/AC-MOT"))
TARGET = ROOT / "scripts" / "optuna_sci_joint_portable_colab.py"

if not TARGET.exists():
    raise RuntimeError(
        f"Joint Optuna script not found: {TARGET}\n"
        "Update the repository first with: git -C /content/AC-MOT pull --ff-only"
    )

TOTAL_TRIALS = int(os.environ.get("ACMOT_OPTUNA_TRIALS", "50"))

# More frequent frame-level progress than the base script's conservative default.
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


def print_overall(
    completed: int,
    total: int,
    started_at: float,
    trial_durations: list[float],
    best: dict | None,
) -> None:
    pct = 100.0 * completed / total if total else 0.0
    elapsed = time.perf_counter() - started_at
    eta = None
    if trial_durations and completed < total:
        # Measured ETA only: average duration of completed trials.
        eta = (sum(trial_durations) / len(trial_durations)) * (total - completed)

    print("\n" + "-" * 96, flush=True)
    print(
        f"[OVERALL OPTUNA] {bar(completed, total)} {pct:6.2f}% | "
        f"completed={completed}/{total} | elapsed={fmt_seconds(elapsed)} | ETA={fmt_seconds(eta)}",
        flush=True,
    )
    if best is None:
        print("[BEST FEASIBLE] none yet (must satisfy FPS>=25 and IDS<=Old-A3)", flush=True)
    else:
        print(
            "[BEST FEASIBLE] "
            f"trial={best['trial']:03d} | MOTA={best['MOTA']:.3f}% | "
            f"IDS={best['IDS']} | FPS={best['FPS']:.2f}",
            flush=True,
        )
    print("-" * 96 + "\n", flush=True)


print("=" * 96, flush=True)
print("AC-MOT JOINT OPTUNA — LIVE PROGRESS RUNNER", flush=True)
print(f"Trials              : {TOTAL_TRIALS}", flush=True)
print(f"Frame progress every: {os.environ['ACMOT_PROGRESS_EVERY']} frames", flush=True)
print(f"Smoke test          : {os.environ.get('ACMOT_SMOKE_TEST', '0')}", flush=True)
print("Validation only     : YES", flush=True)
print("Test-dev            : NOT ACCESSED BY THE OPTIMIZATION SCRIPT", flush=True)
print("=" * 96, flush=True)

cmd = [sys.executable, "-u", str(TARGET)]
env = os.environ.copy()

process = subprocess.Popen(
    cmd,
    cwd=str(ROOT),
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

started_at = time.perf_counter()
trial_started_at: float | None = None
trial_durations: list[float] = []
current_trial: int | None = None
completed = 0

current_result: dict[str, float | int | bool] = {}
best: dict | None = None

trial_header_re = re.compile(r"JOINT OPTUNA TRIAL\s+(\d+)\s*/\s*(\d+)")
trial_result_re = re.compile(r"\[TRIAL\s+(\d+)\]")

assert process.stdout is not None

for raw_line in process.stdout:
    line = raw_line.rstrip("\n")
    print(raw_line, end="", flush=True)

    m = trial_header_re.search(line)
    if m:
        one_based = int(m.group(1))
        current_trial = one_based - 1
        trial_started_at = time.perf_counter()
        current_result = {"trial": current_trial}
        print(
            f"\n[TRIAL START] {bar(one_based - 1, TOTAL_TRIALS)} "
            f"trial={one_based}/{TOTAL_TRIALS}\n",
            flush=True,
        )
        continue

    m = trial_result_re.search(line)
    if m:
        current_trial = int(m.group(1))
        current_result["trial"] = current_trial
        continue

    stripped = line.strip()
    try:
        if stripped.startswith("MOTA ="):
            current_result["MOTA"] = float(stripped.split("=", 1)[1].replace("%", "").strip())
        elif stripped.startswith("IDS ="):
            # Example: IDS = 268 (limit 271)
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
                    candidate["MOTA"],
                    -candidate["IDS"],
                    candidate["FPS"],
                ) > (
                    best["MOTA"],
                    -best["IDS"],
                    best["FPS"],
                ):
                    best = candidate

            print_overall(
                completed=completed,
                total=TOTAL_TRIALS,
                started_at=started_at,
                trial_durations=trial_durations,
                best=best,
            )
    except Exception:
        # Progress parsing must never interrupt the scientific run.
        pass

return_code = process.wait()

print("\n" + "=" * 96, flush=True)
if return_code == 0:
    print("AC-MOT JOINT OPTUNA FINISHED SUCCESSFULLY", flush=True)
else:
    print(f"AC-MOT JOINT OPTUNA FAILED | return_code={return_code}", flush=True)
print_overall(
    completed=completed,
    total=TOTAL_TRIALS,
    started_at=started_at,
    trial_durations=trial_durations,
    best=best,
)
print("=" * 96, flush=True)

if return_code != 0:
    raise SystemExit(return_code)
