#!/usr/bin/env python3
"""Serial, artifact-gated continuation supervisor for the V5-TF project.

This process never starts or restarts cache queues or the existing V5-TF
waiter. It waits for that waiter, validates E36/E39 from their complete output
sets, runs context checks, then invokes exactly one non-interactive Codex agent
at a time from the canonical prompts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / "research" / "context"
OUT = ROOT / "outputs" / "autonomous_v5tf"
LOCK_DIR = OUT / "repo_writer.lock"
LOCK_PID = LOCK_DIR / "pid"
STATE_FILE = CTX / "AGENT_STATE.md"
STATUS_FILE = OUT / "status.json"
LOG_FILE = OUT / "supervisor.log"
POLL_SECONDS = 60
MAX_TRANSIENT_BLOCKER_RETRIES = 6
CURRENT_CHILD: subprocess.Popen[str] | None = None
STOP_REQUESTED = False

CACHE_DIRS = (
    "outputs/det_cache_train_native/yolov8",
    "outputs/det_cache_train/rtdetr",
    "outputs/det_cache_train/visual_cues",
    "outputs/det_cache_train_res_extra/yolov8",
    "outputs/det_cache_train_res_extra/rtdetr",
)
E36_SYSTEMS = ("V4", "static_default", "shared_static", "F1", "F2", "F3", "F5", "F5R")
DETECTORS = ("yolov8", "rtdetr")
SENS = {
    "otsu_bins": (32, 128),
    "gate_window": (5, 20),
    "tf_history": (50, 200),
    "tf_warmup": (3, 10),
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{utc_now()} {message}"
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def acquire_lock() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        LOCK_DIR.mkdir()
    except FileExistsError:
        try:
            existing = int(LOCK_PID.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            existing = -1
        if existing > 0 and pid_alive(existing):
            raise SystemExit(f"supervisor already holds lock (pid {existing})")
        # Exact, repository-local stale lock only.
        try:
            LOCK_PID.unlink(missing_ok=True)
            LOCK_DIR.rmdir()
            LOCK_DIR.mkdir()
        except OSError as exc:
            raise SystemExit(f"cannot recover stale supervisor lock: {exc}") from exc
    LOCK_PID.write_text(f"{os.getpid()}\n", encoding="utf-8")


def release_lock() -> None:
    try:
        if LOCK_PID.exists() and LOCK_PID.read_text(encoding="utf-8").strip() == str(os.getpid()):
            LOCK_PID.unlink()
            LOCK_DIR.rmdir()
    except OSError:
        pass


def read_agent_state() -> dict[str, str]:
    state: dict[str, str] = {}
    if STATE_FILE.exists():
        for line in STATE_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if key and key.replace("_", "").isalnum():
                    state[key] = value
    return state


def write_agent_state(**updates: str) -> None:
    state = read_agent_state()
    state.update({key: str(value).replace("\n", " ") for key, value in updates.items()})
    state["UPDATED_AT"] = utc_now()
    order = (
        "PROJECT_COMPLETE", "STATUS", "CURRENT_STAGE", "LAST_COMPLETED_STAGE",
        "NEXT_STAGE", "CURRENT_AGENT_PROCESS", "BLOCKER", "UPDATED_AT",
    )
    body = [
        "# AUTONOMOUS AGENT STATE",
        "",
        "This file is the machine-readable handoff between the V5-TF supervisor and",
        "successive non-interactive Codex runs. Values after `=` must remain one line.",
        "",
    ]
    body.extend(f"{key}={state.get(key, '')}" for key in order)
    STATE_FILE.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_status(*, current_stage: str, last_completed: str, next_stage: str,
                 process: str, blocker: str = "none") -> None:
    payload = {
        "updated_at": utc_now(),
        "current_stage": current_stage,
        "last_completed_stage": last_completed,
        "next_stage": next_stage,
        "current_agent_process": process,
        "blocker": blocker,
        "supervisor_pid": os.getpid(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    append_log(
        f"stage={current_stage} last={last_completed} next={next_stage} "
        f"process={process} blocker={blocker}"
    )


def process_rows() -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-Aww", "-o", "pid=,command="], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    rows: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            rows.append((int(parts[0]), parts[1]))
        except ValueError:
            continue
    return rows


def find_waiter_pid() -> int | None:
    for pid, command in process_rows():
        if pid != os.getpid() and "tools/v5tf_waiter.sh" in command:
            return pid
    return None


def other_codex_exec_running(own_child: int | None = None) -> tuple[int, str] | None:
    for pid, command in process_rows():
        if pid in {os.getpid(), own_child}:
            continue
        if "codex exec" in command and "autonomous_v5tf_supervisor" not in command:
            return pid, command
    return None


def count_npz(relative: str) -> int:
    return len(list((ROOT / relative).glob("*.npz")))


def load_development_sequences() -> list[str]:
    split = json.loads((ROOT / "research" / "TRAIN_SPLIT_V5.json").read_text(encoding="utf-8"))
    development = split.get("development")
    if not isinstance(development, list) or len(development) != 40:
        raise RuntimeError("TRAIN_SPLIT_V5.json does not contain the declared development-40")
    return [str(item) for item in development]


def validate_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty artifact: {path.relative_to(ROOT)}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON artifact: {path.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"artifact is not a JSON object: {path.relative_to(ROOT)}")
    return value


def validate_e36_e39() -> dict[str, object]:
    cache_counts = {relative: count_npz(relative) for relative in CACHE_DIRS}
    incomplete_caches = {key: value for key, value in cache_counts.items() if value < 56}
    if incomplete_caches:
        raise RuntimeError(f"required caches incomplete: {incomplete_caches}")

    merged_counts = {
        "outputs/det_cache_train_res/yolov8": count_npz("outputs/det_cache_train_res/yolov8"),
        "outputs/det_cache_train_res/rtdetr": count_npz("outputs/det_cache_train_res/rtdetr"),
    }
    if any(value < 56 for value in merged_counts.values()):
        raise RuntimeError(f"merged caches incomplete: {merged_counts}")
    if not (ROOT / "outputs/det_cache_train_res/visual_cues").exists():
        raise RuntimeError("merged visual-cue cache is missing")

    sequences = load_development_sequences()
    v5out = ROOT / "outputs" / "v5tf_dev"
    missing_e36 = [
        str((v5out / system / detector / f"{sequence}.pkl").relative_to(ROOT))
        for system in E36_SYSTEMS for detector in DETECTORS for sequence in sequences
        if not (v5out / system / detector / f"{sequence}.pkl").is_file()
        or (v5out / system / detector / f"{sequence}.pkl").stat().st_size == 0
    ]
    if missing_e36:
        raise RuntimeError(f"E36 artifacts incomplete ({len(missing_e36)} missing); first={missing_e36[0]}")

    choice = validate_json(v5out / "family_choice.json")
    family = choice.get("choice")
    keys = choice.get("keys")
    if family not in {"F3", "F5"} or not isinstance(keys, dict) or not {"F3", "F5"}.issubset(keys):
        raise RuntimeError("family_choice.json does not contain the declared F3/F5 choice")

    variants = [f"S:{family}:{key}={value}" for key, values in SENS.items() for value in values]
    missing_e39 = [
        str((v5out / variant / detector / f"{sequence}.pkl").relative_to(ROOT))
        for variant in variants for detector in DETECTORS for sequence in sequences
        if not (v5out / variant / detector / f"{sequence}.pkl").is_file()
        or (v5out / variant / detector / f"{sequence}.pkl").stat().st_size == 0
    ]
    if missing_e39:
        raise RuntimeError(f"E39 artifacts incomplete ({len(missing_e39)} missing); first={missing_e39[0]}")

    audit = validate_json(v5out / "constant_audit.json")
    verdict = audit.get("verdict")
    if audit.get("family") != family or not isinstance(verdict, dict) or set(verdict) != set(SENS):
        raise RuntimeError("constant_audit.json does not contain all four declared constants")

    return {
        "cache_counts": cache_counts,
        "merged_counts": merged_counts,
        "e36_pickle_count": len(E36_SYSTEMS) * len(DETECTORS) * len(sequences),
        "e39_pickle_count": len(variants) * len(DETECTORS) * len(sequences),
        "family": family,
        "constant_verdict": verdict,
    }


def run_logged(command: list[str], label: str) -> int:
    append_log(f"run={label} command={' '.join(command)}")
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    append_log(f"run={label} exit={completed.returncode}")
    return completed.returncode


def refresh_and_check_context() -> bool:
    if run_logged([sys.executable, "tools/update_project_context.py"], "update_project_context") != 0:
        return False
    return run_logged([sys.executable, "tools/context_health_check.py"], "context_health_check") == 0


def codex_binary() -> str:
    found = shutil.which("codex")
    if found:
        return found
    fallback = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("Codex CLI not found")


def canonical_prompt() -> str:
    autonomous = (CTX / "AUTONOMOUS_AGENT_PROMPT.md").read_text(encoding="utf-8")
    session = (CTX / "NEW_SESSION_PROMPT.md").read_text(encoding="utf-8")
    return autonomous + "\n\n--- CANONICAL NEW SESSION PROMPT ---\n\n" + session


def state_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in (
        "research/context/NEXT_STEPS.md",
        "research/context/PROJECT_STATE.md",
        "research/context/PROJECT_COMPLETION.md",
    ):
        path = ROOT / relative
        if path.exists():
            digest.update(path.read_bytes())
    state = read_agent_state()
    for key in ("PROJECT_COMPLETE", "STATUS", "CURRENT_STAGE",
                "LAST_COMPLETED_STAGE", "NEXT_STAGE", "BLOCKER"):
        digest.update(f"{key}={state.get(key, '')}\n".encode("utf-8"))
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    digest.update(result.stdout.encode("utf-8"))
    return digest.hexdigest()


def invoke_codex(run_number: int) -> int:
    global CURRENT_CHILD
    while True:
        other = other_codex_exec_running()
        if other is None:
            break
        write_status(
            current_stage="waiting_for_existing_codex_agent",
            last_completed=read_agent_state().get("LAST_COMPLETED_STAGE", "E36_E39_artifacts_verified"),
            next_stage="launch_single_autonomous_codex_agent",
            process=f"existing_codex_exec:{other[0]}",
        )
        time.sleep(POLL_SECONDS)

    run_log = OUT / f"agent-run-{run_number:03d}.log"
    last_message = OUT / f"agent-run-{run_number:03d}-last-message.txt"
    command = [
        codex_binary(), "exec", "-C", str(ROOT),
        "-s", "danger-full-access", "-a", "never",
        "-o", str(last_message), "-",
    ]
    with run_log.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.PIPE, stdout=handle,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        CURRENT_CHILD = child
        write_agent_state(
            STATUS="AGENT_RUNNING",
            CURRENT_STAGE="autonomous_codex_continuation",
            CURRENT_AGENT_PROCESS=f"codex:{child.pid}",
            BLOCKER="none",
        )
        state = read_agent_state()
        write_status(
            current_stage="autonomous_codex_continuation",
            last_completed=state.get("LAST_COMPLETED_STAGE", "E36_E39_artifacts_verified"),
            next_stage=state.get("NEXT_STAGE", "continue_NEXT_STEPS"),
            process=f"codex:{child.pid}",
        )
        assert child.stdin is not None
        child.stdin.write(canonical_prompt())
        child.stdin.close()
        return_code = child.wait()
        CURRENT_CHILD = None
        if STOP_REQUESTED:
            raise SystemExit(0)
    append_log(f"agent_run={run_number} pid={child.pid} exit={return_code} log={run_log.relative_to(ROOT)}")
    return return_code


def set_blocker(message: str, current_stage: str) -> None:
    state = read_agent_state()
    write_agent_state(
        STATUS="BLOCKED",
        CURRENT_STAGE=current_stage,
        CURRENT_AGENT_PROCESS=f"supervisor:{os.getpid()}",
        BLOCKER=message,
    )
    write_status(
        current_stage=current_stage,
        last_completed=state.get("LAST_COMPLETED_STAGE", "unknown"),
        next_stage=state.get("NEXT_STAGE", "unknown"),
        process=f"supervisor:{os.getpid()}",
        blocker=message,
    )


def wait_for_e36_e39() -> dict[str, object]:
    waiter_pid = find_waiter_pid()
    if waiter_pid is None:
        try:
            return validate_e36_e39()
        except RuntimeError as exc:
            raise RuntimeError(f"current waiter is not running and artifacts are incomplete: {exc}") from exc

    append_log(f"observed_current_waiter_pid={waiter_pid}; no process will be restarted")
    while True:
        try:
            artifacts = validate_e36_e39()
        except RuntimeError as exc:
            if not pid_alive(waiter_pid):
                raise RuntimeError(f"waiter pid {waiter_pid} exited before valid E36/E39 artifacts: {exc}") from exc
            state = read_agent_state()
            write_status(
                current_stage="waiting_for_current_v5tf_waiter",
                last_completed=state.get("LAST_COMPLETED_STAGE", "live_motion_fix_verified"),
                next_stage="verify_E36_E39_artifacts_then_refresh_context",
                process=f"waiter:{waiter_pid};supervisor:{os.getpid()}",
            )
            time.sleep(POLL_SECONDS)
            continue

        if pid_alive(waiter_pid):
            append_log(f"E36/E39 artifacts validate but waiter pid {waiter_pid} is still finishing")
            time.sleep(5)
            continue
        return artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-artifacts", action="store_true", help="validate current E36/E39 artifacts and exit")
    parser.add_argument("--status", action="store_true", help="print supervisor status and exit")
    args = parser.parse_args()

    if args.status:
        if STATUS_FILE.exists():
            print(STATUS_FILE.read_text(encoding="utf-8"), end="")
            return 0
        print("supervisor status not created yet")
        return 1
    if args.check_artifacts:
        try:
            print(json.dumps(validate_e36_e39(), indent=2))
            return 0
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    acquire_lock()

    def stop_safely(signum, _frame) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True
        child = CURRENT_CHILD
        append_log(f"safe_stop_requested signal={signum}")
        if child is not None and child.poll() is None:
            append_log(f"stopping_active_codex_process_group={child.pid}")
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                append_log(f"child_process_group_did_not_stop_cleanly={child.pid}")
                state = read_agent_state()
                write_status(
                    current_stage="safe_stop_waiting_for_active_child",
                    last_completed=state.get("LAST_COMPLETED_STAGE", "unknown"),
                    next_stage=state.get("NEXT_STAGE", "resume_from_canonical_context"),
                    process=f"codex:{child.pid};supervisor:{os.getpid()}",
                    blocker="TRANSIENT: active Codex child is still shutting down",
                )
                return
        state = read_agent_state()
        write_agent_state(
            STATUS="STOPPED_BY_USER",
            CURRENT_STAGE="autonomous_supervisor_stopped",
            NEXT_STAGE=state.get("NEXT_STAGE", "resume_from_canonical_context"),
            CURRENT_AGENT_PROCESS="none",
            BLOCKER="none",
        )
        write_status(
            current_stage="autonomous_supervisor_stopped",
            last_completed=state.get("LAST_COMPLETED_STAGE", "unknown"),
            next_stage=state.get("NEXT_STAGE", "resume_from_canonical_context"),
            process="none",
            blocker="none",
        )
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_safely)
    signal.signal(signal.SIGINT, stop_safely)
    try:
        write_agent_state(
            PROJECT_COMPLETE="false",
            STATUS="WAITING_FOR_E36_E39",
            CURRENT_STAGE="waiting_for_current_v5tf_waiter",
            CURRENT_AGENT_PROCESS=f"supervisor:{os.getpid()}",
            BLOCKER="none",
        )
        try:
            artifacts = wait_for_e36_e39()
        except RuntimeError as exc:
            set_blocker(str(exc), "E36_E39_artifact_verification")
            return 2

        append_log("E36_E39_ARTIFACTS_VERIFIED " + json.dumps(artifacts, sort_keys=True))
        write_agent_state(
            STATUS="READY_FOR_AGENT",
            CURRENT_STAGE="E36_E39_artifacts_verified",
            LAST_COMPLETED_STAGE="E36_E39_artifacts_verified",
            NEXT_STAGE="record_E36_E39_and_continue_NEXT_STEPS",
            CURRENT_AGENT_PROCESS=f"supervisor:{os.getpid()}",
            BLOCKER="none",
        )
        if not refresh_and_check_context():
            set_blocker("context health check failed after E36/E39 artifact verification",
                        "post_E36_E39_context_health")
            return 3

        failures = 0
        no_progress = 0
        transient_blocker_retries = 0
        run_number = 0
        while True:
            state = read_agent_state()
            if state.get("PROJECT_COMPLETE", "false").lower() == "true":
                write_status(
                    current_stage="project_complete",
                    last_completed=state.get("LAST_COMPLETED_STAGE", "project_complete"),
                    next_stage="none",
                    process=f"supervisor:{os.getpid()}",
                )
                append_log("PROJECT_COMPLETE=true; supervisor stopping")
                return 0
            blocker = state.get("BLOCKER", "none")
            if blocker.lower() not in {"", "none"}:
                if blocker.upper().startswith("TRANSIENT:"):
                    transient_blocker_retries += 1
                    if transient_blocker_retries <= MAX_TRANSIENT_BLOCKER_RETRIES:
                        delay = min(60 * (2 ** (transient_blocker_retries - 1)), 1800)
                        write_status(
                            current_stage="transient_blocker_backoff",
                            last_completed=state.get("LAST_COMPLETED_STAGE", "unknown"),
                            next_stage=state.get("NEXT_STAGE", "retry_current_stage"),
                            process=f"supervisor:{os.getpid()}",
                            blocker=f"{blocker}; retry {transient_blocker_retries}/{MAX_TRANSIENT_BLOCKER_RETRIES} in {delay}s",
                        )
                        time.sleep(delay)
                        write_agent_state(
                            STATUS="RETRYING_TRANSIENT_BLOCKER",
                            CURRENT_STAGE=state.get("NEXT_STAGE", "retry_current_stage"),
                            CURRENT_AGENT_PROCESS=f"supervisor:{os.getpid()}",
                            BLOCKER="none",
                        )
                        continue
                    blocker = f"EXTERNAL: transient blocker persisted after {MAX_TRANSIENT_BLOCKER_RETRIES} retries: {blocker}"
                write_status(
                    current_stage=state.get("CURRENT_STAGE", "blocked"),
                    last_completed=state.get("LAST_COMPLETED_STAGE", "unknown"),
                    next_stage=state.get("NEXT_STAGE", "unknown"),
                    process=f"supervisor:{os.getpid()}",
                    blocker=blocker,
                )
                append_log(f"BLOCKER={blocker}; supervisor stopping")
                return 4

            before = state_fingerprint()
            run_number += 1
            return_code = invoke_codex(run_number)
            write_agent_state(CURRENT_AGENT_PROCESS=f"supervisor:{os.getpid()}")
            if return_code != 0:
                failures += 1
                append_log(f"Codex run failed ({failures}/3)")
                if failures >= 3:
                    set_blocker("non-interactive Codex failed three consecutive times",
                                "autonomous_codex_invocation")
                    return 5
                time.sleep(min(60 * failures, 180))
                continue
            failures = 0

            if not refresh_and_check_context():
                set_blocker("context health check failed after autonomous Codex run",
                            "post_agent_context_health")
                return 6

            after = state_fingerprint()
            if after == before:
                no_progress += 1
                append_log(f"agent made no detectable state progress ({no_progress}/2)")
                if no_progress >= 2:
                    set_blocker("two successful Codex runs made no detectable repository/state progress",
                                "autonomous_no_progress_guard")
                    return 7
            else:
                no_progress = 0
                transient_blocker_retries = 0
            time.sleep(15)
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
