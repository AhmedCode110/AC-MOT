"""AC-MOT v14 realtime throughput + bottleneck profiler.

v14 preserves the v13 research controller/tracker configuration, but adds three
deployment-only optimizations:
1) detector confidence pruning uses the controller's final keep threshold,
2) TensorRT FP16 is preferred automatically on Tesla T4 with PyTorch FP16 fallback,
3) an FPS-aware resolution cap can lower only the deployment inference size while
   preserving one fresh detector inference per frame.

Every measured frame is profiled into read / analysis / controller / detector /
tracker / total latency. Drive-to-local copy, backend build/export and warmup are
excluded from deployment FPS.
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

print("[BOOT] AC-MOT v14 realtime runner started.", flush=True)
print("[BOOT] Loading NumPy and research modules...", flush=True)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import CLASSES, Config, Controller, atomic_json, boxes
from experiment import dataset_manifest, environment, make_tracker, new_model, sync, track, visual

RUNTIME_KEYS = {
    "realtime_governor", "backend", "min_runtime_size", "detector_budget_fraction"
}


def fmt_seconds(value):
    value = max(float(value), 0.0)
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(int(round(value)), 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def bar(done, total, width=28):
    frac = 0.0 if total <= 0 else min(max(done / total, 0.0), 1.0)
    filled = int(round(frac * width))
    if filled >= width:
        return "[" + "=" * width + "]"
    return "[" + "=" * filled + ">" + "." * max(width - filled - 1, 0) + "]"


def progress(stage, done, total, started, detail=""):
    elapsed = time.perf_counter() - started
    if done > 0 and total > done:
        eta = elapsed / done * (total - done)
        eta_text = fmt_seconds(eta)
    elif done >= total > 0:
        eta_text = "0.0s"
    else:
        eta_text = "estimating..."
    pct = 100.0 * done / total if total else 0.0
    print(
        f"{bar(done, total)} {pct:6.2f}% | {stage} | "
        f"elapsed={fmt_seconds(elapsed)} | ETA={eta_text}"
        + (f" | {detail}" if detail else ""),
        flush=True,
    )


def copy_sequence_to_local(dataset, seq, local_root):
    """Copy one sequence with visible progress; copy time is excluded from FPS."""
    sn = seq["sequence"]
    src = dataset / "sequences" / sn
    dst = local_root / sn
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    filenames = list(seq["frame_sha256"])
    started = time.perf_counter()
    print(f"[COPY] {sn}: Drive -> Colab local SSD. Excluded from FPS.", flush=True)
    for i, filename in enumerate(filenames, 1):
        shutil.copy2(src / filename, dst / filename)
        if i == 1 or i % 25 == 0 or i == len(filenames):
            progress("COPY", i, len(filenames), started, f"sequence={sn} file={filename}")
    paths = [dst / f for f in filenames]
    if len(paths) != seq["frames"] or any(not p.is_file() for p in paths):
        raise RuntimeError(f"Local sequence copy verification failed: {sn}")
    print(f"[OK] Local sequence ready: {sn} | frames={len(paths)}", flush=True)
    return paths


def make_backend(weights, preferred, engine_path):
    """Prefer TensorRT FP16; fall back to PyTorch FP16 without hiding the reason."""
    from ultralytics import YOLO

    preferred = str(preferred).lower()
    if preferred not in {"auto", "tensorrt", "pytorch"}:
        raise ValueError(f"Unknown backend: {preferred}")

    if preferred in {"auto", "tensorrt"}:
        try:
            started = time.perf_counter()
            print("[BACKEND] TensorRT FP16 requested.", flush=True)
            if not engine_path.exists():
                print(
                    "[BACKEND] No cached v14 engine. Exporting dynamic YOLOv8n TensorRT FP16 engine "
                    "(excluded from FPS)...",
                    flush=True,
                )
                export_model = YOLO(str(weights))
                exported = Path(
                    export_model.export(
                        format="engine",
                        imgsz=832,
                        half=True,
                        dynamic=True,
                        device=0,
                        workspace=4,
                        verbose=False,
                    )
                )
                engine_path.parent.mkdir(parents=True, exist_ok=True)
                if exported.resolve() != engine_path.resolve():
                    shutil.copy2(exported, engine_path)
            model = YOLO(str(engine_path), task="detect")
            print(
                f"[OK] TensorRT backend ready | elapsed={fmt_seconds(time.perf_counter()-started)} "
                f"| engine={engine_path}",
                flush=True,
            )
            return model, "tensorrt_fp16"
        except Exception as exc:
            if preferred == "tensorrt":
                raise
            print(
                f"[WARN] TensorRT unavailable: {type(exc).__name__}: {exc}",
                flush=True,
            )
            print("[BACKEND] Falling back to PyTorch FP16.", flush=True)

    started = time.perf_counter()
    model = new_model(weights)
    print(
        f"[OK] PyTorch FP16 backend ready | elapsed={fmt_seconds(time.perf_counter()-started)}",
        flush=True,
    )
    return model, "pytorch_fp16"


def detect_realtime(model, img, size, nms, conf, backend):
    """One fresh detector inference with semantically safe pre-NMS confidence pruning."""
    kwargs = dict(
        source=img,
        conf=float(conf),
        iou=float(nms),
        imgsz=int(size),
        classes=CLASSES,
        max_det=1000,
        device=0,
        verbose=False,
    )
    if backend == "pytorch_fp16":
        kwargs["half"] = True
    result = model.predict(**kwargs)[0]
    return boxes(result.boxes.data.cpu().numpy())


def benchmark_size_cap(model, first, nms, conf, backend, target_fps, min_size, budget_fraction):
    """Choose the highest size whose detector latency fits a conservative frame budget."""
    candidates = [832, 736, 640, 576, 512, 448, 416, 384, 352, 320]
    candidates = [s for s in candidates if s >= min_size]
    budget_ms = (1000.0 / target_fps) * budget_fraction
    print(
        f"[CALIBRATE] Detector budget={budget_ms:.2f} ms "
        f"({budget_fraction:.0%} of {1000.0/target_fps:.2f} ms frame budget).",
        flush=True,
    )
    rows = []
    started_all = time.perf_counter()
    for idx, size in enumerate(candidates, 1):
        for _ in range(2):
            detect_realtime(model, first, size, nms, conf, backend)
        samples = []
        for _ in range(5):
            sync()
            t0 = time.perf_counter()
            detect_realtime(model, first, size, nms, conf, backend)
            sync()
            samples.append(time.perf_counter() - t0)
        mean_ms = float(np.mean(samples) * 1000)
        rows.append({"size": size, "detector_ms": mean_ms})
        progress(
            "CALIBRATE",
            idx,
            len(candidates),
            started_all,
            f"imgsz={size} mean_detector={mean_ms:.2f}ms",
        )
    eligible = [r["size"] for r in rows if r["detector_ms"] <= budget_ms]
    selected = max(eligible) if eligible else min(candidates)
    print(f"[CALIBRATE] Initial realtime size cap={selected}.", flush=True)
    return selected, rows


def lower_cap(cap, min_size):
    sizes = [832, 736, 640, 576, 512, 448, 416, 384, 352, 320]
    sizes = [s for s in sizes if s >= min_size]
    if cap not in sizes:
        smaller = [s for s in sizes if s < cap]
        return max(smaller) if smaller else min(sizes)
    i = sizes.index(cap)
    return sizes[min(i + 1, len(sizes) - 1)]


def summarize_stage(samples):
    if not samples:
        return {"mean_ms": None, "p95_ms": None}
    a = np.asarray(samples, dtype=float) * 1000.0
    return {"mean_ms": float(np.mean(a)), "p95_ms": float(np.percentile(a, 95))}


def run_one(system, dataset, manifest, weights, engine_path, target_fps, gate_frames,
            progress_every, system_index, system_total, default_backend):
    import cv2
    import torch

    runtime = {k: system.get(k) for k in RUNTIME_KEYS if k in system}
    cfg_data = {k: v for k, v in system.items() if k not in RUNTIME_KEYS}
    c = Config(**cfg_data).validate()
    governor = bool(runtime.get("realtime_governor", c.policy == "adaptive"))
    preferred_backend = runtime.get("backend", default_backend)
    min_runtime_size = int(runtime.get("min_runtime_size", 320))
    budget_fraction = float(runtime.get("detector_budget_fraction", 0.70))
    if min_runtime_size not in [320, 352, 384, 416, 448, 512, 576, 640, 736, 832]:
        raise ValueError("min_runtime_size must be a supported multiple-of-32 v14 size")
    if not 0.40 <= budget_fraction <= 0.90:
        raise ValueError("detector_budget_fraction must be in [0.40, 0.90]")

    total_frames = sum(x["frames"] for x in manifest)
    print("\n" + "=" * 100, flush=True)
    print(f"[SYSTEM {system_index}/{system_total}] {c.name}", flush=True)
    print(
        f"[SYSTEM] target={target_fps:.2f} FPS | gate={gate_frames} frames | "
        f"governor={governor} | preferred_backend={preferred_backend}",
        flush=True,
    )

    model, backend = make_backend(weights, preferred_backend, engine_path)
    torch.cuda.reset_peak_memory_stats()

    measured_seconds = 0.0
    measured_frames = 0
    frame_totals = []
    stage = {k: [] for k in ["read", "analysis", "controller", "detector", "tracker"]}
    gate_checked = False
    failed_gate = False
    system_wall_start = time.perf_counter()
    cap = c.size if c.policy == "fixed" else 832
    calibration_rows = []

    with tempfile.TemporaryDirectory(prefix="acmot_v14_rt_") as tmp:
        local_root = Path(tmp)

        for seq_index, seq in enumerate(manifest, 1):
            sn = seq["sequence"]
            print(
                f"\n[SEQUENCE {seq_index}/{len(manifest)}] {sn} | frames={seq['frames']} | "
                "step=copy -> validate -> calibrate/warmup -> controller/tracker -> measure",
                flush=True,
            )
            paths = copy_sequence_to_local(dataset, seq, local_root)
            first = cv2.imread(str(paths[0]))
            if first is None:
                raise ValueError(f"Unreadable local frame: {paths[0]}")

            control = Controller(c)
            tracker = make_tracker(c)
            previous = []

            detector_conf_floor = c.low if c.recovery else 0.19

            if seq_index == 1 and governor:
                cap, calibration_rows = benchmark_size_cap(
                    model,
                    first,
                    c.nms,
                    detector_conf_floor,
                    backend,
                    target_fps,
                    min_runtime_size,
                    budget_fraction,
                )
            else:
                warm_sizes = [c.size] if c.policy == "fixed" else sorted({640, 736, min(832, cap)})
                warm_started = time.perf_counter()
                print(f"[WARMUP] sizes={warm_sizes}; excluded from FPS.", flush=True)
                total_warm = len(warm_sizes) * 3
                warm_done = 0
                for size in warm_sizes:
                    for _ in range(3):
                        detect_realtime(model, first, size, c.nms, detector_conf_floor, backend)
                        warm_done += 1
                        progress("WARMUP", warm_done, total_warm, warm_started, f"imgsz={size}")
                sync()

            print("[MEASURE] Fresh controller + ByteTrack created. Starting measured frames.", flush=True)
            sequence_started = time.perf_counter()

            for frame, path in enumerate(paths, 1):
                frame_start = time.perf_counter()

                t0 = time.perf_counter()
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f"Unreadable frame: {path}")
                read_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                v = visual(img) if frame == 1 or frame % 10 == 1 else {}
                analysis_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                params = control.choose(frame, v, previous)
                requested_size = int(params["size"])
                if governor:
                    params["size"] = min(requested_size, cap)
                controller_s = time.perf_counter() - t0

                sync()
                t0 = time.perf_counter()
                dets = detect_realtime(
                    model,
                    img,
                    params["size"],
                    params["nms"],
                    params["conf"],
                    backend,
                )
                sync()
                detector_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                tracks, kept = track(tracker, dets, img.shape[:2], params)
                previous = kept if c.detector_feedback else tracks[:, [0, 1, 2, 3, 5, 6]]
                tracker_s = time.perf_counter() - t0

                elapsed = time.perf_counter() - frame_start
                measured_seconds += elapsed
                measured_frames += 1
                frame_totals.append(elapsed)
                stage["read"].append(read_s)
                stage["analysis"].append(analysis_s)
                stage["controller"].append(controller_s)
                stage["detector"].append(detector_s)
                stage["tracker"].append(tracker_s)

                if governor and measured_frames % 25 == 0 and measured_frames < gate_frames:
                    recent = frame_totals[-50:]
                    recent_fps = len(recent) / sum(recent)
                    if recent_fps < target_fps * 1.08 and cap > min_runtime_size:
                        old = cap
                        cap = lower_cap(cap, min_runtime_size)
                        print(
                            f"[GOVERNOR] recent_FPS={recent_fps:.2f} < safety target "
                            f"{target_fps*1.08:.2f}; size cap {old} -> {cap}.",
                            flush=True,
                        )

                if measured_frames == 1 or measured_frames % progress_every == 0:
                    fps = measured_frames / measured_seconds
                    recent = frame_totals[-min(50, len(frame_totals)):]
                    recent_fps = len(recent) / sum(recent)
                    elapsed_wall = time.perf_counter() - system_wall_start
                    remaining = max(total_frames - measured_frames, 0)
                    eta = elapsed_wall / measured_frames * remaining if measured_frames else 0.0
                    print(
                        f"{bar(measured_frames, total_frames)} "
                        f"{100.0*measured_frames/total_frames:6.2f}% | "
                        f"system={c.name} | seq={sn} | frame={frame}/{seq['frames']} | "
                        f"requested={requested_size} used={params['size']} cap={cap} | "
                        f"scene={params['scene']} SCI={params['sci']:.3f} | "
                        f"FPS={fps:.2f} recent50={recent_fps:.2f} target={target_fps:.2f} | "
                        f"elapsed={fmt_seconds(elapsed_wall)} ETA={fmt_seconds(eta)}",
                        flush=True,
                    )
                    print(
                        "  [PROFILE] "
                        f"read={read_s*1000:.1f}ms analysis={analysis_s*1000:.1f}ms "
                        f"controller={controller_s*1000:.1f}ms detector={detector_s*1000:.1f}ms "
                        f"tracker={tracker_s*1000:.1f}ms total={elapsed*1000:.1f}ms",
                        flush=True,
                    )

                if not gate_checked and measured_frames >= gate_frames:
                    gate_checked = True
                    fps = measured_frames / measured_seconds
                    verdict = "PASS" if fps >= target_fps else "FAIL"
                    print(
                        f"[GATE] {verdict} | system={c.name} | frames={measured_frames} | "
                        f"average_FPS={fps:.2f} | target={target_fps:.2f} | cap={cap} | backend={backend}",
                        flush=True,
                    )
                    if fps < target_fps:
                        failed_gate = True
                        break
                    print("[GATE] Realtime passed; continuing full dataset for stable FPS.", flush=True)

            print(
                f"[SEQUENCE DONE] {sn} | wall={fmt_seconds(time.perf_counter()-sequence_started)}",
                flush=True,
            )
            shutil.rmtree(local_root / sn, ignore_errors=True)
            if failed_gate:
                break

    fps = measured_frames / measured_seconds if measured_seconds else 0.0
    profile = {name: summarize_stage(values) for name, values in stage.items()}
    profile["total"] = summarize_stage(frame_totals)
    result = {
        "version": "v14",
        "system": c.name,
        "configuration": asdict(c),
        "runtime": {
            "backend": backend,
            "realtime_governor": governor,
            "final_size_cap": cap,
            "min_runtime_size": min_runtime_size,
            "detector_budget_fraction": budget_fraction,
            "semantic_detector_conf_pruning": True,
            "one_fresh_inference_per_frame": True,
        },
        "status": "FAIL_REALTIME" if failed_gate else (
            "PASS_REALTIME" if fps >= target_fps else "FAIL_REALTIME"
        ),
        "target_fps": target_fps,
        "gate_frames": gate_frames,
        "measured_frames": measured_frames,
        "seconds": measured_seconds,
        "fps": fps,
        "p95_ms": float(np.percentile(frame_totals, 95) * 1000) if frame_totals else None,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "stage_profile": profile,
        "calibration": calibration_rows,
        "precision": "TensorRT FP16 preferred; PyTorch FP16 fallback",
        "frame_source": "Colab local SSD; copied from Drive before timing",
        "timing_includes": "local frame read + SceneAnalyzer + Controller + one detector inference + ByteTrack",
        "timing_excludes": "Drive copy + backend export/build + calibration/warmup + output serialization",
    }
    print(
        f"[RESULT] {c.name} | {result['status']} | FPS={fps:.2f} | "
        f"p95={result['p95_ms']:.2f}ms | backend={backend} | final_cap={cap}",
        flush=True,
    )
    print("[RESULT] Mean stage latency:", flush=True)
    for name, values in profile.items():
        print(f"  - {name}: mean={values['mean_ms']:.2f}ms p95={values['p95_ms']:.2f}ms", flush=True)

    del model
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--sequences", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--systems", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--target-fps", required=True, type=float)
    p.add_argument("--gate-frames", required=True, type=int)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--backend", default="auto", choices=["auto", "tensorrt", "pytorch"])
    p.add_argument("--engine", type=Path, default=Path("/content/weights/yolov8n_v14_dynamic_fp16.engine"))
    a = p.parse_args()

    if a.target_fps <= 0 or a.gate_frames < 1 or a.progress_every < 1:
        raise ValueError("Invalid speed-test thresholds")

    boot_started = time.perf_counter()
    print("[STEP 1/5] Environment preflight: CUDA/T4/package pins.", flush=True)
    environment()
    print(f"[OK] Step 1 complete | elapsed={fmt_seconds(time.perf_counter()-boot_started)}", flush=True)

    print("[STEP 2/5] Reading sequence list and system configurations.", flush=True)
    names = json.loads(a.sequences.read_text())
    systems = json.loads(a.systems.read_text())
    print(f"[OK] sequences={len(names)} systems={len(systems)}", flush=True)

    print("[STEP 3/5] Building/verifying dataset manifest. This may take time on Drive.", flush=True)
    manifest_started = time.perf_counter()
    manifest = dataset_manifest(a.dataset, names)
    total_frames = sum(x["frames"] for x in manifest)
    print(
        f"[OK] manifest={len(manifest)} sequences/{total_frames} frames | "
        f"elapsed={fmt_seconds(time.perf_counter()-manifest_started)}",
        flush=True,
    )

    print("[STEP 4/5] Creating output metadata.", flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    atomic_json(a.output / "configuration.json", {
        "version": "v14",
        "purpose": "realtime deployment throughput + stage profiler",
        "target_fps": a.target_fps,
        "gate_frames": a.gate_frames,
        "progress_every": a.progress_every,
        "systems": systems,
        "backend_preference": a.backend,
        "engine_path": str(a.engine),
        "inference_rule": "exactly one fresh YOLOv8n inference per measured frame",
        "semantic_pruning": "YOLO conf equals Controller final keep threshold for that frame",
        "timing_includes": "local frame read + analysis + controller + detector + ByteTrack",
        "timing_excludes": "Drive copy + engine build/export + calibration/warmup + serialization",
    })
    atomic_json(a.output / "dataset_manifest.json", manifest)
    print("[OK] Metadata saved.", flush=True)

    print("[STEP 5/5] Running systems. Watch progress, ETA, FPS and per-stage latency below.", flush=True)
    results = []
    for idx, system in enumerate(systems, 1):
        result = run_one(
            system,
            a.dataset,
            manifest,
            a.weights,
            a.engine,
            a.target_fps,
            a.gate_frames,
            a.progress_every,
            idx,
            len(systems),
            a.backend,
        )
        results.append(result)
        atomic_json(a.output / "speedtest_results.json", results)
        print(f"[SAVE] {idx}/{len(systems)} system results saved.", flush=True)

    print("\n=== AC-MOT v14 REALTIME SUMMARY ===", flush=True)
    for r in results:
        print(
            f"{r['system']}: {r['fps']:.2f} FPS -> {r['status']} | "
            f"backend={r['runtime']['backend']} | cap={r['runtime']['final_size_cap']}",
            flush=True,
        )
    print(f"[DONE] Results: {a.output/'speedtest_results.json'}", flush=True)


if __name__ == "__main__":
    main()
