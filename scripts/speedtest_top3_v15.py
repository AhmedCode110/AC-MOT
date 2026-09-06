"""AC-MOT v15 portable live-frame realtime benchmark.

v15 preserves the v14 AC-MOT controller/tracker semantics but separates two
different throughput questions:

1) processing FPS: SceneAnalyzer + Controller + one fresh detector inference +
   ByteTrack, starting from an already decoded image buffer (live-frame interface).
2) dataset playback FPS: the same processing plus local JPEG file read/decode.

JPEG decode is measured and reported, never hidden. The 25 FPS realtime gate is
applied only to the documented decoded/live-frame processing path.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import CLASSES, Config, Controller, atomic_json, boxes
from experiment import dataset_manifest, environment, make_tracker, new_model, sync, track, visual

RUNTIME_KEYS = {
    "realtime_governor", "backend", "min_runtime_size", "detector_budget_fraction"
}
SIZES = [832, 736, 640, 576, 512, 448, 416, 384, 352, 320]


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
    eta = elapsed / done * (total - done) if done > 0 and total > done else 0.0
    eta_text = fmt_seconds(eta) if done else "estimating..."
    pct = 100.0 * done / total if total else 0.0
    print(
        f"{bar(done,total)} {pct:6.2f}% | {stage} | elapsed={fmt_seconds(elapsed)} "
        f"| ETA={eta_text}" + (f" | {detail}" if detail else ""),
        flush=True,
    )


def summarize(samples):
    if not samples:
        return {"mean_ms": None, "p95_ms": None}
    a = np.asarray(samples, dtype=float) * 1000.0
    return {"mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95))}


def copy_sequence_to_local(dataset, seq, local_root):
    sn = seq["sequence"]
    src = dataset / "sequences" / sn
    dst = local_root / sn
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    filenames = list(seq["frame_sha256"])
    started = time.perf_counter()
    print(f"[COPY] {sn}: accessible Drive -> local Colab SSD; excluded from both FPS metrics.", flush=True)
    for i, filename in enumerate(filenames, 1):
        shutil.copy2(src / filename, dst / filename)
        if i == 1 or i % 25 == 0 or i == len(filenames):
            progress("COPY", i, len(filenames), started, f"sequence={sn} file={filename}")
    elapsed = time.perf_counter() - started
    return [dst / f for f in filenames], elapsed


def make_backend(weights, preferred, engine_path):
    from ultralytics import YOLO

    preferred = str(preferred).lower()
    if preferred not in {"auto", "tensorrt", "pytorch"}:
        raise ValueError(f"Unknown backend: {preferred}")

    reason = None
    if preferred in {"auto", "tensorrt"}:
        try:
            started = time.perf_counter()
            print("[BACKEND] Trying TensorRT FP16 (build/load excluded from FPS).", flush=True)
            if not engine_path.exists():
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
                f"[OK] TensorRT FP16 ready | elapsed={fmt_seconds(time.perf_counter()-started)}",
                flush=True,
            )
            return model, "tensorrt_fp16", None
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if preferred == "tensorrt":
                raise
            print(f"[WARN] TensorRT unavailable -> {reason}", flush=True)
            print("[BACKEND] Explicit fallback: PyTorch FP16.", flush=True)

    started = time.perf_counter()
    model = new_model(weights)
    print(f"[OK] PyTorch FP16 ready | elapsed={fmt_seconds(time.perf_counter()-started)}", flush=True)
    return model, "pytorch_fp16", reason


def detect_realtime(model, img, size, nms, conf, backend):
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
    candidates = [s for s in SIZES if s >= min_size]
    budget_ms = (1000.0 / target_fps) * budget_fraction
    print(
        f"[CALIBRATE] Processing detector budget={budget_ms:.2f}ms "
        f"({budget_fraction:.0%} of 40ms target frame budget).",
        flush=True,
    )
    rows = []
    started = time.perf_counter()
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
        mean_ms = float(np.mean(samples) * 1000.0)
        rows.append({"size": size, "detector_ms": mean_ms})
        progress("CALIBRATE", idx, len(candidates), started, f"imgsz={size} detector={mean_ms:.2f}ms")
    eligible = [row["size"] for row in rows if row["detector_ms"] <= budget_ms]
    selected = max(eligible) if eligible else min(candidates)
    print(f"[CALIBRATE] Initial processing size cap={selected}.", flush=True)
    return selected, rows


def lower_cap(cap, min_size):
    sizes = [s for s in SIZES if s >= min_size]
    smaller = [s for s in sizes if s < cap]
    return max(smaller) if smaller else min(sizes)


def decode_chunk(paths):
    import cv2

    images = []
    samples = []
    for path in paths:
        t0 = time.perf_counter()
        image = cv2.imread(str(path))
        elapsed = time.perf_counter() - t0
        if image is None:
            raise ValueError(f"Unreadable JPEG: {path}")
        images.append(image)
        samples.append(elapsed)
    return images, samples


def run_one(system, dataset, manifest, weights, engine_path, target_fps, gate_frames,
            progress_every, system_index, system_total, default_backend, decode_chunk_size):
    import cv2
    import torch

    runtime = {k: system.get(k) for k in RUNTIME_KEYS if k in system}
    cfg_data = {k: v for k, v in system.items() if k not in RUNTIME_KEYS}
    c = Config(**cfg_data).validate()
    governor = bool(runtime.get("realtime_governor", c.policy == "adaptive"))
    preferred_backend = runtime.get("backend", default_backend)
    min_runtime_size = int(runtime.get("min_runtime_size", 320))
    budget_fraction = float(runtime.get("detector_budget_fraction", 0.70))
    if min_runtime_size not in SIZES:
        raise ValueError("Unsupported min_runtime_size")
    if not 0.40 <= budget_fraction <= 0.90:
        raise ValueError("detector_budget_fraction must be in [0.40,0.90]")

    total_frames = sum(x["frames"] for x in manifest)
    print("\n" + "=" * 104, flush=True)
    print(f"[SYSTEM {system_index}/{system_total}] {c.name}", flush=True)
    print(
        f"[DEFINITION] realtime gate={target_fps:.2f} FPS on DECODED/LIVE FRAME processing. "
        "JPEG decode is measured separately.",
        flush=True,
    )

    model, backend, backend_fallback_reason = make_backend(weights, preferred_backend, engine_path)
    torch.cuda.reset_peak_memory_stats()

    processing_seconds = 0.0
    decode_seconds = 0.0
    drive_copy_seconds = 0.0
    measured_frames = 0
    processing_totals = []
    decode_samples = []
    serialized_totals = []
    stage = {k: [] for k in ["analysis", "controller", "detector", "tracker"]}
    gate_checked = False
    failed_processing_gate = False
    cap = c.size if c.policy == "fixed" else 832
    calibration_rows = []
    system_wall_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="acmot_v15_") as tmp:
        local_root = Path(tmp)
        for seq_index, seq in enumerate(manifest, 1):
            sn = seq["sequence"]
            print(
                f"\n[SEQUENCE {seq_index}/{len(manifest)}] {sn} | frames={seq['frames']} | "
                f"chunk={decode_chunk_size}",
                flush=True,
            )
            paths, copy_s = copy_sequence_to_local(dataset, seq, local_root)
            drive_copy_seconds += copy_s

            first = cv2.imread(str(paths[0]))
            if first is None:
                raise ValueError(f"Unreadable first frame: {paths[0]}")
            control = Controller(c)
            tracker = make_tracker(c)
            previous = []
            detector_conf_floor = c.low if c.recovery else 0.19

            if seq_index == 1 and governor:
                cap, calibration_rows = benchmark_size_cap(
                    model, first, c.nms, detector_conf_floor, backend,
                    target_fps, min_runtime_size, budget_fraction,
                )
            else:
                warm_sizes = [c.size] if c.policy == "fixed" else sorted({640, 736, min(832, cap)})
                print(f"[WARMUP] sizes={warm_sizes}; excluded from FPS.", flush=True)
                for size in warm_sizes:
                    for _ in range(3):
                        detect_realtime(model, first, size, c.nms, detector_conf_floor, backend)
                sync()

            for chunk_start in range(0, len(paths), decode_chunk_size):
                chunk_paths = paths[chunk_start:chunk_start + decode_chunk_size]
                images, chunk_decode = decode_chunk(chunk_paths)

                for offset, (img, decode_s) in enumerate(zip(images, chunk_decode), 0):
                    decode_samples.append(decode_s)
                    decode_seconds += decode_s
                    frame = chunk_start + offset + 1
                    process_start = time.perf_counter()

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
                        model, img, params["size"], params["nms"], params["conf"], backend
                    )
                    sync()
                    detector_s = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    tracks, kept = track(tracker, dets, img.shape[:2], params)
                    previous = kept if c.detector_feedback else tracks[:, [0, 1, 2, 3, 5, 6]]
                    tracker_s = time.perf_counter() - t0

                    processing_elapsed = time.perf_counter() - process_start
                    processing_seconds += processing_elapsed
                    measured_frames += 1
                    processing_totals.append(processing_elapsed)
                    serialized_totals.append(processing_elapsed + decode_s)
                    stage["analysis"].append(analysis_s)
                    stage["controller"].append(controller_s)
                    stage["detector"].append(detector_s)
                    stage["tracker"].append(tracker_s)

                    if governor and measured_frames % 25 == 0 and measured_frames < gate_frames:
                        recent = processing_totals[-50:]
                        recent_processing_fps = len(recent) / sum(recent)
                        if recent_processing_fps < target_fps * 1.05 and cap > min_runtime_size:
                            old = cap
                            cap = lower_cap(cap, min_runtime_size)
                            print(
                                f"[GOVERNOR] processing_recent50={recent_processing_fps:.2f} | "
                                f"cap {old}->{cap}",
                                flush=True,
                            )

                    if measured_frames == 1 or measured_frames % progress_every == 0:
                        processing_fps = measured_frames / processing_seconds
                        playback_fps = measured_frames / (processing_seconds + decode_seconds)
                        recent = processing_totals[-min(50, len(processing_totals)):]
                        recent_fps = len(recent) / sum(recent)
                        elapsed_wall = time.perf_counter() - system_wall_start
                        remaining = max(total_frames - measured_frames, 0)
                        eta = elapsed_wall / measured_frames * remaining if measured_frames else 0.0
                        print(
                            f"{bar(measured_frames,total_frames)} "
                            f"{100.0*measured_frames/total_frames:6.2f}% | system={c.name} | "
                            f"seq={sn} frame={frame}/{seq['frames']} | requested={requested_size} "
                            f"used={params['size']} cap={cap} | scene={params['scene']} "
                            f"SCI={params['sci']:.3f}",
                            flush=True,
                        )
                        print(
                            f"  [FPS] processing={processing_fps:.2f} | recent50={recent_fps:.2f} | "
                            f"dataset_playback_serial={playback_fps:.2f} | target={target_fps:.2f} | "
                            f"elapsed={fmt_seconds(elapsed_wall)} ETA={fmt_seconds(eta)}",
                            flush=True,
                        )
                        print(
                            f"  [PROFILE] jpeg_decode={decode_s*1000:.1f}ms | "
                            f"analysis={analysis_s*1000:.1f}ms controller={controller_s*1000:.1f}ms "
                            f"detector={detector_s*1000:.1f}ms tracker={tracker_s*1000:.1f}ms "
                            f"processing_total={processing_elapsed*1000:.1f}ms",
                            flush=True,
                        )

                    if not gate_checked and measured_frames >= gate_frames:
                        gate_checked = True
                        processing_fps = measured_frames / processing_seconds
                        playback_fps = measured_frames / (processing_seconds + decode_seconds)
                        verdict = "PASS" if processing_fps >= target_fps else "FAIL"
                        print(
                            f"[PROCESSING GATE] {verdict} | processing_FPS={processing_fps:.2f} | "
                            f"dataset_playback_serial_FPS={playback_fps:.2f} | target={target_fps:.2f} | "
                            f"backend={backend} cap={cap}",
                            flush=True,
                        )
                        if processing_fps < target_fps:
                            failed_processing_gate = True
                            break
                        print(
                            "[PROCESSING GATE] Passed. Continuing full dataset for stable processing FPS.",
                            flush=True,
                        )

                del images
                if failed_processing_gate:
                    break

            shutil.rmtree(local_root / sn, ignore_errors=True)
            if failed_processing_gate:
                break

    processing_fps = measured_frames / processing_seconds if processing_seconds else 0.0
    playback_seconds = processing_seconds + decode_seconds
    playback_fps = measured_frames / playback_seconds if playback_seconds else 0.0
    processing_profile = {name: summarize(values) for name, values in stage.items()}
    processing_profile["total"] = summarize(processing_totals)

    result = {
        "version": "v15",
        "system": c.name,
        "configuration": asdict(c),
        "runtime": {
            "backend": backend,
            "backend_fallback_reason": backend_fallback_reason,
            "realtime_governor": governor,
            "final_size_cap": cap,
            "min_runtime_size": min_runtime_size,
            "detector_budget_fraction": budget_fraction,
            "decode_chunk_size": decode_chunk_size,
            "semantic_detector_conf_pruning": True,
            "one_fresh_inference_per_frame": True,
        },
        "status": (
            "FAIL_REALTIME_PROCESSING" if failed_processing_gate
            else ("PASS_REALTIME_PROCESSING" if processing_fps >= target_fps else "FAIL_REALTIME_PROCESSING")
        ),
        "dataset_playback_status": (
            "PASS_25FPS_SERIAL_PLAYBACK" if playback_fps >= target_fps else "BELOW_25FPS_SERIAL_PLAYBACK"
        ),
        "target_fps": target_fps,
        "gate_frames": gate_frames,
        "measured_frames": measured_frames,
        "processing_seconds": processing_seconds,
        "processing_fps": processing_fps,
        "processing_p95_ms": float(np.percentile(processing_totals, 95) * 1000) if processing_totals else None,
        "jpeg_decode_seconds": decode_seconds,
        "jpeg_decode_profile": summarize(decode_samples),
        "serialized_dataset_playback_seconds": playback_seconds,
        "serialized_dataset_playback_fps": playback_fps,
        "serialized_dataset_playback_p95_ms": float(np.percentile(serialized_totals,95)*1000) if serialized_totals else None,
        "drive_copy_seconds_excluded": drive_copy_seconds,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "processing_stage_profile": processing_profile,
        "calibration": calibration_rows,
        "realtime_definition": (
            "Decoded/live-frame processing: SceneAnalyzer + Controller + exactly one fresh "
            "YOLO inference + ByteTrack. JPEG dataset read/decode is separately measured."
        ),
        "processing_timing_includes": "SceneAnalyzer + Controller + one fresh detector inference + ByteTrack",
        "processing_timing_excludes": "JPEG read/decode + Drive copy + backend build + warmup/calibration + serialization",
    }

    print(
        f"[RESULT] {c.name} | {result['status']} | processing={processing_fps:.2f} FPS | "
        f"dataset_playback_serial={playback_fps:.2f} FPS | backend={backend} cap={cap}",
        flush=True,
    )
    print("[RESULT] Mean processing stage latency:", flush=True)
    for name, values in processing_profile.items():
        print(f"  - {name}: mean={values['mean_ms']:.2f}ms p95={values['p95_ms']:.2f}ms", flush=True)
    print(
        f"  - jpeg_decode: mean={result['jpeg_decode_profile']['mean_ms']:.2f}ms "
        f"p95={result['jpeg_decode_profile']['p95_ms']:.2f}ms (reported separately)",
        flush=True,
    )

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
    p.add_argument("--engine", type=Path, default=Path("/content/weights/yolov8n_v15_dynamic_fp16.engine"))
    p.add_argument("--decode-chunk-size", type=int, default=16)
    a = p.parse_args()

    if a.target_fps <= 0 or a.gate_frames < 1 or a.progress_every < 1:
        raise ValueError("Invalid speed-test thresholds")
    if not 1 <= a.decode_chunk_size <= 64:
        raise ValueError("decode_chunk_size must be in [1,64]")

    print("[V15 1/5] Environment preflight: CUDA/T4/package pins.", flush=True)
    environment()

    print("[V15 2/5] Reading systems and sequences.", flush=True)
    names = json.loads(a.sequences.read_text())
    systems = json.loads(a.systems.read_text())

    print("[V15 3/5] Building/verifying dataset manifest.", flush=True)
    started = time.perf_counter()
    manifest = dataset_manifest(a.dataset, names)
    total_frames = sum(x["frames"] for x in manifest)
    print(
        f"[OK] {len(manifest)} sequences / {total_frames} frames | "
        f"elapsed={fmt_seconds(time.perf_counter()-started)}",
        flush=True,
    )

    print("[V15 4/5] Saving transparent timing protocol.", flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    atomic_json(a.output / "configuration.json", {
        "version": "v15",
        "purpose": "portable live-frame processing realtime gate + separate dataset playback measurement",
        "target_fps": a.target_fps,
        "gate_frames": a.gate_frames,
        "systems": systems,
        "backend_preference": a.backend,
        "engine_path": str(a.engine),
        "decode_chunk_size": a.decode_chunk_size,
        "realtime_gate_definition": "decoded/live-frame processing only",
        "jpeg_policy": "JPEG read/decode excluded from processing gate but measured and reported separately",
        "inference_rule": "exactly one fresh YOLOv8n inference per measured frame",
    })
    atomic_json(a.output / "dataset_manifest.json", manifest)

    print("[V15 5/5] Running systems.", flush=True)
    results = []
    for idx, system in enumerate(systems, 1):
        result = run_one(
            system, a.dataset, manifest, a.weights, a.engine,
            a.target_fps, a.gate_frames, a.progress_every,
            idx, len(systems), a.backend, a.decode_chunk_size,
        )
        results.append(result)
        atomic_json(a.output / "speedtest_results.json", results)
        print(f"[SAVE] {idx}/{len(systems)} results saved.", flush=True)

    print("\n=== AC-MOT v15 SUMMARY ===", flush=True)
    for r in results:
        print(
            f"{r['system']}: processing={r['processing_fps']:.2f} FPS -> {r['status']} | "
            f"serialized_dataset_playback={r['serialized_dataset_playback_fps']:.2f} FPS | "
            f"backend={r['runtime']['backend']} cap={r['runtime']['final_size_cap']}",
            flush=True,
        )
    print(f"[DONE] Results: {a.output/'speedtest_results.json'}", flush=True)


if __name__ == "__main__":
    main()
