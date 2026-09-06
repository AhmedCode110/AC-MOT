"""AC-MOT v16 paper evaluation runner.

Evaluation-only version. It preserves v15 decoded/live-frame processing semantics,
generates complete tracking recordings for every selected system on all configured
sequences, runs the pinned TrackEval adapter, and merges quality + processing-speed
results into paper-ready CSV/Markdown tables.

No candidate is selected or tuned from v16 results.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import Config, Controller, atomic_json
from experiment import dataset_manifest, environment, make_tracker, recording, sync, track, visual
from scripts.speedtest_top3_v15 import (
    RUNTIME_KEYS,
    SIZES,
    bar,
    benchmark_size_cap,
    copy_sequence_to_local,
    decode_chunk,
    detect_realtime,
    fmt_seconds,
    lower_cap,
    make_backend,
    summarize,
)

META_KEYS = set(RUNTIME_KEYS) | {"role"}
GT_FILTER = dict(categories=[1, 4, 5, 6, 9], score=1, occlusion_lt=2, truncation_lt=2)


def config_from_system(system):
    return Config(**{k: v for k, v in system.items() if k not in META_KEYS}).validate()


def run_system(system, dataset, manifest, weights, engine, target_fps, progress_every,
               default_backend, decode_chunk_size, output, index, total_systems):
    import cv2
    import torch

    c = config_from_system(system)
    runtime = {k: system.get(k) for k in RUNTIME_KEYS if k in system}
    role = system.get("role", "unspecified")
    governor = bool(runtime.get("realtime_governor", c.policy == "adaptive"))
    preferred_backend = runtime.get("backend", default_backend)
    min_runtime_size = int(runtime.get("min_runtime_size", 320))
    budget_fraction = float(runtime.get("detector_budget_fraction", 0.70))
    if min_runtime_size not in SIZES:
        raise ValueError(f"Unsupported min_runtime_size for {c.name}: {min_runtime_size}")

    system_dir = output / c.name
    system_dir.mkdir(parents=True, exist_ok=False)
    total_frames = sum(x["frames"] for x in manifest)

    print("\n" + "=" * 108, flush=True)
    print(f"[V16 SYSTEM {index}/{total_systems}] {c.name} | role={role}", flush=True)
    print(
        f"[PROTOCOL] full 17-sequence accuracy recording + decoded/live processing timing | "
        f"target={target_fps:.2f} FPS | no early abort",
        flush=True,
    )

    model, backend, fallback_reason = make_backend(weights, preferred_backend, engine)
    torch.cuda.reset_peak_memory_stats()

    processing_seconds = 0.0
    decode_seconds = 0.0
    drive_copy_seconds = 0.0
    processing_totals = []
    decode_samples = []
    stage = {k: [] for k in ["analysis", "controller", "detector", "tracker"]}
    used_sizes = []
    calibration = []
    measured_frames = 0
    cap = c.size if c.policy == "fixed" else 832
    wall_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="acmot_v16_") as tmp:
        local_root = Path(tmp)
        for seq_index, seq in enumerate(manifest, 1):
            sn = seq["sequence"]
            print(f"\n[SEQ {seq_index}/{len(manifest)}] {sn} | frames={seq['frames']}", flush=True)
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
                cap, calibration = benchmark_size_cap(
                    model, first, c.nms, detector_conf_floor, backend,
                    target_fps, min_runtime_size, budget_fraction,
                )
            else:
                warm_sizes = [c.size] if c.policy == "fixed" else sorted({640, 736, min(832, cap)})
                print(f"[WARMUP] sizes={warm_sizes}; excluded from processing FPS.", flush=True)
                for size in warm_sizes:
                    for _ in range(3):
                        detect_realtime(model, first, size, c.nms, detector_conf_floor, backend)
                sync()

            target = system_dir / f"{sn}.frames.jsonl.gz"
            sequence_records = 0
            with gzip.open(target, "wt") as dest:
                for chunk_start in range(0, len(paths), decode_chunk_size):
                    chunk_paths = paths[chunk_start:chunk_start + decode_chunk_size]
                    images, chunk_decode = decode_chunk(chunk_paths)
                    for offset, (img, decode_s) in enumerate(zip(images, chunk_decode)):
                        frame = chunk_start + offset + 1
                        decode_seconds += decode_s
                        decode_samples.append(decode_s)

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
                        processing_totals.append(processing_elapsed)
                        stage["analysis"].append(analysis_s)
                        stage["controller"].append(controller_s)
                        stage["detector"].append(detector_s)
                        stage["tracker"].append(tracker_s)
                        used_sizes.append(int(params["size"]))
                        measured_frames += 1
                        sequence_records += 1

                        dest.write(json.dumps(recording(frame, tracks, params, processing_elapsed)) + "\n")

                        if governor and measured_frames % 25 == 0:
                            recent = processing_totals[-50:]
                            recent_fps = len(recent) / sum(recent)
                            if recent_fps < target_fps * 1.05 and cap > min_runtime_size:
                                old = cap
                                cap = lower_cap(cap, min_runtime_size)
                                print(f"[GOVERNOR] recent50={recent_fps:.2f} FPS | cap {old}->{cap}", flush=True)

                        if measured_frames == 1 or measured_frames % progress_every == 0:
                            fps = measured_frames / processing_seconds
                            wall = time.perf_counter() - wall_start
                            remaining = total_frames - measured_frames
                            eta = wall / measured_frames * remaining if measured_frames else 0.0
                            print(
                                f"{bar(measured_frames,total_frames)} "
                                f"{100*measured_frames/total_frames:6.2f}% | {c.name} | "
                                f"seq={sn} frame={frame}/{seq['frames']} | requested={requested_size} "
                                f"used={params['size']} cap={cap} | processing_FPS={fps:.2f} | "
                                f"elapsed={fmt_seconds(wall)} ETA={fmt_seconds(eta)}",
                                flush=True,
                            )
                    del images

            if sequence_records != seq["frames"]:
                raise RuntimeError(f"Recording frame count mismatch for {c.name}/{sn}")
            shutil.rmtree(local_root / sn, ignore_errors=True)

    if measured_frames != total_frames:
        raise RuntimeError(f"Expected {total_frames} frames, recorded {measured_frames} for {c.name}")

    processing_fps = measured_frames / processing_seconds
    playback_seconds = processing_seconds + decode_seconds
    playback_fps = measured_frames / playback_seconds
    profile = {name: summarize(values) for name, values in stage.items()}
    profile["total"] = summarize(processing_totals)
    switches = int(np.count_nonzero(np.diff(used_sizes))) if len(used_sizes) > 1 else 0

    result = {
        "system": c.name,
        "role": role,
        "configuration": asdict(c),
        "backend": backend,
        "backend_fallback_reason": fallback_reason,
        "realtime_governor": governor,
        "final_size_cap": cap,
        "frames": measured_frames,
        "processing_seconds": processing_seconds,
        "processing_fps": processing_fps,
        "processing_realtime_pass": processing_fps >= target_fps,
        "processing_p95_ms": float(np.percentile(processing_totals, 95) * 1000),
        "jpeg_decode_seconds": decode_seconds,
        "jpeg_decode_profile": summarize(decode_samples),
        "serialized_dataset_playback_fps": playback_fps,
        "drive_copy_seconds_excluded": drive_copy_seconds,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "stage_profile": profile,
        "mean_runtime_size": float(np.mean(used_sizes)),
        "resolution_switches": switches,
        "calibration": calibration,
    }
    atomic_json(system_dir / "system_summary.json", result)
    print(
        f"[SYSTEM DONE] {c.name} | processing={processing_fps:.2f} FPS | "
        f"RT={'PASS' if result['processing_realtime_pass'] else 'FAIL'} | "
        f"mean_size={result['mean_runtime_size']:.1f}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return result


def merge_paper_tables(output, systems, timing):
    metric_csv = output / "trackeval" / "summary.csv"
    rows = list(csv.DictReader(metric_csv.open()))
    metrics = {row["system"]: row for row in rows}
    timing_by_name = {row["system"]: row for row in timing}
    roles = {row["name"]: row.get("role", "unspecified") for row in systems}
    columns = [
        "system", "role", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDS", "FN", "FP",
        "processing_fps", "processing_p95_ms", "realtime_25fps", "mean_runtime_size",
        "resolution_switches", "dataset_playback_fps",
    ]
    merged = []
    for system in [s["name"] for s in systems]:
        m = metrics[system]
        t = timing_by_name[system]
        merged.append({
            "system": system,
            "role": roles[system],
            "HOTA": m["HOTA"], "DetA": m["DetA"], "AssA": m["AssA"],
            "MOTA": m["MOTA"], "IDF1": m["IDF1"], "IDS": m["IDS"],
            "FN": m["FN"], "FP": m["FP"],
            "processing_fps": f"{t['processing_fps']:.6f}",
            "processing_p95_ms": f"{t['processing_p95_ms']:.6f}",
            "realtime_25fps": "YES" if t["processing_realtime_pass"] else "NO",
            "mean_runtime_size": f"{t['mean_runtime_size']:.3f}",
            "resolution_switches": str(t["resolution_switches"]),
            "dataset_playback_fps": f"{t['serialized_dataset_playback_fps']:.6f}",
        })

    with (output / "paper_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(merged)

    def f4(v):
        try: return f"{float(v):.4f}"
        except Exception: return str(v)

    lines = [
        "# AC-MOT v16 Paper Comparison",
        "",
        "| Method | Role | HOTA | DetA | AssA | MOTA | IDF1 | IDS | FPS | RT >=25 | Mean size |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for r in merged:
        lines.append(
            f"| {r['system']} | {r['role']} | {f4(r['HOTA'])} | {f4(r['DetA'])} | "
            f"{f4(r['AssA'])} | {f4(r['MOTA'])} | {f4(r['IDF1'])} | {r['IDS']} | "
            f"{float(r['processing_fps']):.2f} | {r['realtime_25fps']} | {float(r['mean_runtime_size']):.1f} |"
        )
    (output / "paper_comparison.md").write_text("\n".join(lines) + "\n")
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--sequences", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--systems", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--trackeval", required=True, type=Path)
    p.add_argument("--target-fps", type=float, default=25.0)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--backend", choices=["auto", "tensorrt", "pytorch"], default="pytorch")
    p.add_argument("--engine", type=Path, default=Path("/content/weights/yolov8n_v16_dynamic_fp16.engine"))
    p.add_argument("--decode-chunk-size", type=int, default=16)
    a = p.parse_args()

    environment()
    names = json.loads(a.sequences.read_text())
    systems = json.loads(a.systems.read_text())
    manifest = dataset_manifest(a.dataset, names)
    if sum(x["frames"] for x in manifest) != 6635 or len(manifest) != 17:
        raise RuntimeError("v16 paper protocol requires the verified 17-sequence / 6635-frame split")

    a.output.mkdir(parents=True, exist_ok=False)
    stripped_systems = [asdict(config_from_system(s)) | {"role": s.get("role", "unspecified")} for s in systems]
    atomic_json(a.output / "configuration.json", {
        "version": "v16",
        "purpose": "final paper comparison and ablation evaluation; no model/config selection",
        "systems": stripped_systems,
        "runtime_systems": systems,
        "target_processing_fps": a.target_fps,
        "ground_truth_filter": GT_FILTER,
        "timing_definition": "decoded/live-frame SceneAnalyzer + Controller + one fresh YOLO inference + ByteTrack",
        "jpeg_decode_policy": "measured separately; not part of processing realtime gate",
        "test_use_policy": "evaluation only; v16 results must not be used to retune v15 proposed settings",
    })
    atomic_json(a.output / "dataset_manifest.json", manifest)

    timing = []
    for i, system in enumerate(systems, 1):
        timing.append(run_system(
            system, a.dataset, manifest, a.weights, a.engine, a.target_fps,
            a.progress_every, a.backend, a.decode_chunk_size, a.output, i, len(systems)
        ))
        atomic_json(a.output / "timing.json", timing)

    print("\n[V16 METRICS] Running pinned TrackEval adapter...", flush=True)
    trackeval_output = a.output / "trackeval"
    subprocess.run([
        sys.executable, str(ROOT / "evaluate.py"), str(a.output),
        "--dataset", str(a.dataset), "--trackeval", str(a.trackeval),
        "--output", str(trackeval_output),
    ], cwd=ROOT, check=True)

    merged = merge_paper_tables(a.output, systems, timing)
    atomic_json(a.output / "paper_protocol.json", {
        "version": "v16",
        "dataset_sequences": len(manifest),
        "dataset_frames": sum(x["frames"] for x in manifest),
        "detector": "YOLOv8n",
        "precision": "PyTorch FP16 on Tesla T4",
        "fresh_inference_per_frame": 1,
        "frame_skipping": False,
        "detection_cache_for_fps": False,
        "target_processing_fps": a.target_fps,
        "metrics": ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDS", "FN", "FP"],
        "quality_evaluator": "evaluate.py + pinned TrackEval",
        "official_visdrone_protocol": False,
        "note": "Custom class-agnostic GT filter is a research protocol and must be stated explicitly in the paper.",
    })

    print("\n=== V16 PAPER TABLE ===", flush=True)
    for r in merged:
        print(
            f"{r['system']}: HOTA={float(r['HOTA']):.4f} IDF1={float(r['IDF1']):.4f} "
            f"MOTA={float(r['MOTA']):.4f} IDS={r['IDS']} FPS={float(r['processing_fps']):.2f} "
            f"RT={r['realtime_25fps']}", flush=True,
        )
    print(f"[DONE] paper CSV: {a.output/'paper_comparison.csv'}", flush=True)
    print(f"[DONE] paper Markdown: {a.output/'paper_comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
