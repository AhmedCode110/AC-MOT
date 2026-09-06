"""AC-MOT v17 presentation-aligned paper evaluation.

Compares the v10-style scientific ablation chain using the modern v15 FP16
live-frame timing protocol and pinned TrackEval metrics.

The script reports, but does not silently retune:
- highest-HOTA system overall;
- highest-HOTA system among systems that actually achieve >= target FPS.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import Config, atomic_json
from core_v17 import PresentationController, PresentationSpec, tracker_settings
from experiment import dataset_manifest, environment, recording, sync, track, visual
from scripts.speedtest_top3_v15 import (
    bar, copy_sequence_to_local, decode_chunk, detect_realtime, fmt_seconds, make_backend, summarize,
)

GT_FILTER = dict(categories=[1, 4, 5, 6, 9], score=1, occlusion_lt=2, truncation_lt=2)


def make_tracker(profile):
    from ultralytics.trackers.byte_tracker import BYTETracker
    t = tracker_settings(profile)
    return BYTETracker(SimpleNamespace(
        track_high_thresh=t["high"], track_low_thresh=t["low"],
        new_track_thresh=t["new"], track_buffer=t["buffer"],
        match_thresh=t["match"], fuse_score=t["fuse"]), frame_rate=30)


def spec_from_dict(d):
    return PresentationSpec(
        name=d["name"],
        adaptive_threshold=bool(d.get("adaptive_threshold", True)),
        adaptive_resolution=bool(d.get("adaptive_resolution", True)),
        smoothing_window=int(d.get("smoothing_window", 7)),
        analysis_stride=int(d.get("analysis_stride", 10)),
        tracker_profile=str(d.get("tracker_profile", "tuned")),
    ).validate()


def run_system(system, dataset, manifest, weights, engine, target_fps, progress_every,
               backend_pref, decode_chunk_size, output, index, total):
    import torch

    spec = spec_from_dict(system)
    tracker_cfg = tracker_settings(spec.tracker_profile)
    controller = PresentationController(spec)
    model, backend, fallback = make_backend(weights, backend_pref, engine)
    torch.cuda.reset_peak_memory_stats()

    folder = output / spec.name
    folder.mkdir(parents=True, exist_ok=False)
    total_frames = sum(s["frames"] for s in manifest)
    processing_seconds = 0.0
    decode_seconds = 0.0
    drive_copy_seconds = 0.0
    latencies, decode_samples, sizes, confs, nms_values, sci_values = [], [], [], [], [], []
    stage = {k: [] for k in ["analysis_controller", "detector", "tracker"]}
    measured = 0
    wall0 = time.perf_counter()

    print("\n" + "=" * 108, flush=True)
    print(f"[V17 {index}/{total}] {spec.name}", flush=True)
    print(
        f"[ABLATION] threshold={spec.adaptive_threshold} resolution={spec.adaptive_resolution} "
        f"smooth={spec.smoothing_window} stride={spec.analysis_stride} tracker={spec.tracker_profile}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="acmot_v17_") as tmp:
        local_root = Path(tmp)
        for seq_i, seq in enumerate(manifest, 1):
            paths, copy_s = copy_sequence_to_local(dataset, seq, local_root)
            drive_copy_seconds += copy_s
            images0, _ = decode_chunk(paths[:1])
            first = images0[0]
            # Warm every resolution that this system can request. Excluded from FPS.
            warm_sizes = [640, 736, 832] if spec.adaptive_resolution else [640]
            for size in warm_sizes:
                for _ in range(3):
                    detect_realtime(model, first, size, 0.45, 0.19, backend)
            sync()

            controller = PresentationController(spec)  # clean state per sequence
            tracker = make_tracker(spec.tracker_profile)
            previous = []
            target = folder / f"{seq['sequence']}.frames.jsonl.gz"
            written = 0

            with gzip.open(target, "wt") as dest:
                for chunk_start in range(0, len(paths), decode_chunk_size):
                    chunk_paths = paths[chunk_start:chunk_start + decode_chunk_size]
                    images, dts = decode_chunk(chunk_paths)
                    for offset, (img, decode_s) in enumerate(zip(images, dts)):
                        frame = chunk_start + offset + 1
                        decode_seconds += decode_s
                        decode_samples.append(decode_s)

                        process0 = time.perf_counter()
                        analyze = frame == 1 or (frame - 1) % spec.analysis_stride == 0
                        t0 = time.perf_counter()
                        v = visual(img) if analyze else {}
                        params = controller.choose(frame, v, previous)
                        params.update(high=tracker_cfg["high"], new=tracker_cfg["new"])
                        ac_s = time.perf_counter() - t0

                        sync(); t0 = time.perf_counter()
                        dets = detect_realtime(
                            model, img, params["size"], params["nms"], params["conf"], backend
                        )
                        sync(); det_s = time.perf_counter() - t0

                        t0 = time.perf_counter()
                        tracks, kept = track(tracker, dets, img.shape[:2], params)
                        trk_s = time.perf_counter() - t0
                        # Presentation SceneAnalyzer cue B: previous detector boxes.
                        previous = kept
                        elapsed = time.perf_counter() - process0

                        processing_seconds += elapsed
                        measured += 1
                        written += 1
                        latencies.append(elapsed)
                        stage["analysis_controller"].append(ac_s)
                        stage["detector"].append(det_s)
                        stage["tracker"].append(trk_s)
                        sizes.append(params["size"])
                        confs.append(params["conf"])
                        nms_values.append(params["nms"])
                        sci_values.append(params["sci"])
                        dest.write(json.dumps(recording(frame, tracks, params, elapsed)) + "\n")

                        if measured == 1 or measured % progress_every == 0:
                            fps = measured / processing_seconds
                            wall = time.perf_counter() - wall0
                            eta = wall / measured * (total_frames - measured) if measured else 0
                            print(
                                f"{bar(measured,total_frames)} {100*measured/total_frames:6.2f}% | "
                                f"{spec.name} | seq={seq['sequence']} frame={frame}/{seq['frames']} | "
                                f"SCI={params['sci']:.3f} conf={params['conf']:.3f} "
                                f"iou={params['nms']:.3f} imgsz={params['size']} | FPS={fps:.2f} | "
                                f"ETA={fmt_seconds(eta)}",
                                flush=True,
                            )
                    del images

            if written != seq["frames"]:
                raise RuntimeError(f"Frame count mismatch: {spec.name}/{seq['sequence']}")

    fps = measured / processing_seconds
    result = dict(
        system=spec.name,
        ablation=dict(
            adaptive_threshold=spec.adaptive_threshold,
            adaptive_resolution=spec.adaptive_resolution,
            smoothing_window=spec.smoothing_window,
            analysis_stride=spec.analysis_stride,
            tracker_profile=spec.tracker_profile,
        ),
        backend=backend,
        backend_fallback_reason=fallback,
        frames=measured,
        processing_seconds=processing_seconds,
        processing_fps=fps,
        realtime_25fps=fps >= target_fps,
        processing_p95_ms=float(np.percentile(latencies, 95) * 1000),
        jpeg_decode_seconds=decode_seconds,
        jpeg_decode_profile=summarize(decode_samples),
        serialized_dataset_playback_fps=measured / (processing_seconds + decode_seconds),
        drive_copy_seconds_excluded=drive_copy_seconds,
        peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
        mean_imgsz=float(np.mean(sizes)),
        mean_conf=float(np.mean(confs)),
        mean_nms_iou=float(np.mean(nms_values)),
        mean_sci=float(np.mean(sci_values)),
        stage_profile={k: summarize(v) for k, v in stage.items()},
    )
    atomic_json(folder / "system_summary.json", result)
    print(f"[DONE] {spec.name} | {fps:.2f} FPS | RT={'YES' if fps >= target_fps else 'NO'}", flush=True)
    del model
    torch.cuda.empty_cache()
    return result


def make_table(output, systems, timing):
    rows = list(csv.DictReader((output / "trackeval" / "summary.csv").open()))
    metrics = {r["system"]: r for r in rows}
    timing_map = {r["system"]: r for r in timing}
    merged = []
    for s in systems:
        name = s["name"]
        m, t = metrics[name], timing_map[name]
        merged.append(dict(
            system=name,
            HOTA=float(m["HOTA"]), DetA=float(m["DetA"]), AssA=float(m["AssA"]),
            MOTA=float(m["MOTA"]), IDF1=float(m["IDF1"]), IDS=int(float(m["IDS"])),
            FN=int(float(m["FN"])), FP=int(float(m["FP"])),
            FPS=float(t["processing_fps"]), p95_ms=float(t["processing_p95_ms"]),
            RT=bool(t["realtime_25fps"]), mean_imgsz=float(t["mean_imgsz"]),
            mean_conf=float(t["mean_conf"]), mean_nms_iou=float(t["mean_nms_iou"]),
        ))

    cols = list(merged[0])
    with (output / "paper_comparison_v17.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(merged)

    overall = max(merged, key=lambda r: (r["HOTA"], r["IDF1"], r["MOTA"]))
    realtime = [r for r in merged if r["RT"]]
    best_rt = max(realtime, key=lambda r: (r["HOTA"], r["IDF1"], r["MOTA"])) if realtime else None
    selection = dict(
        rule="Highest HOTA among measured systems meeting >=25 processing FPS; IDF1 then MOTA are tie-breakers",
        best_quality_overall=overall,
        best_realtime=best_rt,
        warning="This ranking describes this evaluation only. Do not retune on the same held-out results and then claim an unbiased test result.",
    )
    atomic_json(output / "BEST_RESULT.json", selection)

    lines = [
        "# AC-MOT v17 Presentation-Aligned Comparison", "",
        "| Method | HOTA | DetA | AssA | MOTA | IDF1 | IDS | FPS | RT>=25 | Mean imgsz |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for r in merged:
        lines.append(
            f"| {r['system']} | {r['HOTA']:.4f} | {r['DetA']:.4f} | {r['AssA']:.4f} | "
            f"{r['MOTA']:.4f} | {r['IDF1']:.4f} | {r['IDS']} | {r['FPS']:.2f} | "
            f"{'YES' if r['RT'] else 'NO'} | {r['mean_imgsz']:.1f} |"
        )
    lines += ["", f"Best quality overall: **{overall['system']}**"]
    lines.append(f"Best realtime: **{best_rt['system']}**" if best_rt else "Best realtime: **NONE**")
    (output / "paper_comparison_v17.md").write_text("\n".join(lines) + "\n")
    return merged, selection


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--sequences", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--systems", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--trackeval", required=True, type=Path)
    p.add_argument("--target-fps", type=float, default=25.0)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--backend", default="pytorch", choices=["pytorch", "auto", "tensorrt"])
    p.add_argument("--engine", type=Path, default=Path("/content/weights/yolov8n_v17_dynamic_fp16.engine"))
    p.add_argument("--decode-chunk-size", type=int, default=16)
    a = p.parse_args()

    environment()
    names = json.loads(a.sequences.read_text())
    systems = json.loads(a.systems.read_text())
    manifest = dataset_manifest(a.dataset, names)
    if len(manifest) != 17 or sum(s["frames"] for s in manifest) != 6635:
        raise RuntimeError("v17 requires the verified 17-sequence / 6635-frame VisDrone split")

    a.output.mkdir(parents=True, exist_ok=False)
    atomic_json(a.output / "configuration.json", dict(
        version="v17", purpose="presentation-aligned ablation and realtime comparison",
        systems=systems, ground_truth_filter=GT_FILTER,
        smart_calibrator="Recovered exact v10 mapping: adaptive conf + NMS IoU + 640/736/832 resolution",
        timing_definition="decoded/live-frame analysis + controller + one fresh YOLOv8n FP16 inference + ByteTrack",
        selection_rule="highest HOTA among measured >=25 FPS systems; no hidden retuning",
    ))
    atomic_json(a.output / "dataset_manifest.json", manifest)

    timing = []
    for i, system in enumerate(systems, 1):
        timing.append(run_system(
            system, a.dataset, manifest, a.weights, a.engine, a.target_fps,
            a.progress_every, a.backend, a.decode_chunk_size, a.output, i, len(systems)
        ))
    atomic_json(a.output / "timing.json", timing)

    trackeval_out = a.output / "trackeval"
    subprocess.run([
        sys.executable, str(ROOT / "evaluate.py"), str(a.output),
        "--dataset", str(a.dataset), "--trackeval", str(a.trackeval),
        "--output", str(trackeval_out),
    ], cwd=ROOT, check=True)

    merged, selection = make_table(a.output, systems, timing)
    print("\n=== V17 FINAL PAPER TABLE ===")
    for r in merged:
        print(
            f"{r['system']}: HOTA={r['HOTA']:.4f} IDF1={r['IDF1']:.4f} "
            f"MOTA={r['MOTA']:.4f} IDS={r['IDS']} FPS={r['FPS']:.2f} RT={'YES' if r['RT'] else 'NO'}"
        )
    print("\n=== AUTOMATIC RESULT SUMMARY ===")
    print("Best quality overall:", selection["best_quality_overall"]["system"])
    print("Best realtime:", selection["best_realtime"]["system"] if selection["best_realtime"] else "NONE")
    print("BEST_RESULT:", a.output / "BEST_RESULT.json")


if __name__ == "__main__":
    main()
