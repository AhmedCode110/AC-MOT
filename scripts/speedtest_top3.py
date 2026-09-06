"""Measure deployment-style end-to-end FPS for the three selected AC-MOT systems.

The research architecture and tracker/controller settings are unchanged. For the
realtime throughput test only, frames are copied from Drive to Colab local SSD
before timing and YOLOv8n runs in FP16 on a Tesla T4, matching the performance-
critical execution choices used by the original v10 realtime work.

Measured time includes local frame read + SceneAnalyzer/Controller + one fresh
YOLOv8n FP16 inference per frame + ByteTrack. Sequence copy, detector warmup and
output serialization are excluded. This remains a throughput gate, not the final
frozen accuracy evaluation.
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

print('[BOOT] AC-MOT realtime speed test started.', flush=True)
print('[BOOT] Loading Python dependencies...', flush=True)
import numpy as np
print('[OK] NumPy loaded.', flush=True)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print(f'[BOOT] Repository root: {ROOT}', flush=True)
print('[BOOT] Loading AC-MOT research modules...', flush=True)

from core import CLASSES, Config, Controller, atomic_json, boxes
from experiment import dataset_manifest, environment, make_tracker, new_model, sync, track, visual
print('[OK] AC-MOT modules loaded.', flush=True)


def detect_realtime(model, img, size, nms):
    """Fresh YOLOv8n FP16 inference; detector semantics otherwise match experiment.detect."""
    r = model.predict(
        img,
        conf=.01,
        iou=nms,
        imgsz=size,
        classes=CLASSES,
        max_det=1000,
        half=True,
        device=0,
        verbose=False,
    )[0]
    if not bool(model.predictor.model.fp16):
        raise RuntimeError('Realtime speedtest expected FP16 inference on CUDA')
    return boxes(r.boxes.data.cpu().numpy())


def bar(done, total, width=30):
    frac = 0.0 if total <= 0 else min(max(done / total, 0.0), 1.0)
    filled = int(round(frac * width))
    if filled >= width:
        return '[' + '=' * width + ']'
    return '[' + '=' * filled + '>' + '.' * max(width - filled - 1, 0) + ']'


def copy_sequence_to_local(dataset, seq, local_root):
    sn = seq['sequence']
    src = dataset / 'sequences' / sn
    dst = local_root / sn
    if dst.exists():
        shutil.rmtree(dst)
    print(f'[LOCAL] Copying {sn} from Drive to Colab local SSD (excluded from FPS timing)...', flush=True)
    shutil.copytree(src, dst)
    paths = [dst / f for f in seq['frame_sha256']]
    if len(paths) != seq['frames'] or any(not p.is_file() for p in paths):
        raise RuntimeError(f'Local sequence copy verification failed: {sn}')
    print(f'[OK] Local copy ready: {dst} | frames={len(paths)}', flush=True)
    return paths


def run_one(system, dataset, manifest, weights, target_fps, gate_frames, progress_every, system_index, system_total):
    import cv2
    import torch

    c = Config(**system).validate()
    total_frames = sum(x['frames'] for x in manifest)

    print('\n' + '=' * 88, flush=True)
    print(f'[SYSTEM {system_index}/{system_total}] {c.name}', flush=True)
    print(f'[SYSTEM] Policy={c.policy} | base size={c.size} | target={target_fps:.2f} FPS', flush=True)
    print('[SYSTEM] Realtime execution: local SSD frames + YOLOv8n FP16 + current AC-MOT pipeline.', flush=True)
    print(f'[SYSTEM] Realtime decision will be made after {gate_frames} measured frames.', flush=True)
    print('[MODEL] Loading YOLOv8n weights onto Tesla T4...', flush=True)
    model = new_model(weights)
    print('[OK] YOLOv8n model loaded.', flush=True)
    torch.cuda.reset_peak_memory_stats()

    measured_seconds = 0.0
    measured_frames = 0
    latencies = []
    gate_checked = False
    failed_gate = False
    start_wall = time.perf_counter()

    print(f'[RUN] Preparing {len(manifest)} sequences / {total_frames} frames.', flush=True)
    print('[RUN] Timing includes local frame read + SceneAnalyzer/Controller + YOLOv8n FP16 + ByteTrack.', flush=True)
    print('[RUN] Drive-to-local copy, warmup and output serialization are excluded from measured FPS.', flush=True)

    with tempfile.TemporaryDirectory(prefix='acmot_rt_') as tmp:
        local_root = Path(tmp)

        for seq_index, seq in enumerate(manifest, 1):
            sn = seq['sequence']
            print(f'\n[SEQUENCE {seq_index}/{len(manifest)}] {sn} | {seq["frames"]} frames', flush=True)
            paths = copy_sequence_to_local(dataset, seq, local_root)

            print('[SEQUENCE] Reading first local frame for validation and warmup...', flush=True)
            first = cv2.imread(str(paths[0]))
            if first is None:
                raise ValueError(f'Unreadable first local frame: {paths[0]}')
            print('[OK] First local frame readable.', flush=True)

            warm_sizes = [c.size] if c.policy == 'fixed' else [640, 736, 832]
            print(f'[WARMUP] Starting FP16 detector warmup at sizes: {warm_sizes}', flush=True)
            for size in warm_sizes:
                for rep in range(1, 4):
                    print(f'[WARMUP] imgsz={size} | pass {rep}/3 | waiting for GPU inference...', flush=True)
                    detect_realtime(model, first, size, c.nms)
            sync()
            print('[OK] Warmup complete. Measured realtime timing starts now.', flush=True)

            print('[TRACKER] Creating fresh Controller + ByteTrack state for this sequence...', flush=True)
            control = Controller(c)
            tracker = make_tracker(c)
            previous = []
            print('[OK] Tracker/controller ready.', flush=True)

            for frame, path in enumerate(paths, 1):
                sync()
                t0 = time.perf_counter()

                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f'Unreadable frame: {path}')

                v = visual(img) if frame == 1 or frame % 10 == 1 else {}
                params = control.choose(frame, v, previous)
                dets = detect_realtime(model, img, params['size'], params['nms'])
                tracks, kept = track(tracker, dets, img.shape[:2], params)
                previous = kept if c.detector_feedback else tracks[:, [0, 1, 2, 3, 5, 6]]

                sync()
                elapsed = time.perf_counter() - t0
                measured_seconds += elapsed
                measured_frames += 1
                latencies.append(elapsed)

                fps = measured_frames / measured_seconds
                should_print = measured_frames == 1 or measured_frames % progress_every == 0 or measured_frames == total_frames
                if should_print:
                    status = 'GATING' if measured_frames < gate_frames else ('REALTIME' if fps >= target_fps else 'BELOW-RT')
                    elapsed_wall = time.perf_counter() - start_wall
                    eta = (elapsed_wall / measured_frames) * (total_frames - measured_frames) if measured_frames else 0.0
                    print(
                        f'{bar(measured_frames, total_frames)} '
                        f'{100.0 * measured_frames / total_frames:6.2f}% | '
                        f'system={c.name} | seq={sn} | frame={frame}/{seq["frames"]} | '
                        f'imgsz={params["size"]} | scene={params["scene"]} | SCI={params["sci"]:.3f} | '
                        f'FPS={fps:6.2f} | target={target_fps:.2f} | {status} | ETA={eta/60:.1f}m',
                        flush=True,
                    )

                if not gate_checked and measured_frames >= gate_frames:
                    gate_checked = True
                    fps = measured_frames / measured_seconds
                    print(f'[GATE] {gate_frames} measured frames reached. Checking realtime requirement...', flush=True)
                    if fps < target_fps:
                        failed_gate = True
                        print(
                            f'REALTIME CHECK FAIL | system={c.name} | average_FPS={fps:.2f} | '
                            f'target={target_fps:.2f} | stopping this system early',
                            flush=True,
                        )
                        break
                    print(
                        f'REALTIME CHECK PASS | system={c.name} | average_FPS={fps:.2f} | target={target_fps:.2f}',
                        flush=True,
                    )
                    print('[RUN] Realtime gate passed. Continuing through remaining frames for a stable FPS estimate.', flush=True)

            shutil.rmtree(local_root / sn, ignore_errors=True)
            print(f'[LOCAL] Released local copy for {sn}.', flush=True)
            if failed_gate:
                break

    fps = measured_frames / measured_seconds if measured_seconds else 0.0
    result = {
        'system': c.name,
        'configuration': asdict(c),
        'status': 'FAIL_REALTIME' if failed_gate else ('PASS_REALTIME' if fps >= target_fps else 'FAIL_REALTIME'),
        'target_fps': target_fps,
        'gate_frames': gate_frames,
        'measured_frames': measured_frames,
        'seconds': measured_seconds,
        'fps': fps,
        'p95_ms': float(np.percentile(latencies, 95) * 1000) if latencies else None,
        'peak_gpu_bytes': int(torch.cuda.max_memory_allocated()),
        'precision': 'FP16',
        'frame_source': 'Colab local SSD; copied from Drive before timing',
        'timing_includes': 'local frame read + scene analysis/controller + YOLOv8n FP16 + ByteTrack',
        'timing_excludes': 'Drive-to-local sequence copy + detector warmup + output serialization',
    }
    print(
        f'[RESULT] {c.name} | status={result["status"]} | '
        f'frames={measured_frames} | FPS={fps:.2f} | p95={result["p95_ms"]:.2f} ms',
        flush=True,
    )
    print('[CLEANUP] Releasing YOLO model and GPU cache before next system...', flush=True)
    del model
    torch.cuda.empty_cache()
    print('[OK] Cleanup complete.', flush=True)
    return result


def main():
    print('[ARGS] Reading speed-test arguments...', flush=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--sequences', required=True, type=Path)
    p.add_argument('--weights', required=True, type=Path)
    p.add_argument('--systems', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--target-fps', required=True, type=float)
    p.add_argument('--gate-frames', required=True, type=int)
    p.add_argument('--progress-every', type=int, default=25)
    a = p.parse_args()
    print('[OK] Arguments loaded.', flush=True)

    if a.target_fps <= 0 or a.gate_frames < 1 or a.progress_every < 1:
        raise ValueError('Invalid speed-test thresholds')

    print('[GPU] Checking CUDA, package versions, and Tesla T4 requirement...', flush=True)
    environment()
    print('[OK] Environment check passed.', flush=True)

    print(f'[DATASET] Reading sequence list from: {a.sequences}', flush=True)
    names = json.loads(a.sequences.read_text())
    print(f'[OK] {len(names)} sequences selected.', flush=True)

    print(f'[SYSTEMS] Reading Top-3 configurations from: {a.systems}', flush=True)
    systems = json.loads(a.systems.read_text())
    print('[OK] Systems: ' + ', '.join(x['name'] for x in systems), flush=True)

    print('[DATASET] Building and verifying dataset manifest from Drive...', flush=True)
    manifest = dataset_manifest(a.dataset, names)
    total_frames = sum(x['frames'] for x in manifest)
    print(f'[OK] Dataset manifest ready: {len(manifest)} sequences, {total_frames} frames.', flush=True)

    print(f'[OUTPUT] Creating result folder: {a.output}', flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    print('[OK] Output folder created.', flush=True)

    print('[SAVE] Writing speed-test configuration and dataset manifest...', flush=True)
    atomic_json(a.output / 'configuration.json', {
        'purpose': 'deployment-style throughput gate only; not final frozen accuracy evaluation',
        'target_fps': a.target_fps,
        'gate_frames': a.gate_frames,
        'progress_every': a.progress_every,
        'systems': systems,
        'precision': 'FP16',
        'frame_source': 'copy each sequence from Drive to Colab local SSD before timing',
        'timing_includes': 'local frame read + scene analysis/controller + YOLOv8n FP16 + ByteTrack',
        'timing_excludes': 'Drive-to-local sequence copy + detector warmup + output serialization',
        'inference_rule': 'exactly one fresh YOLOv8n inference per measured frame',
    })
    atomic_json(a.output / 'dataset_manifest.json', manifest)
    print('[OK] Metadata saved. Starting Top-3 realtime throughput test.', flush=True)

    results = []
    for idx, system in enumerate(systems, 1):
        result = run_one(
            system,
            a.dataset,
            manifest,
            a.weights,
            a.target_fps,
            a.gate_frames,
            a.progress_every,
            idx,
            len(systems),
        )
        results.append(result)
        print('[SAVE] Saving current speed-test results to Drive...', flush=True)
        atomic_json(a.output / 'speedtest_results.json', results)
        print('[OK] Results saved.', flush=True)

    passed = [r['system'] for r in results if r['status'] == 'PASS_REALTIME']
    failed = [r['system'] for r in results if r['status'] != 'PASS_REALTIME']
    print('\n=== SPEEDTEST SUMMARY ===', flush=True)
    for r in results:
        print(f'{r["system"]}: {r["fps"]:.2f} FPS -> {r["status"]}', flush=True)
    print('PASS:', ', '.join(passed) if passed else 'none', flush=True)
    print('FAIL:', ', '.join(failed) if failed else 'none', flush=True)
    print(f'RESULTS: {a.output / "speedtest_results.json"}', flush=True)
    print('[DONE] AC-MOT Top-3 realtime speed test finished.', flush=True)


if __name__ == '__main__':
    main()
