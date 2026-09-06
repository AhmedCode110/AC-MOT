"""Measure fresh end-to-end FPS for the three selected AC-MOT systems on a Tesla T4.

This is a throughput gate, not the final frozen accuracy evaluation. It uses fresh
YOLOv8n FP32 inference plus SceneAnalyzer/Controller and ByteTrack, includes frame
read time, and excludes warmup and output serialization.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core import Config, Controller, atomic_json
from experiment import dataset_manifest, detect, environment, make_tracker, new_model, sync, track, visual


def bar(done, total, width=30):
    frac = 0.0 if total <= 0 else min(max(done / total, 0.0), 1.0)
    filled = int(round(frac * width))
    return '[' + '=' * filled + '>' + '.' * max(width - filled - 1, 0) + ']'


def run_one(system, dataset, manifest, weights, target_fps, gate_frames, progress_every):
    import cv2
    import torch

    c = Config(**system).validate()
    total_frames = sum(x['frames'] for x in manifest)
    model = new_model(weights)
    torch.cuda.reset_peak_memory_stats()

    measured_seconds = 0.0
    measured_frames = 0
    latencies = []
    gate_checked = False
    failed_gate = False
    start_wall = time.perf_counter()

    print(f'\nSPEEDTEST START | system={c.name} | target={target_fps:.2f} FPS | gate={gate_frames} frames', flush=True)

    for seq in manifest:
        sn = seq['sequence']
        paths = [dataset / 'sequences' / sn / f for f in seq['frame_sha256']]
        first = cv2.imread(str(paths[0]))
        if first is None:
            raise ValueError(f'Unreadable first frame: {paths[0]}')

        warm_sizes = [c.size] if c.policy == 'fixed' else [640, 736, 832]
        for size in warm_sizes:
            for _ in range(3):
                detect(model, first, size, c.nms)

        control = Controller(c)
        tracker = make_tracker(c)
        previous = []

        for frame, path in enumerate(paths, 1):
            sync()
            t0 = time.perf_counter()
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f'Unreadable frame: {path}')

            v = visual(img) if frame == 1 or frame % 10 == 1 else {}
            params = control.choose(frame, v, previous)
            dets = detect(model, img, params['size'], params['nms'])
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
                    f'FPS={fps:6.2f} | target={target_fps:.2f} | {status} | ETA={eta/60:.1f}m',
                    flush=True,
                )

            if not gate_checked and measured_frames >= gate_frames:
                gate_checked = True
                fps = measured_frames / measured_seconds
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
        'timing_includes': 'frame read + scene analysis/controller + YOLOv8n FP32 + ByteTrack',
        'timing_excludes': 'detector warmup + output serialization',
    }
    print(
        f'SPEEDTEST DONE | system={c.name} | status={result["status"]} | '
        f'frames={measured_frames} | FPS={fps:.2f} | p95={result["p95_ms"]:.2f} ms',
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return result


def main():
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

    if a.target_fps <= 0 or a.gate_frames < 1 or a.progress_every < 1:
        raise ValueError('Invalid speed-test thresholds')

    environment()
    names = json.loads(a.sequences.read_text())
    systems = json.loads(a.systems.read_text())
    manifest = dataset_manifest(a.dataset, names)
    a.output.mkdir(parents=True, exist_ok=False)

    atomic_json(a.output / 'configuration.json', {
        'purpose': 'throughput gate only; not final frozen accuracy evaluation',
        'target_fps': a.target_fps,
        'gate_frames': a.gate_frames,
        'progress_every': a.progress_every,
        'systems': systems,
        'precision': 'FP32',
        'timing_includes': 'frame read + scene analysis/controller + YOLOv8n + ByteTrack',
        'timing_excludes': 'detector warmup + output serialization',
    })
    atomic_json(a.output / 'dataset_manifest.json', manifest)

    results = []
    for system in systems:
        result = run_one(
            system,
            a.dataset,
            manifest,
            a.weights,
            a.target_fps,
            a.gate_frames,
            a.progress_every,
        )
        results.append(result)
        atomic_json(a.output / 'speedtest_results.json', results)

    passed = [r['system'] for r in results if r['status'] == 'PASS_REALTIME']
    failed = [r['system'] for r in results if r['status'] != 'PASS_REALTIME']
    print('\n=== SPEEDTEST SUMMARY ===', flush=True)
    for r in results:
        print(f'{r["system"]}: {r["fps"]:.2f} FPS -> {r["status"]}', flush=True)
    print('PASS:', ', '.join(passed) if passed else 'none', flush=True)
    print('FAIL:', ', '.join(failed) if failed else 'none', flush=True)
    print(f'RESULTS: {a.output / "speedtest_results.json"}', flush=True)


if __name__ == '__main__':
    main()
