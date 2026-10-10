"""
Real end-to-end T4 runtime benchmark: actual detector forward pass (not
cache playback) + tracker update, for each of the two operating points
used in the final experiment. Measures mean/P95 latency, FPS, detector
time, tracker time, GPU memory, for both ByteTrack and OATrack.
"""
import json
import sys
import time

import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, "/teamspace/studios/this_studio")
from adapters.trackers.bytetrack import ByteTrackAdapter  # noqa: E402
from adapters.trackers.oatrack import OATrackAdapter  # noqa: E402
from adapters.types import Detection  # noqa: E402

BASE = "/teamspace/studios/this_studio"
CKPT = f"{BASE}/detector/frozen/yolo11m_visdrone_frozen.pt"
SEQ_DIR = f"{BASE}/data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v"
N_FRAMES = 200
CKPT_TO_EVAL5 = {0: 0, 3: 1, 4: 2, 5: 3, 8: 4}

assert torch.cuda.is_available()
model = YOLO(CKPT)

OPERATING_POINTS = [
    dict(name="systems_1_2_default", imgsz=1536, nms=0.70, conf=0.01),  # conf=0.01 = cache floor, i.e. no extra filter
    dict(name="systems_3_4_frozen_acmot", imgsz=1088, nms=0.45, conf=0.40),
]


def make_tracker(host):
    return ByteTrackAdapter() if host == "bytetrack" else OATrackAdapter()


def bench(op, host):
    import glob
    frames = sorted(glob.glob(f"{SEQ_DIR}/*.jpg"))[:N_FRAMES]
    tracker = make_tracker(host)
    det_times, trk_times, total_times = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(f, imgsz=op["imgsz"], conf=op["conf"], iou=op["nms"], max_det=1000, half=True,
                           verbose=False)[0]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()
        dets = [Detection(x1=b[0], y1=b[1], x2=b[2], y2=b[3], confidence=s, class_id=CKPT_TO_EVAL5[c])
                for b, c, s in zip(boxes, classes, scores) if c in CKPT_TO_EVAL5]
        shape = r.orig_shape
        tracks = tracker.update(dets, shape)
        t2 = time.perf_counter()
        det_times.append(t1 - t0)
        trk_times.append(t2 - t1)
        total_times.append(t2 - t0)
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    arr = np.array(total_times[5:])  # drop first 5 frames (warmup)
    det_arr = np.array(det_times[5:])
    trk_arr = np.array(trk_times[5:])
    return dict(op=op["name"], host=host, n_frames=len(frames),
                mean_latency_ms=float(arr.mean() * 1000), p95_latency_ms=float(np.percentile(arr, 95) * 1000),
                fps=float(1.0 / arr.mean()), mean_detector_ms=float(det_arr.mean() * 1000),
                mean_tracker_ms=float(trk_arr.mean() * 1000),
                peak_gpu_allocated_gb=peak_alloc / 2**30, peak_gpu_reserved_gb=peak_reserved / 2**30)


results = []
for op in OPERATING_POINTS:
    for host in ("bytetrack", "oatrack"):
        r = bench(op, host)
        print(json.dumps(r, indent=1))
        results.append(r)

json.dump(dict(gpu=torch.cuda.get_device_name(0), results=results),
          open(f"{BASE}/outputs/acmot_oatrack/RUNTIME_BENCHMARK_RESULT.json", "w"), indent=1)
print("DONE")
