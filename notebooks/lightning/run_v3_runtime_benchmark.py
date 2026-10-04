"""
Exact end-to-end T4 runtime benchmark for the FROZEN v3 adaptive AC-MOT
controller: real detector forward pass (not cache playback) + causal
SceneLayer decision + tracker update, measuring the controller's own
overhead separately from detector/tracker time.
"""
import glob
import json
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, "/teamspace/studios/this_studio")
from acmot_sci import SceneLayer, SceneSpec  # noqa: E402
from adapters.trackers.bytetrack import ByteTrackAdapter  # noqa: E402


def analyze_visual(frame):
    # verbatim copy of run_universal_acmot.analyze_visual (frozen, unchanged);
    # inlined here to avoid pulling in that module's unrelated heavy deps (core.Config etc.)
    small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return dict(edges=float(cv2.Canny(gray, 50, 120).mean() / 255.0), brightness=float(gray.mean()),
                blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()))
from adapters.trackers.oatrack import OATrackAdapter  # noqa: E402
from adapters.types import Detection  # noqa: E402

BASE = "/teamspace/studios/this_studio"
CKPT = f"{BASE}/detector/frozen/yolo11m_visdrone_frozen.pt"
SEQ_DIR = f"{BASE}/data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v"
N_FRAMES = 200
CKPT_TO_EVAL5 = {0: 0, 3: 1, 4: 2, 5: 3, 8: 4}
ACTION_MAP = {"LOW": (1536, 0.70, 0.10), "MEDIUM": (1280, 0.45, 0.40), "HIGH": (1088, 0.60, 0.40)}
SPEC = SceneSpec(t_med=0.29747709701135766, t_high=0.5114928923560124,
                  w_crowd=1.0, w_tiny=0.0, w_edge=0.0, w_dark=0.0, w_blur=0.0)

assert torch.cuda.is_available()
model = YOLO(CKPT)


def make_tracker(host):
    return ByteTrackAdapter() if host == "bytetrack" else OATrackAdapter()


def bench(host):
    frames = sorted(glob.glob(f"{SEQ_DIR}/*.jpg"))[:N_FRAMES]
    tracker = make_tracker(host)
    scene = SceneLayer(SPEC)
    controller_times, det_times, trk_times, total_times = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for fpath in frames:
        t_start = time.perf_counter()
        img = cv2.imread(fpath)
        t_c0 = time.perf_counter()
        stats = analyze_visual(img)
        decision = scene.decide(len(controller_times) + 1, stats)
        res, nms, conf = ACTION_MAP[decision.level]
        t_c1 = time.perf_counter()

        r = model.predict(fpath, imgsz=res, conf=conf, iou=nms, max_det=1000, half=True, verbose=False)[0]
        torch.cuda.synchronize()
        t_det = time.perf_counter()

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()
        dets = [Detection(x1=b[0], y1=b[1], x2=b[2], y2=b[3], confidence=s, class_id=CKPT_TO_EVAL5[c])
                for b, c, s in zip(boxes, classes, scores) if c in CKPT_TO_EVAL5]
        tracks = tracker.update(dets, r.orig_shape)
        scene.observe([[tr.x1, tr.y1, tr.x2, tr.y2] for tr in tracks])
        t_end = time.perf_counter()

        controller_times.append(t_c1 - t_c0)
        det_times.append(t_det - t_c1)
        trk_times.append(t_end - t_det)
        total_times.append(t_end - t_start)
    peak_reserved = torch.cuda.max_memory_reserved()
    arr = np.array(total_times[5:])
    return dict(host=host, n_frames=len(frames), mean_latency_ms=float(arr.mean() * 1000),
                p95_latency_ms=float(np.percentile(arr, 95) * 1000), fps=float(1.0 / arr.mean()),
                mean_detector_ms=float(np.mean(det_times[5:]) * 1000),
                mean_tracker_ms=float(np.mean(trk_times[5:]) * 1000),
                mean_controller_overhead_ms=float(np.mean(controller_times[5:]) * 1000),
                peak_gpu_reserved_gb=peak_reserved / 2**30)


results = [bench(h) for h in ("bytetrack", "oatrack")]
for r in results:
    print(json.dumps(r, indent=1))
json.dump(dict(gpu=torch.cuda.get_device_name(0), action_map=ACTION_MAP, results=results),
          open(f"{BASE}/outputs/acmot_oatrack/V3_RUNTIME_BENCHMARK_RESULT.json", "w"), indent=1)
print("DONE")
