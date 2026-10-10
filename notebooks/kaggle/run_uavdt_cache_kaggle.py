"""
Kaggle GPU kernel: UAVDT transfer cache generation for the frozen v3
AC-MOT controller's 3 exact (resolution, NMS) action pairs. Images
sourced from the Kaggle-mounted shakaibkaggle/uavdt-dataset (verified
exact match to the frozen 20-sequence / 16592-frame protocol). Detector
checkpoint fetched from its original Hugging Face source and sha256-
verified. Output cache (small .npz files) is the only thing retrieved
back to the Mac.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/kaggle/working")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.200"], check=True)

import cv2
import numpy as np
import torch
from ultralytics import YOLO

IMG_ROOT = "/kaggle/input/datasets/shakaibkaggle/uavdt-dataset"  # dataset_sources mounts here by default (last path segment)
OUT = "/kaggle/working/uavdt_cache_smoke" if os.environ.get("UAVDT_SMOKE") == "1" else "/kaggle/working/uavdt_cache"
# the exact bytes we froze and used for every VisDrone experiment, uploaded as our own private
# dataset -- NOT re-downloaded from Hugging Face, since that URL's content changed since freeze
# (re-download attempts here got 40595948 bytes / sha mismatch vs our frozen 40534380 bytes).
# Confirmed via smoke test (kernel v7): with 2 dataset_sources attached, our own dataset mounts
# WITHOUT the owner/"datasets" prefix, while shakaibkaggle's keeps it -- empirically resolved,
# not worth over-explaining the platform's mount-naming inconsistency.
CKPT = "/kaggle/input/acmot-frozen-yolo11m-visdrone/yolo11m_visdrone_frozen.pt"
FROZEN_SHA256 = "c6f98247e7c731a567aaf37ca7b808beff588b687623e028b25a2e1d846e96e7"

ACTION_RES_NMS = [(1536, 0.70), (1280, 0.45), (1088, 0.60)]
CKPT_TO_EVAL5 = {0: 0, 3: 1, 4: 2, 5: 3, 8: 4}

SEQS = "M0203 M0205 M0208 M0209 M0403 M0601 M0602 M0606 M0701 M0801 M0802 M1001 M1004 M1007 M1009 M1101 M1301 M1302 M1303 M1401".split()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print("=== SMOKE CHECKLIST ===")
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
HALF = torch.cuda.is_available()
assert HALF, "No CUDA GPU visible."
print("GPU visible:", torch.cuda.get_device_name(0))
import ultralytics
print("ultralytics:", ultralytics.__version__)

assert os.path.exists(CKPT), f"checkpoint dataset not mounted at {CKPT}"
print("checkpoint size:", os.path.getsize(CKPT), "bytes (expected 40534380)")
ckpt_sha = _sha256(CKPT)
print("frozen detector sha256:", ckpt_sha, "MATCH" if ckpt_sha == FROZEN_SHA256 else "MISMATCH")
assert ckpt_sha == FROZEN_SHA256, "checkpoint sha256 mismatch"

det_model = YOLO(CKPT)
print("detector loaded OK")

SMOKE = os.environ.get("UAVDT_SMOKE", "0") == "1"  # FULL RUN

# build a seq -> file index in ONE directory walk (20 separate recursive globs over the
# 77819-file dataset was the actual bottleneck -- this does a single pass instead)
t_index0 = time.time()
seq_set = set(SEQS)
seq_files = {s: [] for s in SEQS}
for root, dirs, files in os.walk(IMG_ROOT):
    if os.path.basename(root) != "img":
        continue
    for f in files:
        if not f.endswith(".jpg"):
            continue
        seq = f.split("_img")[0]
        if seq in seq_set:
            seq_files[seq].append(os.path.join(root, f))
for s in SEQS:
    seq_files[s].sort()
    assert len(seq_files[s]) > 0, f"no images found for {s}"
    if SMOKE:
        seq_files[s] = seq_files[s][:3]
    print(s, "frames found:", len(seq_files[s]))
print("file index built in", round(time.time() - t_index0, 1), "s")

total_expected = sum(len(v) for v in seq_files.values())
print("total frames:", total_expected, "SMOKE" if SMOKE else "FULL")

t0 = time.time()
forward_passes = 0
for res, nms in ACTION_RES_NMS:
    d = f"{OUT}/r{res}_n{int(round(100 * nms))}"
    os.makedirs(d, exist_ok=True)
    for s in SEQS:
        out_f = f"{d}/{s}.npz"
        if os.path.exists(out_f):
            print("skip", res, nms, s)
            continue
        frames = seq_files[s]
        rows = []
        shape = None
        for t_, f in enumerate(frames, 1):
            r = det_model.predict(f, imgsz=res, conf=0.01, iou=nms, max_det=1000, half=HALF, verbose=False)[0]
            forward_passes += 1
            shape = r.orig_shape
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for b, sc, c in zip(xyxy, conf, cls):
                if int(c) not in CKPT_TO_EVAL5:
                    continue
                rows.append([t_, *b.tolist(), float(sc), CKPT_TO_EVAL5[int(c)]])
        np.savez_compressed(out_f, det=np.asarray(rows, np.float64).reshape(-1, 7),
                            shape=np.asarray(shape), frames=len(frames))
        print(res, nms, s, "frames", len(frames), "dets", len(rows), "elapsed", round(time.time() - t0, 1))

print("DONE. forward_passes", forward_passes, "wall_seconds", time.time() - t0)
json.dump(dict(forward_passes=forward_passes, wall_seconds=time.time() - t0, action_res_nms=ACTION_RES_NMS,
               seq_frame_counts={s: len(v) for s, v in seq_files.items()}),
          open(f"{OUT}/UAVDT_CACHE_STATS.json", "w"), indent=1)
