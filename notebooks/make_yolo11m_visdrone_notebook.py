"""Writes notebooks/YOLO11m_VisDrone_train_and_cache.ipynb (run: python notebooks/make_yolo11m_visdrone_notebook.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
split = json.loads((ROOT / "research/acmot_paper_v2/DETECTOR_SPLIT.json").read_text())

cells = []


def md(s):
    cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))


def code(s):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))


md("""
# YOLO11m on VisDrone2019-MOT — training and detection caches (Colab T4)

Detector for the OATrack comparison (Sensors 26(16):5222 used YOLO11m trained on VisDrone-MOT train, 1536 px input).

What this notebook does:
1. converts VisDrone2019-MOT-train to YOLO format (5 evaluated classes: pedestrian, car, van, truck, bus);
2. trains YOLO11m at 1536 px on the **detector-train** sequences (32); 8 other train sequences are kept out
   for AC-MOT calibration and serve as the training monitor; the 16 protected confirmation sequences are not used;
3. writes detection caches (score floor 0.01) for VisDrone2019-MOT-val, the 8 calibration sequences and UAVDT test,
   in the repository's cache format, plus SHA-256 manifests.

No tracking metric is computed here. Everything is saved to Google Drive, and training resumes after a disconnect
(run all cells again).

Runtime → Change runtime type → T4 GPU.
""")
code("!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv\n!pip -q install ultralytics==8.3.200")
code("from google.colab import drive\ndrive.mount('/content/drive')")
md("## Paths (edit)")
code("""
# VisDrone2019-MOT-train: folder containing sequences/ and annotations/, or the zip
TRAIN = '/content/drive/MyDrive/PATH/TO/VisDrone2019-MOT-train'
TRAIN_ZIP = ''
# VisDrone2019-MOT-val (folder or zip) — used only for caching detections
VAL = '/content/drive/MyDrive/PATH/TO/VisDrone2019-MOT-val'
VAL_ZIP = ''
# UAVDT test in VisDrone layout (sequences/<seq>/*.jpg) — caching only; leave '' to skip
UAVDT = ''
OUT = '/content/drive/MyDrive/acmot_oatrack'          # everything persistent goes here

IMGSZ = 1536
EPOCHS = 40
BATCH = 4            # T4 16 GB at 1536 px
FRAME_STRIDE = 5     # every 5th frame of each training sequence
SEED = 0
CACHE_RES = [1088, 1280, 1536]   # resolutions cached for AC-MOT
""")
code("""
%%bash
# helper: locate the datasets on Drive
find /content/drive -maxdepth 6 \\( -iname 'VisDrone2019-MOT-*' -o -iname '*UAVDT*' \\) 2>/dev/null | head -20
""")
md("## Split (fixed before training)")
code(f"""
CALIBRATION = {json.dumps(split['calibration'])}
DETECTOR_TRAIN = {json.dumps(split['detector_train'])}
EXCLUDED_PROTECTED = {json.dumps(split['excluded_confirmation'])}
assert not set(CALIBRATION) & set(DETECTOR_TRAIN)
assert not (set(CALIBRATION) | set(DETECTOR_TRAIN)) & set(EXCLUDED_PROTECTED)
print(len(DETECTOR_TRAIN), 'training sequences,', len(CALIBRATION), 'calibration sequences')
""")
md("## Data to local disk")
code("""
import os, zipfile, shutil, glob
def local(path, zpath, name):
    dst = f'/content/data/{name}'
    if os.path.isdir(f'{dst}/sequences'):
        return dst
    os.makedirs('/content/data', exist_ok=True)
    if zpath:
        with zipfile.ZipFile(zpath) as z:
            z.extractall('/content/data/_x')
        src = glob.glob('/content/data/_x/**/sequences', recursive=True)[0][:-len('/sequences')]
        shutil.move(src, dst)
        shutil.rmtree('/content/data/_x', ignore_errors=True)
    else:
        shutil.copytree(path, dst)
    return dst
TR = local(TRAIN, TRAIN_ZIP, 'train')
seqs = sorted(os.listdir(f'{TR}/sequences'))
missing = [s for s in DETECTOR_TRAIN + CALIBRATION if s not in seqs]
assert not missing, missing
print(len(seqs), 'train sequences found')
""")
md("## Convert to YOLO format")
code("""
import numpy as np
from PIL import Image
CLS = {1: 0, 4: 1, 5: 2, 6: 3, 9: 4}            # pedestrian, car, van, truck, bus
NAMES = ['pedestrian', 'car', 'van', 'truck', 'bus']
Y = '/content/yolo'
def convert(seq_list, part, stride):
    n = 0
    for s in seq_list:
        a = np.loadtxt(f'{TR}/annotations/{s}.txt', delimiter=',', ndmin=2)
        frames = sorted(glob.glob(f'{TR}/sequences/{s}/*.jpg'))
        W, H = Image.open(frames[0]).size
        for k, f in enumerate(frames):
            if k % stride:
                continue
            t = k + 1
            img = f'{Y}/images/{part}/{s}_{t:07d}.jpg'
            if not os.path.exists(img):
                os.makedirs(os.path.dirname(img), exist_ok=True)
                os.symlink(f, img)
            rows = a[(a[:, 0] == t) & (a[:, 6] == 1) & np.isin(a[:, 7], list(CLS))]
            os.makedirs(f'{Y}/labels/{part}', exist_ok=True)
            with open(f'{Y}/labels/{part}/{s}_{t:07d}.txt', 'w') as fh:
                for r in rows:
                    x, y, w, h = r[2:6]
                    x, y = max(x, 0), max(y, 0)
                    w, h = min(w, W - x), min(h, H - y)
                    if w > 1 and h > 1:
                        fh.write(f'{CLS[int(r[7])]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\\n')
            n += 1
    return n
print('train images', convert(DETECTOR_TRAIN, 'train', FRAME_STRIDE))
print('monitor images', convert(CALIBRATION, 'val', 10))
open(f'{Y}/data.yaml', 'w').write(f'path: {Y}\\ntrain: images/train\\nval: images/val\\nnames: {NAMES}\\n')
""")
md("## Train (resumes from Drive after a disconnect)")
code("""
from ultralytics import YOLO
run_dir = f'{OUT}/train/yolo11m_vd1536'
last = f'{run_dir}/weights/last.pt'
if os.path.exists(last):
    model = YOLO(last)
    model.train(resume=True)
else:
    model = YOLO('yolo11m.pt')
    model.train(data=f'{Y}/data.yaml', imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, seed=SEED, deterministic=True,
                project=f'{OUT}/train', name='yolo11m_vd1536', exist_ok=True, workers=2, save_period=1,
                close_mosaic=10, plots=True)
""")
code("""
import hashlib, json
W8 = f'{OUT}/train/yolo11m_vd1536/weights/last.pt'   # last epoch, not 'best' (no selection on any split)
sha = hashlib.sha256(open(W8, 'rb').read()).hexdigest()
json.dump(dict(weights=W8, sha256=sha, imgsz=IMGSZ, epochs=EPOCHS, frame_stride=FRAME_STRIDE, seed=SEED,
               detector_train=DETECTOR_TRAIN, monitor=CALIBRATION), open(f'{OUT}/DETECTOR_MANIFEST.json', 'w'), indent=1)
print(sha)
""")
md("## Detection caches (score floor 0.01; no metric)")
code("""
import time, torch
VA = local(VAL, VAL_ZIP, 'val') if (VAL or VAL_ZIP) else None
det_model = YOLO(W8)
def cache(root, seq_list, tag):
    for res in CACHE_RES:
        d = f'{OUT}/cache/yolo11m_vd/{tag}/{res}'
        os.makedirs(d, exist_ok=True)
        for s in seq_list:
            if os.path.exists(f'{d}/{s}.npz'):
                continue
            frames = sorted(glob.glob(f'{root}/sequences/{s}/*.jpg'))
            rows, times = [], []
            for t, f in enumerate(frames, 1):
                t0 = time.perf_counter()
                r = det_model.predict(f, imgsz=res, conf=0.01, iou=0.7, max_det=1000, half=True, verbose=False)[0]
                (torch.cuda.synchronize() if torch.cuda.is_available() else None); times.append(time.perf_counter() - t0)
                b = r.boxes
                for xyxy, sc, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
                    rows.append([t, *xyxy.tolist(), float(sc), int(c)])
            shape = r.orig_shape
            np.savez_compressed(f'{d}/{s}.npz', det=np.asarray(rows, np.float64).reshape(-1, 7),
                                shape=np.asarray(shape), frames=len(frames))
            json.dump(dict(mean_ms=1000 * float(np.mean(times[10:] or times)), frames=len(frames),
                           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'), open(f'{d}/{s}.timing.json', 'w'))
            print(tag, res, s, len(rows), 'boxes')
if VA:
    cache(VA, sorted(os.listdir(f'{VA}/sequences')), 'visdrone_val')
cache(TR, CALIBRATION, 'visdrone_calib')
if UAVDT:
    cache(UAVDT, sorted(os.listdir(f'{UAVDT}/sequences')), 'uavdt_test')
""")
code("""
%%bash -s "$OUT"
cd "$1/cache" && tar -cf ../yolo11m_vd_cache.tar yolo11m_vd && cd .. && sha256sum yolo11m_vd_cache.tar | tee yolo11m_vd_cache.tar.sha256
ls -la "$1"
""")
md("""
Send back: `DETECTOR_MANIFEST.json`, `yolo11m_vd_cache.tar` and its `.sha256` (Drive folder `acmot_oatrack/`).
Class ids in the cache: 0 pedestrian, 1 car, 2 van, 3 truck, 4 bus.
""")

nb = dict(cells=cells, metadata=dict(accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
                                     kernelspec=dict(name="python3", display_name="Python 3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=0)
(ROOT / "notebooks/YOLO11m_VisDrone_train_and_cache.ipynb").write_text(json.dumps(nb, indent=1))
