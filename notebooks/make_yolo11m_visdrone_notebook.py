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
# YOLO11m on VisDrone2019-MOT — paper-aligned detector and detection caches

Detector for the OATrack comparison. OATrack (Sensors 26(16):5222) used "YOLO11m-smallobj" trained on all of
VisDrone-MOT train (56 sequences, 24,201 frames) at 1536 px. Its epochs, augmentation, class mapping and the
"smallobj" data configuration are not published, so this is a **paper-aligned re-implementation**, not an exact
detector reproduction.

What this notebook does:
1. converts all 56 VisDrone2019-MOT-train sequences, every frame, to YOLO format (class set: `CLASS_SET`);
2. trains YOLO11m at 1536 px on all of them and keeps the last epoch (no checkpoint selection);
3. writes detection caches for VisDrone2019-MOT-val (final benchmark), 8 train sequences (AC-MOT calibration
   only) and UAVDT test (transfer), at each resolution in `CACHE_RES` and each NMS IoU in `NMS_IOU`: one
   forward pass per frame and resolution, then the Ultralytics postprocess once per NMS value (identical to
   `predict(iou=...)`), score floor 0.01, so AC-MOT can change resolution, confidence and NMS exactly.

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
EPOCHS = 40          # not stated by the paper (choice recorded in the manifest)
BATCH = 4            # T4 16 GB at 1536 px
FRAME_STRIDE = 1     # every frame (paper: the converted training split)
SEED = 0
CLASS_SET = 'eval5'  # 'eval5': pedestrian, car, van, truck, bus | 'all10': the ten VisDrone categories
CACHE_RES = [1088, 1280, 1536]   # detector input sizes AC-MOT may choose
NMS_IOU = [0.45, 0.60, 0.70]     # NMS operating points (0.70 = Ultralytics default)
""")
code("""
%%bash
# helper: locate the datasets on Drive
find /content/drive -maxdepth 6 \\( -iname 'VisDrone2019-MOT-*' -o -iname '*UAVDT*' \\) 2>/dev/null | head -20
""")
md("""## Split (fixed before training)
All 56 sequences train the detector. The 8 calibration sequences (from the development-40 part of the train split)
are used later only to calibrate AC-MOT; VisDrone val is never used for training or calibration.""")
code(f"""
CALIBRATION = {json.dumps(split['calibration'])}
DETECTOR_TRAIN = {json.dumps(split['detector_train'])}
EXCLUDED_PROTECTED = {json.dumps(split['excluded_confirmation'])}
assert not set(CALIBRATION) & set(EXCLUDED_PROTECTED)
print(len(CALIBRATION), 'calibration sequences')
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
assert len(seqs) == 56, len(seqs)
missing = [s for s in CALIBRATION if s not in seqs]
assert not missing, missing
n_frames = sum(len(glob.glob(f'{TR}/sequences/{s}/*.jpg')) for s in seqs)
print(len(seqs), 'train sequences,', n_frames, 'frames (paper: 56 / 24,201)')
TRAIN_SEQS = seqs
""")
md("## Convert to YOLO format")
code("""
import numpy as np
from PIL import Image
if CLASS_SET == 'eval5':
    CLS = {1: 0, 4: 1, 5: 2, 6: 3, 9: 4}
    NAMES = ['pedestrian', 'car', 'van', 'truck', 'bus']
else:
    CLS = {k: k - 1 for k in range(1, 11)}
    NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
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
print('train images', convert(TRAIN_SEQS, 'train', FRAME_STRIDE))
print('monitor images', convert(CALIBRATION, 'val', 10))   # inside the training set: progress monitor only
open(f'{Y}/data.yaml', 'w').write(f'path: {Y}\\ntrain: images/train\\nval: images/val\\nnames: {NAMES}\\n')
""")
md("""## Train (resumes from Drive after a disconnect)
Time: one epoch over all 24,201 frames at 1536 px takes several hours on a T4 (Ultralytics prints the epoch time
after the first epoch); 40 epochs need many Colab sessions. Re-running all cells continues from the last saved
epoch. A faster GPU (A100/L4) or Kaggle (two T4) shortens this; the recipe stays the same.""")
code("""
from ultralytics import YOLO
run_dir = f'{OUT}/train/yolo11m_vd1536'
last = f'{run_dir}/weights/last.pt'
if os.path.exists(last):
    try:
        YOLO(last).train(resume=True)
    except AssertionError as e:          # training already finished
        print(e)
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
json.dump(dict(weights=W8, sha256=sha, model='yolo11m.pt', imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH,
               frame_stride=FRAME_STRIDE, seed=SEED, class_set=CLASS_SET, class_map=CLS, names=NAMES,
               train_sequences=TRAIN_SEQS, train_frames=n_frames, calibration=CALIBRATION,
               status='paper-aligned re-implementation (epochs, augmentation, class mapping and the smallobj '
                      'configuration of OATrack are not published)'),
          open(f'{OUT}/DETECTOR_MANIFEST.json', 'w'), indent=1)
print(sha)
""")
md("""## Detection caches (score floor 0.01; no metric)
Layout: `cache/yolo11m_vd/<split>/r<res>_n<nms>/<seq>.npz` with `det` rows `[frame, x1, y1, x2, y2, score, class]`.""")
code("""
import time, torch, cv2
VA = local(VAL, VAL_ZIP, 'val') if (VAL or VAL_ZIP) else None
det_model = YOLO(W8)
HALF = torch.cuda.is_available()

def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(v) for v in x)
    return x

def multi_nms_predict(model, path, imgsz, nms_list):
    # one forward pass; the predictor's own postprocess per NMS value (== predict(iou=tau), tested)
    model.predict(path, imgsz=imgsz, conf=0.01, iou=nms_list[0], max_det=1000, half=HALF, verbose=False)
    p = model.predictor
    im0 = cv2.imread(path)
    with torch.inference_mode():
        im = p.preprocess([im0])
        raw = p.inference(im)
        out = {}
        for tau in nms_list:
            p.args.iou = tau
            b = p.postprocess(_clone(raw), im, [im0])[0].boxes
            out[tau] = torch.cat([b.xyxy, b.conf[:, None], b.cls[:, None]], 1).cpu().numpy()
    return out, im0.shape[:2]

def cache(root, seq_list, tag):
    for res in CACHE_RES:
        dirs = {t: f'{OUT}/cache/yolo11m_vd/{tag}/r{res}_n{int(round(100 * t))}' for t in NMS_IOU}
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        for s in seq_list:
            if all(os.path.exists(f'{d}/{s}.npz') for d in dirs.values()):
                continue
            frames = sorted(glob.glob(f'{root}/sequences/{s}/*.jpg'))
            rows = {t: [] for t in NMS_IOU}
            times = []
            for t_, f in enumerate(frames, 1):
                t0 = time.perf_counter()
                out, shape = multi_nms_predict(det_model, f, res, NMS_IOU)
                (torch.cuda.synchronize() if torch.cuda.is_available() else None); times.append(time.perf_counter() - t0)
                for tau, a in out.items():
                    rows[tau] += [[t_, *r[:4].tolist(), float(r[4]), int(r[5])] for r in a]
            for tau, d in dirs.items():
                np.savez_compressed(f'{d}/{s}.npz', det=np.asarray(rows[tau], np.float64).reshape(-1, 7),
                                    shape=np.asarray(shape), frames=len(frames))
            json.dump(dict(mean_ms_all_nms=1000 * float(np.mean(times[10:] or times)), frames=len(frames),
                           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'),
                      open(f'{dirs[NMS_IOU[-1]]}/{s}.timing.json', 'w'))
            print(tag, res, s, {t: len(v) for t, v in rows.items()})
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
Class ids in the cache follow `NAMES` of the chosen `CLASS_SET` (recorded in the manifest).
""")

nb = dict(cells=cells, metadata=dict(accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
                                     kernelspec=dict(name="python3", display_name="Python 3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=0)
(ROOT / "notebooks/YOLO11m_VisDrone_train_and_cache.ipynb").write_text(json.dumps(nb, indent=1))
