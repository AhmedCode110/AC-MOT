"""
Kaggle versions of notebooks/YOLO11m_VisDrone_train_and_cache.ipynb.

Only the environment changes (paths, persistence across Kaggle sessions, a
GPU assertion, a session-time guard and a smoke mode); every scientific cell
(split, conversion, training arguments, manifest, multi-NMS caching) is taken
verbatim from the Colab notebook, and the script asserts that.

  python notebooks/kaggle/make_kaggle_kernels.py <kaggle-username> <train> <val> [<uavdt>] [--state <dataset>]

<train>, <val>, <uavdt>: dataset roots under /kaggle/input (folder containing
sequences/ and, for VisDrone, annotations/). Writes two kernel folders:
  notebooks/kaggle/smoke/  (SMOKE = True)   and   notebooks/kaggle/train/  (SMOKE = False)
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "YOLO11m_VisDrone_train_and_cache.ipynb"
SLUG = {"smoke": "acmot-oatrack-yolo11m-smoke", "train": "acmot-oatrack-yolo11m-train"}


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def code(s):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n"))


def md(s):
    return dict(cell_type="markdown", metadata={}, source=s.strip("\n"))


ENV = r"""
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
!pip -q install ultralytics==8.3.200
import sys, os, json, time, glob, shutil, torch, ultralytics
SESSION_START = time.time()
SESSION_BUDGET_H = 11.0          # Kaggle GPU sessions end at 12 h; stop training after a full epoch before that
ENV = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
           cudnn=torch.backends.cudnn.version(), cuda_available=torch.cuda.is_available(),
           gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           ultralytics=ultralytics.__version__)
print(json.dumps(ENV, indent=1))
assert torch.cuda.is_available(), 'No GPU: this run must not continue on CPU.'
"""

STATE = r"""
SMOKE = {smoke}
OUT = '/kaggle/working/acmot_oatrack'
# state of the previous session (private dataset '{state}' = earlier /kaggle/working/acmot_oatrack)
prev = sorted(glob.glob('/kaggle/input/{state_dir}/**/acmot_oatrack', recursive=True))
if prev and not os.path.exists(OUT):
    shutil.copytree(prev[0], OUT)
    print('restored state from', prev[0])
os.makedirs(OUT, exist_ok=True)
print(sorted(os.listdir(OUT)))
"""

PATHS = r"""
TRAIN = '{train}'
TRAIN_ZIP = ''
VAL = '{val}'
VAL_ZIP = ''
UAVDT = '{uavdt}'
"""

FIND = r"""
%%bash
ls /kaggle/input
find /kaggle/input -maxdepth 4 -type d -name sequences 2>/dev/null | head -20
"""

SMOKE_CELL = r"""
if SMOKE:
    # data and protocol checks
    assert len(TRAIN_SEQS) == 56, len(TRAIN_SEQS)
    print('train frames', n_frames, 'relative to 24,201:', round(n_frames / 24201, 4))
    assert abs(n_frames - 24201) / 24201 < 0.02, n_frames
    assert IMGSZ == 1536 and FRAME_STRIDE == 1 and EPOCHS == 40, (IMGSZ, FRAME_STRIDE, EPOCHS)
    open(f'{OUT}/_write_test', 'w').write('ok'); os.remove(f'{OUT}/_write_test')
    # resume logic: 2-epoch run on a small fraction, stopped after epoch 1 (as the session guard does),
    # restored from the full checkpoint and resumed to the end
    from ultralytics import YOLO
    sd = f'{OUT}/smoke/run'; full_s = f'{sd}/weights/resume_full.pt'
    def keep_full_s(tr): shutil.copy2(tr.last, f'{tr.save_dir}/weights/resume_full.pt')
    def stop_after_first(tr):
        if tr.epoch == 0 and not os.path.exists(f'{tr.save_dir}/_resumed'): tr.stop = True
    torch.cuda.reset_peak_memory_stats()
    m = YOLO('yolo11m.pt'); m.add_callback('on_model_save', keep_full_s); m.add_callback('on_fit_epoch_end', stop_after_first)
    m.train(data=f'{Y}/data.yaml', imgsz=IMGSZ, epochs=2, batch=BATCH, seed=SEED, deterministic=True, fraction=0.004,
            project=f'{OUT}/smoke', name='run', exist_ok=True, workers=2, close_mosaic=0, plots=False)
    print('peak GPU memory (GB)', round(torch.cuda.max_memory_reserved() / 2**30, 2), 'at batch', BATCH, 'imgsz', IMGSZ)
    shutil.copy2(full_s, f'{sd}/weights/last.pt'); open(f'{sd}/_resumed', 'w').write('1')
    m = YOLO(f'{sd}/weights/last.pt'); m.add_callback('on_model_save', keep_full_s)
    m.train(resume=True)
    rows = open(f'{sd}/results.csv').read().strip().splitlines()[1:]
    print('epochs in results.csv after resume:', len(rows))
    assert len(rows) == 2, rows
    print('SMOKE CHECKS PASSED')
"""

TRAIN = r"""
from ultralytics import YOLO
run_dir = f'{OUT}/train/yolo11m_vd1536'
last = f'{run_dir}/weights/last.pt'
full = f'{run_dir}/weights/resume_full.pt'      # full checkpoint (optimizer kept) after every epoch
DONE = f'{OUT}/TRAINING_DONE'

def keep_full(trainer):
    shutil.copy2(trainer.last, full)

def session_guard(trainer):
    # stop after this epoch if the next one would not finish inside the session; no other effect on training
    done = trainer.epoch + 1 - trainer.start_epoch
    per_epoch = (time.time() - trainer.train_time_start) / max(done, 1)
    if trainer.epoch + 1 < trainer.epochs and time.time() - SESSION_START + 1.3 * per_epoch > SESSION_BUDGET_H * 3600:
        print(f'session guard: stopping after epoch {trainer.epoch + 1}; resume in the next session')
        trainer.stop = True

def epochs_done():
    f = f'{run_dir}/results.csv'
    return len(open(f).read().strip().splitlines()) - 1 if os.path.exists(f) else 0

if not SMOKE and not os.path.exists(DONE):
    if os.path.exists(full):
        shutil.copy2(full, last)                   # an early stop strips the optimizer from last.pt
        model = YOLO(last)
        model.add_callback('on_model_save', keep_full); model.add_callback('on_fit_epoch_end', session_guard)
        model.train(resume=True)
    else:
        model = YOLO('yolo11m.pt')
        model.add_callback('on_model_save', keep_full); model.add_callback('on_fit_epoch_end', session_guard)
        model.train(data=f'{Y}/data.yaml', imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, seed=SEED, deterministic=True,
                    project=f'{OUT}/train', name='yolo11m_vd1536', exist_ok=True, workers=2, save_period=1,
                    close_mosaic=10, plots=True)
    print('epochs completed:', epochs_done(), 'of', EPOCHS)
    if epochs_done() == EPOCHS:
        open(DONE, 'w').write(json.dumps(dict(epochs=EPOCHS, env=ENV)))
TRAINING_DONE = os.path.exists(DONE)
print('TRAINING_DONE', TRAINING_DONE)
"""


def build(mode, args):
    nb = json.loads(SRC.read_text())
    cells = nb["cells"]
    s = [src(c) for c in cells]
    # environment cells being replaced (asserted, so a change in the Colab notebook is noticed)
    assert s[1].startswith("!nvidia-smi") and "google.colab" in s[2] and s[5].startswith("%%bash")
    assert s[13].startswith("from ultralytics import YOLO") and "model.train(data=" in s[13]
    out = copy.deepcopy(cells)
    out[0] = md(s[0].split("No tracking metric")[0] + "No tracking metric is computed here.\n\n"
                "**Kaggle version**: state is kept in `/kaggle/working/acmot_oatrack` and carried to the next session "
                "through a private dataset; a session guard stops training after a complete epoch before Kaggle's "
                "12-hour limit, and the next session resumes from the full checkpoint. Training arguments, data, "
                "split, class set and caching are identical to the Colab notebook.")
    out[1] = code(ENV)
    out[2] = code(STATE.format(smoke=mode == "smoke", state=args.state or "", state_dir=args.state or "_none_"))
    params = s[4].split("OUT = ")[1].split("\n", 1)[1]           # IMGSZ ... NMS_IOU, verbatim
    out[4] = code(PATHS.format(train=args.train, val=args.val, uavdt=args.uavdt or "") + params)
    out[5] = code(FIND)
    out[9] = code(s[9].replace("/content/data", "/tmp/data").replace("        shutil.copytree(path, dst)\n    return dst",
                                                                     "        return path\n    return dst"))
    assert "return path" in src(out[9])
    out[11] = code(s[11].replace("/content/yolo", "/tmp/yolo"))
    out[13] = code(TRAIN)
    out[14] = code("if not SMOKE and TRAINING_DONE:\n" + "\n".join("    " + l for l in s[14].splitlines())
                   .replace("open(f'{OUT}/DETECTOR_MANIFEST.json', 'w')",
                            "open(f'{OUT}/DETECTOR_MANIFEST.json', 'w')")
                   .replace("status='paper-aligned", "env=ENV, status='paper-aligned"))
    out[16] = code("if not SMOKE and TRAINING_DONE:\n" + "\n".join("    " + l for l in s[16].splitlines()))
    out[17] = code("%%bash -s \"$OUT\"\nif [ -f \"$1/TRAINING_DONE\" ] && [ -d \"$1/cache\" ]; then\n"
                   "cd \"$1/cache\" && tar -cf ../yolo11m_vd_cache.tar yolo11m_vd && cd .. && "
                   "sha256sum yolo11m_vd_cache.tar | tee yolo11m_vd_cache.tar.sha256\nfi\nls -la \"$1\"")
    # scientific cells copied verbatim
    for i in (6, 7, 10, 15):
        assert out[i] == cells[i]
    out.insert(12, code(SMOKE_CELL))
    for key in ("model.train(data=f'{Y}/data.yaml', imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, seed=SEED, deterministic=True",
                "project=f'{OUT}/train', name='yolo11m_vd1536', exist_ok=True, workers=2, save_period=1",
                "close_mosaic=10, plots=True"):
        assert key in s[13] and key in TRAIN, key
    nb["cells"] = out
    nb["metadata"]["kaggle"] = dict(accelerator="nvidiaTeslaT4", isGpuEnabled=True, isInternetEnabled=True)
    d = HERE / mode
    d.mkdir(exist_ok=True)
    name = f"{SLUG[mode]}.ipynb"
    (d / name).write_text(json.dumps(nb, indent=1))
    meta = dict(id=f"{args.user}/{SLUG[mode]}", title=SLUG[mode].replace("-", " "), code_file=name,
                language="python", kernel_type="notebook", is_private=True, enable_gpu=True, enable_tpu=False,
                enable_internet=True, machine_shape="NvidiaTeslaT4",
                dataset_sources=[x for x in args.datasets] + ([args.state] if args.state else []),
                competition_sources=[], kernel_sources=[], model_sources=[])
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("wrote", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("train")
    ap.add_argument("val")
    ap.add_argument("uavdt", nargs="?", default="")
    ap.add_argument("--datasets", nargs="*", default=[], help="Kaggle dataset refs owner/slug to attach")
    ap.add_argument("--state", default="", help="private dataset ref holding the previous session's acmot_oatrack/")
    a = ap.parse_args()
    for m in ("smoke", "train"):
        build(m, a)
