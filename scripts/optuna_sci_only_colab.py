"""AC-MOT SCI-only Optuna tuning for Google Colab.

Scientific protocol:
- fixed pretrained YOLOv8n detector (no detector training)
- fixed ByteTrack settings
- tune only the five SCI weights on VisDrone2019-MOT-val
- maximize MOTA, minimize IDS, maximize FPS
- require FPS >= 25 and IDS <= old A3 on validation
- freeze weights, then evaluate once on VisDrone2019-MOT-test-dev

The metric adapter remains the repository's existing custom class-agnostic TrackEval
protocol so results stay comparable with prior AC-MOT runs. Do not call these full
official 5-class VisDrone leaderboard scores because fixed COCO YOLOv8n has no
independent van class.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


def sh(args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


# ----------------------------- configuration -----------------------------
DATA_ROOT = Path('/content/drive/MyDrive/AC-MOT-data')
RESULT_ROOT = Path('/content/drive/MyDrive/AC-MOT-results/optuna_sci_only')
VAL_DIR = DATA_ROOT / 'VisDrone2019-MOT-val'
TEST_DIR = DATA_ROOT / 'VisDrone2019-MOT-test-dev'
VAL_GDRIVE_ID = '1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu'
TEST_GDRIVE_ID = '14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu'
N_TRIALS = int(os.environ.get('ACMOT_OPTUNA_TRIALS', '30'))
SEED = 42
MIN_FPS = 25.0
PROGRESS_EVERY = 250
ENFORCE_IDS_NOT_WORSE_THAN_OLD_A3 = True
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------ Colab setup -------------------------------
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
DATA_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

sh([
    sys.executable, '-m', 'pip', 'install', '-q',
    'ultralytics==8.3.200', 'numpy==2.2.6', 'scipy==1.15.3',
    'lap', 'opencv-python-headless', 'optuna>=4,<5', 'gdown', 'pandas'
])

TRACKEVAL_DIR = Path('/content/TrackEval')
if TRACKEVAL_DIR.exists():
    shutil.rmtree(TRACKEVAL_DIR)
sh(['git', 'clone', '-q', 'https://github.com/JonathonLuiten/TrackEval.git', str(TRACKEVAL_DIR)])
sh(['git', '-C', str(TRACKEVAL_DIR), 'checkout', '-q', '12c8791b303e0a0b50f753af204249e622d0281a'])

WEIGHTS_DIR = Path('/content/weights')
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS = WEIGHTS_DIR / 'yolov8n.pt'
if not WEIGHTS.exists():
    from ultralytics import YOLO
    old = os.getcwd()
    os.chdir(WEIGHTS_DIR)
    try:
        YOLO('yolov8n.pt')
    finally:
        os.chdir(old)
if not WEIGHTS.exists():
    raise RuntimeError('Could not obtain fixed yolov8n.pt')
print('[OK] Fixed detector:', WEIGHTS)
print('[OK] No detector training will be performed')


# ------------------------------ data setup --------------------------------
import gdown
import zipfile


def valid_dataset(path: Path):
    return path.is_dir() and (path / 'sequences').is_dir() and (path / 'annotations').is_dir()


def ensure_dataset(target: Path, file_id: str):
    if valid_dataset(target):
        print('[EXISTS]', target)
        return
    archive = DATA_ROOT / f'{target.name}.zip'
    if not archive.exists():
        print('[DOWNLOAD]', target.name)
        out = gdown.download(id=file_id, output=str(archive), quiet=False)
        if not out or not archive.exists():
            raise RuntimeError(f'Download failed. Put the official archive at {archive}')
    with zipfile.ZipFile(archive, 'r') as z:
        z.extractall(DATA_ROOT)
    if not valid_dataset(target):
        candidates = [p for p in DATA_ROOT.iterdir() if p.is_dir()
                      and (p / 'sequences').is_dir() and (p / 'annotations').is_dir()
                      and target.name.lower() in p.name.lower()]
        if len(candidates) == 1:
            candidates[0].rename(target)
    if not valid_dataset(target):
        raise RuntimeError(f'Dataset structure not found: {target}')


ensure_dataset(VAL_DIR, VAL_GDRIVE_ID)
ensure_dataset(TEST_DIR, TEST_GDRIVE_ID)


# ---------------------------- AC-MOT imports ------------------------------
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import boxes
from core_v17 import PresentationController as OldPresentationController, PresentationSpec
from experiment import dataset_manifest
import scripts.paper_eval_v17 as pe

if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU required. Select T4 in Colab.')
print('[GPU]', torch.cuda.get_device_name(0))
if 'T4' not in torch.cuda.get_device_name(0):
    print('[WARNING] FPS is not directly comparable with prior T4 runs.')

val_names = sorted(p.name for p in (VAL_DIR / 'sequences').iterdir() if p.is_dir())
test_names = sorted(p.name for p in (TEST_DIR / 'sequences').iterdir() if p.is_dir())
VAL_MANIFEST = dataset_manifest(VAL_DIR, val_names)
TEST_MANIFEST = dataset_manifest(TEST_DIR, test_names)
print('[OK] Validation sequences:', len(VAL_MANIFEST))
print('[OK] Test sequences:', len(TEST_MANIFEST))


# ----------------------------- tunable SCI --------------------------------
class TunableSCIController:
    def __init__(self, spec: PresentationSpec, weights: dict):
        self.spec = spec.validate()
        self.history = deque(maxlen=spec.smoothing_window)
        self.sci = 0.0
        self.tiny = 0.0
        self.scene = 'clear'
        keys = ['crowd', 'tiny', 'edge', 'night', 'blur']
        vals = np.asarray([float(weights[k]) for k in keys], dtype=float)
        if np.any(vals < 0) or vals.sum() <= 0:
            raise ValueError('Invalid SCI weights')
        vals /= vals.sum()
        self.weights = dict(zip(keys, vals.tolist()))

    def choose(self, frame, visual, previous):
        s = self.spec
        analyze = frame == 1 or (frame - 1) % s.analysis_stride == 0
        if analyze:
            previous = boxes(previous)
            n = len(previous)
            self.tiny = float(np.mean((previous[:,2]-previous[:,0])*(previous[:,3]-previous[:,1]) < 32*32)) if n else 0.0
            crowd = min(n / 30.0, 1.0)
            edge = float(visual['edges'])
            brightness = float(visual['brightness'])
            blur_value = float(visual['blur'])
            cues = dict(
                crowd=crowd,
                tiny=self.tiny,
                edge=min(edge / 0.14, 1.0),
                night=1.0 if brightness < 80 else 0.0,
                blur=1.0 if blur_value < 180 else 0.0,
            )
            raw = sum(self.weights[k] * cues[k] for k in self.weights)
            self.history.append(float(np.clip(raw, 0.0, 1.0)))
            self.sci = float(np.mean(self.history))
            if brightness < 80: self.scene = 'night'
            elif blur_value < 180: self.scene = 'blur'
            elif self.tiny > 0.50: self.scene = 'tiny'
            elif crowd > 0.65 or edge > 0.13: self.scene = 'crowded'
            else: self.scene = 'clear'

        conf, nms, size = 0.25, 0.45, 640
        if s.adaptive_threshold:
            conf = 0.245 - 0.050 * self.sci
            nms = 0.490 - 0.050 * self.sci
            if self.scene in {'crowded', 'tiny', 'night'}: conf -= 0.012
            if self.scene == 'blur': nms -= 0.012
            conf = float(np.clip(conf, 0.19, 0.28))
            nms = float(np.clip(nms, 0.40, 0.52))
        if s.adaptive_resolution:
            if self.sci > 0.60 or self.tiny > 0.50: size = 832
            elif self.sci > 0.35 or self.scene in {'crowded', 'tiny'}: size = 736
        return dict(conf=conf, nms=nms, size=size, sci=float(self.sci), scene=self.scene)


def trial_weights(trial):
    raw = {k: trial.suggest_float('raw_' + k, 0.02, 1.0)
           for k in ['crowd', 'tiny', 'edge', 'night', 'blur']}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def params_weights(params):
    raw = {k: params['raw_' + k] for k in ['crowd', 'tiny', 'edge', 'night', 'blur']}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# ------------------------------ run helpers -------------------------------
GT_FILTER = dict(categories=[1,4,5,6,9], score=1, occlusion_lt=2, truncation_lt=2)
A3 = dict(tracker_profile='tuned', adaptive_threshold=True, adaptive_resolution=True,
          smoothing_window=7, analysis_stride=10)
A0 = dict(tracker_profile='default', adaptive_threshold=False, adaptive_resolution=False,
          smoothing_window=7, analysis_stride=10)


def write_meta(root, manifest, systems):
    (root / 'configuration.json').write_text(json.dumps(dict(
        version='optuna_sci_only', systems=systems, ground_truth_filter=GT_FILTER
    ), indent=2))
    (root / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2))


def parse_metrics(path, name):
    row = next(r for r in csv.DictReader(path.open()) if r['system'] == name)
    return dict(HOTA=float(row['HOTA']), DetA=float(row['DetA']), AssA=float(row['AssA']),
                MOTA=float(row['MOTA']), IDF1=float(row['IDF1']), IDS=int(float(row['IDS'])),
                FN=int(float(row['FN'])), FP=int(float(row['FP'])))


def run_one(dataset, manifest, system, root, controller_factory, keep=False):
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    write_meta(root, manifest, [system])
    original = pe.PresentationController
    try:
        pe.PresentationController = controller_factory
        timing = pe.run_system(system, dataset, manifest, WEIGHTS,
                               Path('/content/weights/unused.engine'), MIN_FPS,
                               PROGRESS_EVERY, 'pytorch', 16, root, 1, 1)
    finally:
        pe.PresentationController = original
    eval_out = root / 'trackeval'
    sh([sys.executable, str(ROOT / 'evaluate.py'), str(root), '--dataset', str(dataset),
        '--trackeval', str(TRACKEVAL_DIR), '--output', str(eval_out)], cwd=ROOT)
    result = parse_metrics(eval_out / 'summary.csv', system['name'])
    result['FPS'] = float(timing['processing_fps'])
    result['mean_imgsz'] = float(timing['mean_imgsz'])
    result['mean_conf'] = float(timing['mean_conf'])
    result['mean_nms_iou'] = float(timing['mean_nms_iou'])
    if not keep:
        (root / 'compact_result.json').write_text(json.dumps(result, indent=2))
        for p in list(root.iterdir()):
            if p.name not in {'compact_result.json', 'trackeval'}:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    gc.collect(); torch.cuda.empty_cache()
    return result


# --------------------------- old A3 validation ----------------------------
old_system = dict(name='A3_OLD_SCI', **A3)
old_val = run_one(VAL_DIR, VAL_MANIFEST, old_system,
                  Path('/content/old_a3_validation'), OldPresentationController)
BASELINE_IDS = old_val['IDS']
(RESULT_ROOT / 'OLD_A3_VALIDATION.json').write_text(json.dumps(old_val, indent=2))
print('\n=== OLD A3 VALIDATION ===')
print(json.dumps(old_val, indent=2))


# ------------------------------- Optuna -----------------------------------
TRIAL_ROOT = Path('/content/acmot_optuna_sci_trials')
if TRIAL_ROOT.exists(): shutil.rmtree(TRIAL_ROOT)
TRIAL_ROOT.mkdir(parents=True)


def objective(trial):
    w = trial_weights(trial)
    system = dict(name=f'TRIAL_{trial.number:03d}', **A3)
    root = TRIAL_ROOT / system['name']
    try:
        result = run_one(VAL_DIR, VAL_MANIFEST, system, root,
                         lambda spec: TunableSCIController(spec, w))
    except Exception as exc:
        trial.set_user_attr('error', repr(exc))
        return -1.0, 10**9, 0.0
    trial.set_user_attr('weights', w)
    for k, v in result.items():
        if isinstance(v, (int, float, str, bool)): trial.set_user_attr(k, v)
    print(f"[TRIAL {trial.number:03d}] MOTA={100*result['MOTA']:.3f} IDS={result['IDS']} FPS={result['FPS']:.2f} weights={w}")
    return result['MOTA'], result['IDS'], result['FPS']


study = optuna.create_study(
    directions=['maximize', 'minimize', 'maximize'],
    sampler=TPESampler(seed=SEED, multivariate=True),
    study_name='acmot_sci_only_validation'
)


def save_trials(study, trial):
    study.trials_dataframe(attrs=('number','values','params','user_attrs','state')).to_csv(
        RESULT_ROOT / 'OPTUNA_SCI_VALIDATION_TRIALS.csv', index=False)


study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True,
               show_progress_bar=True, callbacks=[save_trials])

complete = [t for t in study.trials if t.state == TrialState.COMPLETE and t.values is not None and t.values[0] >= 0]
feasible = [t for t in study.best_trials if t.values[2] >= MIN_FPS and
            (not ENFORCE_IDS_NOT_WORSE_THAN_OLD_A3 or int(t.values[1]) <= BASELINE_IDS)]
if not feasible:
    feasible = [t for t in complete if t.values[2] >= MIN_FPS and
                (not ENFORCE_IDS_NOT_WORSE_THAN_OLD_A3 or int(t.values[1]) <= BASELINE_IDS)]
if not feasible:
    raise RuntimeError('No trial satisfies FPS>=25 and IDS<=old A3. Increase trials or redesign SCI.')

best = max(feasible, key=lambda t: (float(t.values[0]), -int(t.values[1]), float(t.values[2])))
BEST_WEIGHTS = params_weights(best.params)
FROZEN = dict(
    protocol='SCI-only tuning with fixed YOLOv8n and fixed ByteTrack',
    trial_number=int(best.number),
    selection_rule='Validation only: FPS>=25 and IDS<=old-A3; then highest MOTA, lowest IDS, highest FPS.',
    weights=BEST_WEIGHTS,
    validation=dict(MOTA=float(best.values[0]), IDS=int(best.values[1]), FPS=float(best.values[2]), old_A3_IDS=int(BASELINE_IDS)),
    detector='fixed yolov8n.pt', tracker='fixed tuned ByteTrack for A3', what_changed='SCI weights only'
)
FROZEN_PATH = RESULT_ROOT / 'FROZEN_OPTUNA_SCI_ONLY.json'
FROZEN_PATH.write_text(json.dumps(FROZEN, indent=2))
print('\n=== FROZEN SCI WEIGHTS ===')
print(json.dumps(FROZEN, indent=2))


# ----------------------------- final test ---------------------------------
TEST_LOCK = RESULT_ROOT / 'FINAL_TEST_DONE.json'
if TEST_LOCK.exists():
    raise RuntimeError(f'Final test already exists: {TEST_LOCK}. Do not repeatedly retune against test.')

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
FINAL_ROOT = RESULT_ROOT / f'FINAL_TEST_{stamp}'
FINAL_ROOT.mkdir(parents=True, exist_ok=False)

systems = [
    (dict(name='A0_DEFAULT_FIXED640', **A0), OldPresentationController),
    (dict(name='A3_OLD_SCI', **A3), OldPresentationController),
    (dict(name='A3_OPTUNA_SCI_FROZEN', **A3), lambda spec: TunableSCIController(spec, BEST_WEIGHTS)),
]
rows = []
for system, factory in systems:
    result = run_one(TEST_DIR, TEST_MANIFEST, system,
                     Path('/content') / (system['name'] + '_final_run'), factory)
    rows.append(dict(System=system['name'], MOTA=100*result['MOTA'], HOTA=100*result['HOTA'],
                     IDF1=100*result['IDF1'], IDS=result['IDS'], FP=result['FP'], FN=result['FN'], FPS=result['FPS']))

final_df = pd.DataFrame(rows)
final_df.to_csv(FINAL_ROOT / 'FINAL_TEST_COMPARISON.csv', index=False)
(FINAL_ROOT / 'FINAL_TEST_COMPARISON.json').write_text(json.dumps(rows, indent=2))
print('\n=== FINAL TEST COMPARISON ===')
print(final_df.round(3).to_string(index=False))
TEST_LOCK.write_text(json.dumps(dict(
    completed_utc=datetime.now(timezone.utc).isoformat(),
    final_result_folder=str(FINAL_ROOT), frozen_weights_file=str(FROZEN_PATH),
    detector='fixed yolov8n.pt', tracker='unchanged', changed_component='SCI weights only'
), indent=2))
print('\nSaved:', FINAL_ROOT)
