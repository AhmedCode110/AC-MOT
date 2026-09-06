"""Official TrackEval metrics on preserved, class-agnostic research-filter tracks.

This adapter is NOT the official VisDrone benchmark preprocessing protocol.
"""
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import importlib.metadata
from pathlib import Path
import numpy as np

REVISION = '12c8791b303e0a0b50f753af204249e622d0281a'

# Pinned upstream uses removed NumPy scalar aliases. Restore aliases only;
# no metric equations or matching logic are changed.
for alias, scalar in [('float', float), ('int', int)]:
    if alias not in np.__dict__:
        setattr(np, alias, scalar)

def iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    for boxes in (a, b):
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError('Non-finite or non-positive box')
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    union = (a[:, 2:] - a[:, :2]).prod(axis=1)[:, None] + (b[:, 2:] - b[:, :2]).prod(axis=1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

def prepare(gt, frames, count):
    if [r['frame'] for r in frames] != list(range(1, count + 1)):
        raise ValueError('Recording must contain every frame exactly once, in order')
    gt = np.asarray(gt, dtype=float).reshape(-1, 10)
    if not np.isfinite(gt).all() or np.any(gt[:, 0] < 1) or np.any(gt[:, 0] > count):
        raise ValueError('Invalid annotation frame or value')
    gt = gt[np.isin(gt[:, 7], [1, 4, 5, 6, 9]) & (gt[:, 6] == 1) & (gt[:, 8] < 2) & (gt[:, 9] < 2)]
    gids = sorted(set(gt[:, 1].tolist()))
    pids = sorted({v for r in frames for v in r['ids']})
    gm, pm = {v:i for i,v in enumerate(gids)}, {v:i for i,v in enumerate(pids)}
    data = dict(num_timesteps=count, num_gt_ids=len(gids), num_tracker_ids=len(pids),
                num_gt_dets=len(gt), num_tracker_dets=sum(len(r['ids']) for r in frames),
                gt_ids=[], tracker_ids=[], similarity_scores=[])
    for r in frames:
        g = gt[gt[:, 0] == r['frame']]
        ids = r['ids']
        if len(ids) != len(r['boxes_xyxy']) or len(set(ids)) != len(ids) or len(set(g[:, 1])) != len(g):
            raise ValueError('Duplicate IDs or mismatched boxes/IDs')
        gb = g[:, 2:6].copy()
        gb[:, 2:] += gb[:, :2]
        data['gt_ids'].append(np.array([gm[v] for v in g[:, 1]], dtype=int))
        data['tracker_ids'].append(np.array([pm[v] for v in ids], dtype=int))
        data['similarity_scores'].append(iou(gb, r['boxes_xyxy']))
    return data

def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--trackeval', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    rev = subprocess.check_output(['git', '-C', str(a.trackeval), 'rev-parse', 'HEAD'], text=True).strip()
    if rev != REVISION:
        raise ValueError('TrackEval revision differs from pinned revision')
    if subprocess.check_output(['git','-C',str(a.trackeval),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('TrackEval has modified tracked files')
    sys.path.insert(0, str(a.trackeval))
    import trackeval
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({'THRESHOLD': .5, 'PRINT_CONFIG': False}),
               trackeval.metrics.Identity({'THRESHOLD': .5, 'PRINT_CONFIG': False})]
    cfg = json.loads((a.run/'configuration.json').read_text())
    manifest = json.loads((a.run/'dataset_manifest.json').read_text())
    if not manifest or len({s['sequence'] for s in manifest}) != len(manifest):
        raise ValueError('Empty or duplicate sequence manifest')
    results, hashes = {}, {}
    for system in cfg['systems']:
        name = system['name']
        per_metric = {m.get_name(): {} for m in metrics}
        for seq in manifest:
            sn = seq['sequence']
            ann = a.dataset/'annotations'/f'{sn}.txt'
            digest = hashlib.sha256(ann.read_bytes()).hexdigest()
            if digest != seq['annotation_sha256']:
                raise ValueError(f'Annotation hash differs: {sn}')
            recording = a.run/name/f'{sn}.frames.jsonl.gz'
            hashes[str(recording)] = hashlib.sha256(recording.read_bytes()).hexdigest()
            with gzip.open(recording, 'rt') as f:
                frames = [json.loads(line) for line in f]
            data = prepare(np.loadtxt(ann, delimiter=',', ndmin=2), frames, seq['frames'])
            for m in metrics:
                per_metric[m.get_name()][sn] = m.eval_sequence(data)
        results[name] = {m.get_name(): dict(per_sequence=per_metric[m.get_name()],
            combined=m.combine_sequences(per_metric[m.get_name()])) for m in metrics}
    a.output.mkdir(parents=True, exist_ok=False)
    def serial(v):
        return v.tolist() if hasattr(v, 'tolist') else v
    (a.output/'metrics.json').write_text(json.dumps(results, default=serial, indent=2, allow_nan=False))
    import csv
    with (a.output/'summary.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['system', 'HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'IDS', 'FN', 'FP'])
        for name, r in results.items():
            h, c, i = [r[k]['combined'] for k in ['HOTA', 'CLEAR', 'Identity']]
            writer.writerow([name, *[float(np.mean(h[k])) for k in ['HOTA','DetA','AssA']],
                             c['MOTA'], i['IDF1'], c['IDSW'], c['CLR_FN'], c['CLR_FP']])
    (a.output/'protocol.json').write_text(json.dumps(dict(trackeval_commit=rev,
        protocol='Custom class-agnostic research filter; no benchmark ignore-region preprocessing',
        official_visdrone=False, gt_filter=cfg['ground_truth_filter'],
        aggregation='TrackEval combine_sequences, not macro mean', scale='0–1',
        packages={p:importlib.metadata.version(p) for p in ['numpy','scipy']},
        source_hashes=hashes, evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    print(a.output/'summary.csv')

if __name__ == '__main__':
    main()
