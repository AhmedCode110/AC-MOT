"""Portable orchestration only; original research modules remain unchanged."""
import argparse
import datetime as dt
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]

def environment(device='cpu', require_cuda=False, require_t4=False):
    import torch
    info = {'python':sys.version, 'packages':{}, 'cuda_available':torch.cuda.is_available(),
            'torch_cuda':torch.version.cuda, 'gpu':None}
    for name in ['torch','torchvision','ultralytics','numpy','scipy','lap','opencv-python']:
        info['packages'][name] = md.version(name)
    if info['packages']['ultralytics'] != '8.3.200':
        raise RuntimeError('Requires ultralytics==8.3.200')
    if require_cuda or require_t4:
        if device != '0': raise ValueError('Preserved v12 GPU pipeline requires device 0')
        if not info['cuda_available']: raise RuntimeError('CUDA required; CPU fallback is prohibited')
    if info['cuda_available']: info['gpu'] = torch.cuda.get_device_name(0)
    if require_t4 and 'T4' not in (info['gpu'] or ''): raise RuntimeError('Tesla T4 required')
    return info

def execute(cfg):
    mode=cfg['mode']
    if mode not in ['evaluate_saved','replay','live','cache','speedtest']:raise ValueError('Unknown mode')
    gpu=mode in ['live','cache','speedtest']
    info=environment(str(cfg.get('device','0' if gpu else 'cpu')),gpu or cfg.get('require_cuda',False),gpu or cfg.get('require_t4',False))
    dataset=Path(cfg['dataset']).expanduser().resolve()
    if not (dataset/'annotations').is_dir():raise ValueError(f'Missing dataset annotations: {dataset}')
    if gpu and not (dataset/'sequences').is_dir():raise ValueError('Missing sequences directory')
    output_root=Path(cfg['output_root']).expanduser().resolve()
    # A new envelope for every invocation prevents legacy cache/replay resume from overwriting old runs.
    output_root.mkdir(parents=True,exist_ok=True)
    envelope=output_root/(mode+'_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    envelope.mkdir(exist_ok=False)
    output=envelope/'result'
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip())
    receipt={'git_commit':commit,'git_dirty':dirty,'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
             'environment':info,'configuration':cfg,'dataset_path':str(dataset),'seed':cfg.get('seed'),
             'seed_note':'Unspecified preserves original behavior; exact bitwise determinism is not guaranteed',
             'output':str(output),'status':'started'}
    def save(): (envelope/'run_metadata.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    with (envelope/'environment.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    def run(script,*args):
        command=[sys.executable,str(ROOT/script),*map(str,args)]
        receipt.setdefault('commands',[]).append(command);save()
        # Optional seeding is opt-in, applied inside the child process before running source.
        if cfg.get('seed') is not None:
            seed=int(cfg['seed']);code="import random, numpy as np, torch, runpy, sys; s=int(sys.argv.pop(1)); random.seed(s); np.random.seed(s); torch.manual_seed(s); p=sys.argv.pop(1); sys.argv[0]=p; runpy.run_path(p,run_name='__main__')"
            command=[sys.executable,'-c',code,str(seed),*command[1:]]
        subprocess.run(command,cwd=ROOT,check=True)
    try:
        if mode=='evaluate_saved':
            run('evaluate.py',cfg['saved_run'],'--dataset',dataset,'--trackeval',cfg['trackeval'],'--output',output)
        elif mode=='replay':
            args=['replay','--cache',cfg['cache'],'--output',output]
            if cfg.get('systems'):args+=['--config',cfg['systems']]
            if cfg.get('frozen'):args+=['--frozen',cfg['frozen']]
            run('experiment.py',*args)
        else:
            weights=Path(cfg['weights']).expanduser().resolve()
            if weights.name!='yolov8n.pt':raise ValueError('Preserved detector requires yolov8n.pt')
            if not weights.exists():
                weights.parent.mkdir(parents=True,exist_ok=True)
                subprocess.run([sys.executable,'-c',"from ultralytics import YOLO; YOLO('yolov8n.pt')"],cwd=weights.parent,check=True)
            names=cfg.get('sequences') or sorted(p.name for p in (dataset/'sequences').iterdir() if p.is_dir())
            if not names or any(Path(n).name!=n for n in names):raise ValueError('Invalid sequence list')
            if cfg.get('expected_sequences') is not None and len(names)!=cfg['expected_sequences']:raise ValueError('Unexpected sequence count')
            count=sum(len(list((dataset/'sequences'/n).glob('*.jpg'))) for n in names)
            if cfg.get('expected_frames') is not None and count!=cfg['expected_frames']:raise ValueError('Unexpected frame count')
            if mode=='cache' and (cfg['split']!='development' or 'test' in dataset.name.lower()):raise ValueError('Use a separate development dataset for cache search')
            seq=envelope/'sequences.json';seq.write_text(json.dumps(names))
            if mode=='speedtest':
                systems=envelope/'systems.json';systems.write_text(json.dumps(cfg['systems'],indent=2))
                run('scripts/speedtest_top3.py',
                    '--dataset',dataset,
                    '--sequences',seq,
                    '--weights',weights,
                    '--systems',systems,
                    '--output',output,
                    '--target-fps',str(cfg.get('target_fps',25.0)),
                    '--gate-frames',str(cfg.get('gate_frames',300)),
                    '--progress-every',str(cfg.get('progress_every',25)))
            else:
                args=[mode,'--dataset',dataset,'--sequences',seq,'--weights',weights,'--output',output,'--split',cfg['split']]
                if mode=='live':args+=['--frozen',cfg['frozen'],'--repeats',str(cfg.get('repeats',3))]
                run('experiment.py',*args)
                if mode=='live' and cfg.get('evaluate_repeats',True):
                    import shutil
                    for repeat in range(cfg.get('repeats',3)):
                        folder=output/f'repeat_{repeat}'
                        for name in ['configuration.json','dataset_manifest.json']:shutil.copy2(output/name,folder/name)
                        run('evaluate.py',folder,'--dataset',dataset,'--trackeval',cfg['trackeval'],'--output',output/f'trackeval_repeat_{repeat}')
        receipt['status']='completed'
    except BaseException as exc:
        receipt['status']='failed';receipt['error_type']=type(exc).__name__;raise
    finally:
        save();print(f'OUTPUT FOLDER: {envelope}',flush=True)
    return envelope

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--check',action='store_true');a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.check:
        gpu=cfg['mode'] in ['cache','live','speedtest']
        print(json.dumps(environment(str(cfg.get('device','cpu')),gpu or cfg.get('require_cuda',False),gpu or cfg.get('require_t4',False)),indent=2))
    else:execute(cfg)
if __name__=='__main__':main()
