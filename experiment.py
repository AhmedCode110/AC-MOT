"""Cache once, replay ByteTrack candidates, or measure a frozen system on T4."""
import argparse
import gzip
import json
import importlib.metadata as metadata
import time
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import numpy as np
from core import Config,Controller,ULTRALYTICS,CLASSES,atomic_json,sha,fingerprint,boxes,candidates

def environment():
    if metadata.version('ultralytics') != ULTRALYTICS:
        raise RuntimeError(f'Requires ultralytics=={ULTRALYTICS}')
    return {p:metadata.version(p) for p in ['ultralytics','numpy','torch','torchvision','scipy']}

def dataset_manifest(dataset, names):
    if not names or len(set(names))!=len(names): raise ValueError('Empty/duplicate sequence list')
    result=[]
    for name in names:
        if Path(name).name != name: raise ValueError('Invalid sequence name')
        files=sorted((dataset/'sequences'/name).glob('*.jpg'))
        if not files or [int(f.stem) for f in files]!=list(range(1,len(files)+1)):
            raise ValueError(f'Missing/non-contiguous frames: {name}')
        ann=dataset/'annotations'/f'{name}.txt'
        gt=np.loadtxt(ann,delimiter=',',ndmin=2)
        if gt.shape[1]!=10 or not np.isfinite(gt).all() or np.any(gt[:,0]<1) or np.any(gt[:,0]>len(files)):
            raise ValueError(f'Invalid GT: {ann}')
        result.append(dict(sequence=name,frames=len(files),annotation_sha256=sha(ann),
                           frame_sha256={f.name:sha(f) for f in files}))
    return result

def visual(img):
    import cv2
    gray=cv2.cvtColor(cv2.resize(img,None,fx=.25,fy=.25),cv2.COLOR_BGR2GRAY)
    return dict(brightness=float(gray.mean()),blur=float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                edges=float(cv2.Canny(gray,50,120).mean()/255))

def sync():
    import torch
    torch.cuda.synchronize()

def new_model(weights):
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('T4 required for detector cache and live timing')
    return YOLO(str(weights))

def detect(model,img,size,nms):
    r=model.predict(img,conf=.01,iou=nms,imgsz=size,classes=CLASSES,max_det=1000,
                    half=False,device=0,verbose=False)[0]
    if bool(model.predictor.model.fp16): raise RuntimeError('Unexpected FP16')
    return boxes(r.boxes.data.cpu().numpy())

def cache(a):
    import cv2
    env=environment()
    names=json.loads(a.sequences.read_text())
    manifest=dataset_manifest(a.dataset,names)
    cfg=dict(dataset=str(a.dataset),split=a.split,manifest=manifest,environment=env,
             weights_sha256=sha(a.weights),sizes=[640,736,832],nms=[.45,.55],conf=.01,
             max_det=1000,classes=CLASSES,precision='FP32',source_sha256=sha(Path(__file__)),
             core_sha256=sha(Path(__file__).with_name('core.py')))
    cfg['fingerprint']=fingerprint(cfg)
    a.output.mkdir(parents=True,exist_ok=True)
    meta=a.output/'cache.json'
    if meta.exists() and json.loads(meta.read_text())!=cfg: raise ValueError('Resume cache mismatch')
    atomic_json(meta,cfg)
    model=new_model(a.weights)
    for seq in manifest:
        name=seq['sequence']; target=a.output/f'{name}.jsonl.gz'; receipt=a.output/f'{name}.complete.json'
        if target.exists() and receipt.exists() and json.loads(receipt.read_text())['sha256']==sha(target): continue
        first=cv2.imread(str(a.dataset/'sequences'/name/next(iter(seq['frame_sha256']))))
        for size in cfg['sizes']: detect(model,first,size,.45)
        partial=target.with_suffix('.partial')
        with gzip.open(partial,'wt') as f:
            for index,filename in enumerate(seq['frame_sha256'],1):
                img=cv2.imread(str(a.dataset/'sequences'/name/filename))
                if img is None: raise ValueError(f'Unreadable {filename}')
                bank={}; times={}
                for size in cfg['sizes']:
                    for nms in cfg['nms']:
                        key=f'{size}_{nms:.2f}'; sync(); start=time.perf_counter()
                        bank[key]=detect(model,img,size,nms).tolist(); sync()
                        times[key]=time.perf_counter()-start
                f.write(json.dumps(dict(frame=index,shape=list(img.shape[:2]),visual=visual(img),bank=bank,cache_detection_seconds=times))+'\n')
        partial.replace(target)
        atomic_json(receipt,dict(sha256=sha(target),fingerprint=cfg['fingerprint']))
        print('Cached',name,flush=True)

def make_tracker(cfg):
    from ultralytics.trackers.byte_tracker import BYTETracker
    return BYTETracker(SimpleNamespace(track_high_thresh=cfg.high,track_low_thresh=cfg.low,
        new_track_thresh=cfg.new,track_buffer=cfg.buffer,match_thresh=cfg.match,fuse_score=cfg.fuse),frame_rate=30)

def track(tracker,dets,shape,params):
    from ultralytics.engine.results import Boxes
    dets=boxes(dets); dets=dets[dets[:,4]>=params['conf']]
    tracker.args.track_high_thresh=params['high'];tracker.args.new_track_thresh=params['new']
    result=np.asarray(tracker.update(Boxes(dets,shape)),dtype=float).reshape(-1,8)
    if len(set(result[:,4]))!=len(result): raise ValueError('Duplicate track IDs')
    return result,dets

def recording(frame,t,params,elapsed):
    return dict(frame=frame,ids=t[:,4].astype(int).tolist(),boxes_xyxy=t[:,:4].tolist(),
                scores=t[:,5].tolist(),classes=t[:,6].astype(int).tolist(),settings=params,
                elapsed_seconds=elapsed)

def replay(a):
    env=environment(); cache_cfg=json.loads((a.cache/'cache.json').read_text())
    if env!=cache_cfg['environment']: raise ValueError('Cache/replay environment mismatch')
    configs=json.loads(a.config.read_text()) if a.config else candidates()
    if cache_cfg['split']!='development' and not a.frozen:
        raise ValueError('Search prohibited on test data. Supply a frozen development selection.')
    if a.frozen:
        locked=json.loads(a.frozen.read_text())
        if locked.get('development_split')!='development': raise ValueError('Invalid development lock')
        configs=locked['systems']
    checked=[Config(**c).validate() for c in configs]
    if len({c.name for c in checked})!=len(checked):raise ValueError('Duplicate system names')
    cfg=dict(systems=[asdict(c) for c in checked],cache_fingerprint=cache_cfg['fingerprint'],
        dataset=cache_cfg['dataset'],split=cache_cfg['split'],environment=env,weights_sha256=cache_cfg['weights_sha256'],
        source_sha256=sha(Path(__file__)),core_sha256=sha(Path(__file__).with_name('core.py')),
        fps_protocol='Replay wall time only; never deployment FPS',
        ground_truth_filter=dict(categories=[1,4,5,6,9],score=1,occlusion_lt=2,truncation_lt=2))
    a.output.mkdir(parents=True,exist_ok=True)
    config_path=a.output/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())!=cfg:raise ValueError('Replay resume mismatch')
    atomic_json(config_path,cfg);atomic_json(a.output/'dataset_manifest.json',cache_cfg['manifest'])
    for c in checked:
        folder=a.output/c.name;folder.mkdir(exist_ok=True)
        for seq in cache_cfg['manifest']:
            sn=seq['sequence'];target=folder/f'{sn}.frames.jsonl.gz';receipt=folder/f'{sn}.complete.json'
            src=a.cache/f'{sn}.jsonl.gz'
            cache_receipt=json.loads((a.cache/f'{sn}.complete.json').read_text())
            if sha(src)!=cache_receipt['sha256']:raise ValueError('Corrupt cache')
            if target.exists() and receipt.exists() and sha(target)==json.loads(receipt.read_text())['sha256']:continue
            tracker=make_tracker(c);control=Controller(c);previous=[];stats=[]; sizes=[]
            partial=target.with_suffix('.partial');count=0
            with gzip.open(src,'rt') as source,gzip.open(partial,'wt') as dest:
                for count,line in enumerate(source,1):
                    r=json.loads(line)
                    if r['frame']!=count: raise ValueError('Missing frame')
                    start=time.perf_counter();params=control.choose(count,r['visual'],previous)
                    key=f"{params['size']}_{params['nms']:.2f}"
                    t,d=track(tracker,r['bank'][key],tuple(r['shape']),params)
                    elapsed=time.perf_counter()-start
                    previous=d if c.detector_feedback else t[:,[0,1,2,3,5,6]]
                    stats.append(elapsed);sizes.append(params['size'])
                    dest.write(json.dumps(recording(count,t,params,elapsed))+'\n')
            if count!=seq['frames']:raise ValueError('Truncated cache')
            partial.replace(target)
            atomic_json(receipt,dict(sha256=sha(target),replay_seconds=sum(stats),frames=count,
                resolution_switches=int(np.count_nonzero(np.diff(sizes))),mean_size=float(np.mean(sizes))))
        print('Replayed',c.name,flush=True)

def progress_bar(done,total,width=30):
    ratio=0.0 if total<=0 else min(max(done/total,0.0),1.0)
    filled=int(round(ratio*width))
    return '[' + '='*filled + '>'*(filled<width) + '.'*max(width-filled-1,0) + ']'

def live(a):
    """Fresh detector + controller + tracker timing for a frozen candidate, repeated."""
    import cv2
    import torch
    environment();locked=json.loads(a.frozen.read_text())
    if locked.get('development_split')!='development':raise ValueError('Development lock required')
    if locked['weights_sha256']!=sha(a.weights) or locked['environment']!=environment():
        raise ValueError('Frozen development weights/environment differ')
    systems=[Config(**c).validate() for c in locked['systems']]
    manifest=dataset_manifest(a.dataset,json.loads(a.sequences.read_text()))
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_json(a.output/'dataset_manifest.json',manifest)
    atomic_json(a.output/'configuration.json',dict(systems=[asdict(c) for c in systems],
        weights_sha256=sha(a.weights),environment=environment(),split=a.split,
        source_sha256=sha(Path(__file__)),core_sha256=sha(Path(__file__).with_name('core.py')),
        realtime_gate=dict(min_fps=a.min_realtime_fps,check_after_frames=a.realtime_check_after_frames,
                           enabled=not a.no_realtime_abort),
        ground_truth_filter=dict(categories=[1,4,5,6,9],score=1,occlusion_lt=2,truncation_lt=2)))
    measurements=[]
    total_frames = sum(seq['frames'] for seq in manifest)
    grand_total = total_frames * len(systems) * a.repeats
    grand_done = 0
    grand_start = time.perf_counter()
    print(f'LIVE RUN START | systems={len(systems)} repeats={a.repeats} sequences={len(manifest)} frames/repeat/system={total_frames} total_frame_runs={grand_total}', flush=True)
    print(f'REALTIME GATE | target={a.min_realtime_fps:.2f} FPS check_after={a.realtime_check_after_frames} measured frames enabled={not a.no_realtime_abort}', flush=True)
    for repeat in range(a.repeats):
        for c in systems:
            system_start = time.perf_counter()
            realtime_checked=False
            print(f'BEGIN repeat={repeat+1}/{a.repeats} system={c.name}', flush=True)
            model=new_model(a.weights);total=0.;n=0;latencies=[]
            torch.cuda.reset_peak_memory_stats()
            for seq in manifest:
                sn=seq['sequence'];folder=a.output/f'repeat_{repeat}'/c.name;folder.mkdir(parents=True,exist_ok=True)
                seq_start = time.perf_counter()
                print(f'  SEQ {sn} frames={seq["frames"]}', flush=True)
                paths=[a.dataset/'sequences'/sn/f for f in seq['frame_sha256']]
                first=cv2.imread(str(paths[0]))
                for size in [640,736,832]:
                    for _ in range(3):detect(model,first,size,c.nms)
                control=Controller(c);tracker=make_tracker(c);previous=[]
                with gzip.open(folder/f'{sn}.frames.jsonl.gz','wt') as f:
                    for frame,path in enumerate(paths,1):
                        sync();start=time.perf_counter();img=cv2.imread(str(path))
                        if img is None:raise ValueError(str(path))
                        # Visual analysis follows the same schedule as replay.
                        v=visual(img) if frame==1 or frame%10==1 else {}
                        params=control.choose(frame,v,previous)
                        dets=detect(model,img,params['size'],params['nms'])
                        t,d=track(tracker,dets,img.shape[:2],params)
                        previous=d if c.detector_feedback else t[:,[0,1,2,3,5,6]]
                        sync();elapsed=time.perf_counter()-start;total+=elapsed;n+=1;latencies.append(elapsed)
                        f.write(json.dumps(recording(frame,t,params,elapsed))+'\n')
                        grand_done += 1
                        fps = n / total if total > 0 else 0.0
                        if (not realtime_checked) and n >= a.realtime_check_after_frames:
                            realtime_checked=True
                            verdict='PASS' if fps >= a.min_realtime_fps else 'FAIL'
                            print(f'  REALTIME CHECK {verdict} | system={c.name} measured_frames={n} average_FPS={fps:.2f} target={a.min_realtime_fps:.2f}', flush=True)
                            if verdict=='FAIL' and not a.no_realtime_abort:
                                raise RuntimeError(
                                    f'Real-time gate failed for {c.name}: {fps:.2f} FPS < {a.min_realtime_fps:.2f} FPS '
                                    f'after {n} measured frames. Run stopped early to avoid wasting compute.'
                                )
                        if frame == 1 or frame == seq['frames'] or frame % 50 == 0:
                            now = time.perf_counter()
                            sys_elapsed = now - system_start
                            grand_elapsed = now - grand_start
                            done_pct = 100.0 * grand_done / grand_total
                            eta = (grand_elapsed / grand_done) * (grand_total - grand_done) if grand_done else 0.0
                            bar=progress_bar(grand_done,grand_total)
                            rt='REALTIME' if fps >= a.min_realtime_fps else 'BELOW-RT'
                            print(
                                f'  {bar} {done_pct:6.2f}% global={grand_done}/{grand_total} '
                                f'repeat={repeat+1}/{a.repeats} system={c.name} seq={sn} frame={frame}/{seq["frames"]} '
                                f'FPS={fps:.2f} [{rt}] target={a.min_realtime_fps:.2f} '
                                f'system_elapsed={sys_elapsed/60:.1f}m ETA={eta/60:.1f}m',
                                flush=True
                            )
                print(f'  DONE SEQ {sn} elapsed={(time.perf_counter()-seq_start)/60:.1f}m', flush=True)
            measurements.append(dict(system=c.name,repeat=repeat,frames=n,seconds=total,fps=n/total,
                p95_ms=float(np.percentile(latencies,95)*1000),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                realtime_target_fps=a.min_realtime_fps,realtime_pass=(n/total)>=a.min_realtime_fps,
                excludes='warmup and output serialization; includes frame read, analysis, detector and tracker'))
            atomic_json(a.output/'timing.json',measurements)
            print(f'DONE repeat={repeat+1}/{a.repeats} system={c.name} frames={n} seconds={total:.3f} fps={n/total:.3f}', flush=True)
            del model;torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ['cache','live']:
        s=sub.add_parser(name);s.add_argument('--dataset',required=True,type=Path);s.add_argument('--sequences',required=True,type=Path)
        s.add_argument('--weights',required=True,type=Path);s.add_argument('--output',required=True,type=Path)
        s.add_argument('--split',required=True,choices=['development','test'])
        if name=='live':
            s.add_argument('--frozen',required=True,type=Path)
            s.add_argument('--repeats',type=int,default=3)
            s.add_argument('--min-realtime-fps',type=float,default=25.0)
            s.add_argument('--realtime-check-after-frames',type=int,default=300)
            s.add_argument('--no-realtime-abort',action='store_true')
    s=sub.add_parser('replay');s.add_argument('--cache',required=True,type=Path);s.add_argument('--output',required=True,type=Path)
    s.add_argument('--config',type=Path);s.add_argument('--frozen',type=Path)
    a=p.parse_args()
    if a.command=='live':
        if a.repeats<1:raise ValueError('repeats must be positive')
        if a.min_realtime_fps<=0:raise ValueError('min realtime FPS must be positive')
        if a.realtime_check_after_frames<1:raise ValueError('realtime check frames must be positive')
    globals()[a.command](a)

if __name__=='__main__':main()
