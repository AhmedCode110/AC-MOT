"""Dedicated portable runner for AC-MOT v17."""
import argparse,json,subprocess,sys,datetime as dt,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from portable_v17 import resolve_portable_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/paper_eval_v17.json');a=p.parse_args()
    cfg=resolve_portable_config(json.loads((ROOT/a.config).read_text()),verbose=True)
    dataset=Path(cfg['dataset']);outroot=Path(cfg['output_root']);outroot.mkdir(parents=True,exist_ok=True)
    seqs=sorted(p.name for p in (dataset/'sequences').iterdir() if p.is_dir())
    frames=sum(len(list((dataset/'sequences'/s).glob('*.jpg'))) for s in seqs)
    if len(seqs)!=cfg['expected_sequences'] or frames!=cfg['expected_frames']:
        raise RuntimeError(f"Dataset mismatch: sequences={len(seqs)} frames={frames}")
    import torch
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Tesla T4 required for comparable v17 run')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    envelope=outroot/f'paper_eval_v17_{stamp}';envelope.mkdir(exist_ok=False);result=envelope/'result'
    seqfile=envelope/'sequences.json';seqfile.write_text(json.dumps(seqs))
    sysfile=envelope/'systems.json';sysfile.write_text(json.dumps(cfg['systems'],indent=2))
    print('\n[V17 PLAN] 17 videos x',len(cfg['systems']),'systems')
    print('[V17 PLAN] Each video: copy to local SSD (excluded) -> decode -> SCI/SmartCalibrator -> fresh YOLOv8n FP16 -> ByteTrack -> save tracks -> TrackEval.')
    print('[V17 PLAN] Progress lines show video, frame, SCI, conf, NMS IoU, imgsz, processing FPS and ETA.')
    print('[V17 REALTIME] >=25 strict | 20-24.99 acceptable | <20 below realtime\n')
    cmd=[sys.executable,str(ROOT/cfg['paper_eval_script']),'--dataset',str(dataset),'--sequences',str(seqfile),'--weights',cfg['weights'],'--systems',str(sysfile),'--output',str(result),'--trackeval',cfg['trackeval'],'--target-fps',str(cfg['acceptable_realtime_fps']),'--progress-every',str(cfg['progress_every']),'--backend',cfg['backend'],'--decode-chunk-size',str(cfg['decode_chunk_size'])]
    subprocess.run(cmd,cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/v17_finalize.py'),str(result),'--strict',str(cfg['strict_realtime_fps']),'--acceptable',str(cfg['acceptable_realtime_fps'])],cwd=ROOT,check=True)
    (envelope/'run_metadata.json').write_text(json.dumps({'version':'v17','configuration':cfg,'result':str(result)},indent=2)+'\n')
    print('\nOUTPUT FOLDER:',envelope)

if __name__=='__main__':main()
