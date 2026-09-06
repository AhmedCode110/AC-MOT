"""AC-MOT experimental controls. No measured improvements are assumed."""
from collections import deque
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import numpy as np

ULTRALYTICS = '8.3.200'
TRACKEVAL = '12c8791b303e0a0b50f753af204249e622d0281a'
CLASSES = [0, 2, 5, 7]  # COCO person/car/bus/truck. No fabricated van class.

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

def boxes(value):
    a = np.asarray(value, dtype=np.float32).reshape(-1, 6)
    if not np.isfinite(a).all() or np.any(a[:,2:4] <= a[:,:2]):
        raise ValueError('Invalid detection coordinates')
    if np.any((a[:,4] < 0) | (a[:,4] > 1)) or np.any(a[:,5] != np.floor(a[:,5])):
        raise ValueError('Invalid score or class')
    return a

@dataclass(frozen=True)
class Config:
    name: str
    policy: str = 'fixed'
    size: int = 640
    recovery: bool = False
    stable: bool = False
    detector_feedback: bool = False
    adaptive_nms: bool = False
    nms: float = .45
    high: float = .25
    low: float = .10
    new: float = .25
    buffer: int = 30
    match: float = .80
    fuse: bool = True
    adaptive_birth: bool = False

    def validate(self):
        if self.policy not in ['fixed','adaptive','cycle'] or self.size not in [640,736,832]:
            raise ValueError('Unknown policy or size')
        if not (0 <= self.low < self.high <= self.new <= 1) or not 0 < self.match <= 1 or self.buffer < 1:
            raise ValueError('Invalid tracker thresholds')
        if not 0 < self.nms < 1:
            raise ValueError('Invalid NMS')
        if not self.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in self.name):
            raise ValueError('Unsafe system name')
        return self

def candidates():
    base = Config('A0_pinned_default')
    tuned = dict(high=.18, low=.04, new=.20, buffer=45, match=.86)
    controls = [base, Config('A1_tuned', **tuned),
        Config('R_recovery_only', recovery=True, **tuned),
        Config('A3_adaptive_reference',policy='adaptive', **tuned),
        Config('R_stable_only',policy='adaptive',stable=True, **tuned),
        Config('R_low_stable',policy='adaptive',stable=True,recovery=True, **tuned),
        Config('R_feedback',policy='adaptive',stable=True,recovery=True,detector_feedback=True, **tuned),
        Config('A3_v12_candidate',policy='adaptive',stable=True,recovery=True,detector_feedback=True,adaptive_birth=True, **tuned)]
    for size in [640,736,832]:
        controls.append(Config(f'Fixed_{size}_recovery',size=size,recovery=True, **tuned))
    controls += [Config('Cycle_recovery',policy='cycle',recovery=True, **tuned),
        Config('NMS_055',policy='adaptive',stable=True,recovery=True,detector_feedback=True,adaptive_birth=True,nms=.55, **tuned),
        Config('NMS_adaptive',policy='adaptive',stable=True,recovery=True,detector_feedback=True,adaptive_birth=True,adaptive_nms=True, **tuned)]
    for match in [.75,.80]:
        controls.append(Config(f'Assoc_{int(match*100)}',policy='adaptive',stable=True,recovery=True,detector_feedback=True,adaptive_birth=True, **(tuned|dict(match=match))))
    for buffer in [30,60]:
        controls.append(Config(f'Buffer_{buffer}',policy='adaptive',stable=True,recovery=True,detector_feedback=True,adaptive_birth=True, **(tuned|dict(buffer=buffer))))
    return [asdict(c.validate()) for c in controls]

class Controller:
    def __init__(self, cfg):
        self.cfg = cfg.validate()
        self.history = deque(maxlen=7)
        self.size = cfg.size
        self.last_change = 1
        self.sci = 0.
        self.tiny = 0.
        self.scene = 'clear'
        self.peak_count = 0
        self.drop_age = 0

    def choose(self, frame, visual, previous):
        c = self.cfg
        # Causal: only prior selected-resolution outputs enter the controller.
        if frame == 1 or frame % 10 == 1:
            previous = boxes(previous)
            b = previous[previous[:,4] >= .18]
            n = len(b)
            self.tiny = float(np.mean((b[:,2]-b[:,0])*(b[:,3]-b[:,1]) < 1024)) if n else 0.
            crowd = min(n/30,1.)
            raw = .30*crowd + .30*self.tiny + .20*min(visual['edges']/.14,1.)
            raw += .10*(visual['brightness']<80) + .05*(visual['blur']<180)
            self.history.append(raw)
            self.sci = float(np.mean(self.history))
            self.scene = ('night' if visual['brightness']<80 else 'blur' if visual['blur']<180 else
                          'tiny' if self.tiny>.50 else 'crowded' if crowd>.65 or visual['edges']>.13 else 'clear')
            # Short recovery probe when observations collapse, not perpetual high resolution.
            dropped = self.peak_count >= 5 and n < .4*self.peak_count
            self.drop_age = self.drop_age + 1 if dropped else 0
            self.peak_count = max(n, int(self.peak_count*.8))
            target = 832 if self.sci>.60 or self.tiny>.50 else 736 if self.sci>.35 or self.scene in ['crowded','tiny'] else 640
            if c.detector_feedback and 0 < self.drop_age <= 3:
                target = 832
            if c.stable:
                if target < self.size:
                    if self.size == 832 and (self.sci>.50 or self.tiny>.40): target=832
                    elif self.size == 736 and self.sci>.25: target=736
                if frame-self.last_change < 30: target=self.size
            if c.policy == 'adaptive' and target != self.size:
                self.size, self.last_change = target, frame
        size = self.size if c.policy == 'adaptive' else [640,736,832][((frame-1)//30)%3] if c.policy == 'cycle' else c.size
        # NMS alternatives have their own exact cache banks, never reapplied to already suppressed boxes.
        nms = .55 if c.adaptive_nms and self.scene in ['crowded','tiny'] else c.nms
        adaptive = float(np.clip(.245-.050*self.sci - (.012 if self.scene in ['crowded','tiny','night'] else 0),.19,.28))
        conf = c.low if c.recovery else adaptive if c.policy=='adaptive' else .25
        high = adaptive if c.adaptive_birth else c.high
        new = max(high,c.new) if not c.adaptive_birth else min(1.,high+.02)
        return dict(size=size,nms=nms,conf=conf,high=high,new=new,sci=self.sci,scene=self.scene)

def iou(a,b):
    a,b=np.asarray(a).reshape(-1,4),np.asarray(b).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    union=(a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter
    return np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
