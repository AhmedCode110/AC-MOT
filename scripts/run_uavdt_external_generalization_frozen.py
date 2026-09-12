#!/usr/bin/env python3
"""Frozen UAVDT external-generalization test for Baseline, V1 Trial24, V2 Trial22.
No training, tuning, recalibration, parameter changes, or test-driven reselection.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, os, re, shutil, subprocess, sys, time, traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch
import ultralytics
from ultralytics import YOLO

BASE_COMMIT="5becc52a569f271ee8b73ce47c67d495de0d64a5"
TE_COMMIT="12c8791b303e0a0b50f753af204249e622d0281a"
ADAPTER_SHA="8957b4da276121408b5769c23eb3de84dd30667c88d539e437ba6b6afc28648a"
CLASSES=[2,5,7]
TEST=["M0203","M0205","M0208","M0209","M0403","M0601","M0602","M0606","M0701","M0801",
      "M0802","M1001","M1004","M1007","M1009","M1101","M1301","M1302","M1303","M1401"]
COUNTS={"M0203":1007,"M0205":646,"M0208":265,"M0209":1576,"M0403":514,"M0601":372,"M0602":480,
"M0606":1374,"M0701":1308,"M0801":298,"M0802":1101,"M1001":1859,"M1004":269,"M1007":659,
"M1009":604,"M1101":864,"M1301":1182,"M1302":719,"M1303":445,"M1401":1050}
IGNORE=set(["M0203","M0205","M0208","M0403","M0601","M0602","M0606","M0701","M0802","M1001",
"M1004","M1007","M1009","M1101","M1301","M1302","M1303","M1401"])
V2_EXPECT={"weight_crowd":.1646452656714585,"weight_tiny":.1746265279504544,
"weight_edge":.5076112530333374,"weight_night":.1206956469372537,"weight_blur":.0324213064074956,
"conf_easy":.4,"conf_hard":.4,"nms_easy":.6,"nms_hard":.35,
"threshold_mid":.2927135841069045,"threshold_high":.6661671600900015}

REPO=Path(os.getenv("ACMOT_REPO","/content/AC-MOT"))
TE=Path(os.getenv("ACMOT_TRACKEVAL","/content/TrackEval"))
ROOT=Path(os.getenv("UAVDT_EXTERNAL_ROOT","/content/drive/MyDrive/AC-MOT-shared/UAVDT_EXTERNAL_GENERALIZATION"))
V1ROOT=Path("/content/drive/MyDrive/AC-MOT-shared/defensible_acmot_3workers")
V2ROOT=Path("/content/drive/MyDrive/AC-MOT-shared/V2_POST_SELECTION_TEST_2026-09-12")
ADAPTER=ROOT/"UAVDT_ADAPTER_V1_FREEZE.json"; LOCK=ROOT/"UAVDT_EXTERNAL_TEST_SYSTEMS_LOCK.json"
V1F=V1ROOT/"FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"; CALF=V1ROOT/"DETECTOR_DERIVED_CUE_CALIBRATION.json"
V2F=V2ROOT/"V2_TRIAL22_FINAL_FREEZE.json"
ACCESS=ROOT/"UAVDT_EXTERNAL_TEST_ACCESS_STARTED.json"
OUT=ROOT/"results"/"FROZEN_3SYSTEM_TEST_2026-09-12"
WEIGHTS=Path("/content/weights/yolov8n.pt"); VIEW=Path("/content/UAVDT_DATASET_VIEW")
PROGRESS=int(os.getenv("ACMOT_PROGRESS_EVERY","100"))

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def atom(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+".tmp")
    q.write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if hasattr(v,"tolist") else float(v),allow_nan=False)); q.replace(p)
def head(p): return subprocess.check_output(["git","-C",str(p),"rev-parse","HEAD"],text=True).strip()
def sortkey(p):
    d=re.findall(r"\d+",p.stem); return (int(d[-1]) if d else 10**18,p.name)
def unique(root,name):
    h=list(Path(root).rglob(name))
    if len(h)!=1: raise RuntimeError(f"Expected one {name}, found {len(h)}")
    return h[0]
def find_trial(o):
    if isinstance(o,dict):
        for k in ("selected_trial","trial"):
            try:
                if int(o.get(k,-1))==22:return 22
            except: pass
        for v in o.values():
            r=find_trial(v)
            if r:return r
    if isinstance(o,list):
        for v in o:
            r=find_trial(v)
            if r:return r
def find_v2(o):
    ks=set(V2_EXPECT)
    if isinstance(o,dict):
        if ks.issubset(o): return {k:float(o[k]) for k in ks}
        for v in o.values():
            r=find_v2(v)
            if r:return r
    if isinstance(o,list):
        for v in o:
            r=find_v2(v)
            if r:return r

def preflight():
    if ultralytics.__version__!="8.3.200" or np.__version__!="2.2.6" or scipy.__version__!="1.15.3":
        raise RuntimeError(f"Runtime mismatch: ultralytics={ultralytics.__version__}, numpy={np.__version__}, scipy={scipy.__version__}")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0): raise RuntimeError("Tesla T4 required")
    if head(REPO)!=BASE_COMMIT or head(TE)!=TE_COMMIT: raise RuntimeError("Frozen repo/TrackEval commit mismatch")
    for p in [ADAPTER,LOCK,V1F,CALF,V2F]:
        if not p.is_file(): raise RuntimeError(f"Missing frozen file: {p}")
    a=json.loads(ADAPTER.read_text()); l=json.loads(LOCK.read_text()); v1=json.loads(V1F.read_text()); v2=json.loads(V2F.read_text())
    if a["sha256"]!=ADAPTER_SHA or a["adapter"]["detector_class_mapping"]["classes"]!=CLASSES: raise RuntimeError("Adapter freeze mismatch")
    if l["status"]!="FROZEN_BEFORE_UAVDT_TEST" or l["test_sequences"]!=TEST or int(l["expected_test_frames"])!=16592: raise RuntimeError("System lock mismatch")
    for k,v in l["rules"].items():
        if v is not False: raise RuntimeError(f"Frozen rule changed: {k}")
    ss={x["name"]:x for x in l["systems"]}
    if int(ss["V1_Trial24_Frozen"]["selected_trial"])!=24 or int(ss["V2_Trial22_Frozen"]["selected_trial"])!=22: raise RuntimeError("Candidate lock mismatch")
    if ss["V1_Trial24_Frozen"]["source_freeze_sha256"]!=sha(V1F) or ss["V1_Trial24_Frozen"]["calibration_sha256"]!=sha(CALF) or ss["V2_Trial22_Frozen"]["source_freeze_sha256"]!=sha(V2F): raise RuntimeError("Frozen source changed")
    if int(v1["selected_trial"])!=24 or find_trial(v2)!=22: raise RuntimeError("Freeze trial mismatch")
    p2=find_v2(v2)
    if p2 is None: raise RuntimeError("V2 params missing")
    for k,e in V2_EXPECT.items():
        if abs(p2[k]-e)>1e-10: raise RuntimeError(f"V2 parameter mismatch: {k}")
    return l,v1,p2,json.loads(CALF.read_text())

def mark_access(lock):
    x={"status":"UAVDT_EXTERNAL_TEST_ACCESS_STARTED","started_utc":now(),"systems":["Baseline_Frozen","V1_Trial24_Frozen","V2_Trial22_Frozen"],
       "adapter_sha256":ADAPTER_SHA,"systems_lock_sha256":sha(LOCK),"v1_freeze_sha256":sha(V1F),
       "v1_calibration_sha256":sha(CALF),"v2_freeze_sha256":sha(V2F),"repository_commit":BASE_COMMIT,
       "trackeval_commit":TE_COMMIT,"gpu":torch.cuda.get_device_name(0),"tuning":False,"recalibration":False,
       "parameter_changes":False,"candidate_reselection":False,"selection_on_test":False}
    if ACCESS.exists():
        old=json.loads(ACCESS.read_text())
        for k in ["adapter_sha256","systems_lock_sha256","v1_freeze_sha256","v1_calibration_sha256","v2_freeze_sha256","repository_commit","trackeval_commit"]:
            if old.get(k)!=x[k]: raise RuntimeError(f"Existing access marker mismatch: {k}")
    else: atom(ACCESS,x)

def discover():
    hs=[p for p in (ROOT/"data"/"extracted").rglob("M0203") if p.is_dir() and len(list(p.glob("*.jpg")))==COUNTS["M0203"]]
    if len(hs)!=1: raise RuntimeError(f"Image root ambiguity: {hs}")
    imroot=hs[0].parent; gtroot=unique(ROOT/"data"/"extracted","M0203_gt.txt").parent
    man=[]; total=0
    for s in TEST:
        fs=sorted((imroot/s).glob("*.jpg"),key=sortkey)
        if len(fs)!=COUNTS[s]: raise RuntimeError(f"{s}: {len(fs)} != {COUNTS[s]}")
        if not (gtroot/f"{s}_gt.txt").is_file(): raise RuntimeError(f"Missing GT {s}")
        if s in IGNORE and not (gtroot/f"{s}_gt_ignore.txt").is_file(): raise RuntimeError(f"Missing ignore {s}")
        man.append({"sequence":s,"frames":len(fs),"frame_sha256":{p.name:"" for p in fs}})
        total+=len(fs)
    if total!=16592: raise RuntimeError(f"Total frames {total} != 16592")
    if VIEW.exists(): shutil.rmtree(VIEW)
    (VIEW/"sequences").mkdir(parents=True)
    for s in TEST: os.symlink(str(imroot/s),str(VIEW/"sequences"/s),target_is_directory=True)
    return imroot,gtroot,man

if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from core import boxes
from core_v17 import PresentationSpec
import scripts.paper_eval_v17 as pe

def detect(model,img,size,nms,conf,backend):
    kw=dict(source=img,conf=float(conf),iou=float(nms),imgsz=int(size),classes=CLASSES,max_det=1000,device=0,verbose=False)
    if backend=="pytorch_fp16": kw["half"]=True
    return boxes(model.predict(**kw)[0].boxes.data.cpu().numpy())

def rank(v,a):
    a=np.asarray(a,float); return float(np.searchsorted(a,v,side="right")/len(a)) if len(a) else 0.
def visual(img):
    sm=cv2.resize(img,None,fx=.25,fy=.25,interpolation=cv2.INTER_AREA); g=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    return {"brightness":float(g.mean()),"blur":float(cv2.Laplacian(g,cv2.CV_64F).var()),"edges":float(np.sqrt(gx*gx+gy*gy).mean())}

class Static:
    def __init__(self,spec): self.spec=spec.validate()
    def choose(self,frame,v,prev): return {"conf":.25,"nms":.45,"size":640,"sci":0.,"scene":"baseline"}

class Empirical:
    def __init__(self,spec,p,cal,res):
        self.spec=spec.validate(); self.p=p; self.cal=cal; self.res=list(map(int,res)); self.hist=deque(maxlen=spec.smoothing_window); self.sci=0.; self.tiny=0.
        ks=["crowd","tiny","edge","night","blur"]; w=np.array([float(p[f"weight_{k}"]) for k in ks]); w/=w.sum(); self.w=dict(zip(ks,w))
    def choose(self,frame,v,prev):
        if frame==1 or (frame-1)%self.spec.analysis_stride==0:
            prev=boxes(prev); n=len(prev); self.tiny=float(np.mean((prev[:,2]-prev[:,0])*(prev[:,3]-prev[:,1])<1024)) if n else 0.
            c={"crowd":rank(n,self.cal["crowd_sorted"]),"tiny":self.tiny,"edge":rank(float(v["edges"]),self.cal["edge_sorted"]),
               "night":1-rank(float(v["brightness"]),self.cal["brightness_sorted"]),"blur":1-rank(float(v["blur"]),self.cal["blur_sorted"])}
            self.hist.append(float(np.clip(sum(self.w[k]*c[k] for k in self.w),0,1))); self.sci=float(np.mean(self.hist))
        p=self.p; conf=float(p["conf_easy"]+self.sci*(p["conf_hard"]-p["conf_easy"])); nms=float(p["nms_easy"]+self.sci*(p["nms_hard"]-p["nms_easy"]))
        r0,r1,r2=self.res; size=r2 if self.sci>=float(p["threshold_high"]) else r1 if self.sci>=float(p["threshold_mid"]) else r0
        return {"conf":conf,"nms":nms,"size":int(size),"sci":self.sci,"scene":"empirical_external"}

for alias,scalar in [("float",float),("int",int)]:
    if alias not in np.__dict__: setattr(np,alias,scalar)
sys.path.insert(0,str(TE))
import trackeval

def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4); b=np.asarray(b,float).reshape(-1,4)
    if not len(a) or not len(b): return np.zeros((len(a),len(b)))
    lt=np.maximum(a[:,None,:2],b[None,:,:2]); rb=np.minimum(a[:,None,2:],b[None,:,2:]); wh=np.maximum(rb-lt,0); inter=wh[:,:,0]*wh[:,:,1]
    aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]); ab=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]); u=aa[:,None]+ab[None,:]-inter
    return np.divide(inter,u,out=np.zeros_like(inter),where=u>0)
def xywh(x):
    x=np.asarray(x,float).reshape(-1,4); y=x.copy(); y[:,2]=x[:,0]+x[:,2]; y[:,3]=x[:,1]+x[:,3]; return y
def ignmap(gtroot,s):
    if s not in IGNORE:return {}
    a=np.loadtxt(gtroot/f"{s}_gt_ignore.txt",delimiter=",",ndmin=2); d={}
    for f in np.unique(a[:,0].astype(int)): d[int(f)]=xywh(a[a[:,0].astype(int)==f,2:6])
    return d
def prep(gtroot,s,rec,nframes):
    g0=np.loadtxt(gtroot/f"{s}_gt.txt",delimiter=",",ndmin=2); g=g0[(g0[:,6]!=0)&(g0[:,0]>=1)].copy(); F=int(g[:,0].max())
    gid=sorted(set(g[:,1].astype(int))); gm={v:i for i,v in enumerate(gid)}
    with gzip.open(rec,"rt") as f: rr=[json.loads(z) for z in f]
    if [int(r["frame"]) for r in rr]!=list(range(1,nframes+1)): raise RuntimeError(f"{s}: incomplete recording")
    ig=ignmap(gtroot,s); proc=[]; tids=set(); removed=0
    for r in rr[:F]:
        fr=int(r["frame"]); B=np.asarray(r["boxes_xyxy"],float).reshape(-1,4); I=np.asarray(r["ids"],int)
        if len(set(I.tolist()))!=len(I): raise RuntimeError(f"{s}/{fr}: duplicate IDs")
        Q=ig.get(fr,np.zeros((0,4)))
        if len(B) and len(Q):
            rm=((B[:,None,0]>Q[None,:,0])&(B[:,None,1]>Q[None,:,1])&(B[:,None,2]<Q[None,:,2])&(B[:,None,3]<Q[None,:,3])).any(1)
            removed+=int(rm.sum()); B=B[~rm]; I=I[~rm]
        tids.update(I.tolist()); proc.append((B,I))
    tm={v:i for i,v in enumerate(sorted(tids))}
    data={"num_timesteps":F,"num_gt_ids":len(gid),"num_tracker_ids":len(tm),"num_gt_dets":len(g),"num_tracker_dets":0,"gt_ids":[],"tracker_ids":[],"similarity_scores":[]}
    for fr in range(1,F+1):
        q=g[g[:,0].astype(int)==fr]; gi=np.array([gm[int(v)] for v in q[:,1]],int); gb=xywh(q[:,2:6]); B,I=proc[fr-1]; ti=np.array([tm[int(v)] for v in I],int)
        data["gt_ids"].append(gi); data["tracker_ids"].append(ti); data["similarity_scores"].append(iou(gb,B)); data["num_tracker_dets"]+=len(ti)
    return data,{"raw_gt":len(g0),"eval_gt":len(g),"score0_removed":int(np.sum(g0[:,6]==0)),"ignore_pred_removed":removed,"timesteps":F}

def metrics():
    return [trackeval.metrics.HOTA(),trackeval.metrics.CLEAR({"THRESHOLD":.5,"PRINT_CONFIG":False}),trackeval.metrics.Identity({"THRESHOLD":.5,"PRINT_CONFIG":False})]
def summ(d):
    h=d["HOTA"]; c=d["CLEAR"]; i=d["Identity"]
    return {"HOTA":float(np.mean(h["HOTA"])),"DetA":float(np.mean(h["DetA"])),"AssA":float(np.mean(h["AssA"])),"MOTA":float(c["MOTA"]),"IDF1":float(i["IDF1"]),"IDS":int(c["IDSW"]),"FN":int(c["CLR_FN"]),"FP":int(c["CLR_FP"])}
def evaluate(gtroot,local,name,man):
    ms=metrics(); pm={m.get_name():{} for m in ms}; rows=[]; pp={}
    for q in man:
        s=q["sequence"]; d,meta=prep(gtroot,s,local/name/f"{s}.frames.jsonl.gz",q["frames"]); pp[s]=meta; one={}
        for m in ms: one[m.get_name()]=m.eval_sequence(d); pm[m.get_name()][s]=one[m.get_name()]
        rows.append({"system":name,"sequence":s,**summ(one)})
    comb={m.get_name():m.combine_sequences(pm[m.get_name()]) for m in ms}; agg=summ(comb); e=local/"uavdt_eval"; e.mkdir()
    atom(e/"aggregate.json",agg); atom(e/"metrics.json",{"per_metric":pm,"combined":comb}); atom(e/"preprocess.json",pp)
    with (e/"per_sequence.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return agg

def specs(v1,p2,cal):
    p1=dict(v1["optimized_parameters"]); r=list(map(int,v1["resolution_levels"]))
    if r!=[512,928,960] or (int(v1["smoothing_window"]),int(v1["analysis_stride"]))!=(7,10): raise RuntimeError(f"Unexpected frozen V1 design {r}")
    return {
    "Baseline_Frozen":(dict(name="Baseline_Frozen",tracker_profile="default",adaptive_threshold=False,adaptive_resolution=False,smoothing_window=1,analysis_stride=1),lambda s:Static(s),lambda x:{},[640]),
    "V1_Trial24_Frozen":(dict(name="V1_Trial24_Frozen",tracker_profile="tuned",adaptive_threshold=True,adaptive_resolution=True,smoothing_window=7,analysis_stride=10),lambda s:Empirical(s,p1,cal,r),visual,r),
    "V2_Trial22_Frozen":(dict(name="V2_Trial22_Frozen",tracker_profile="tuned",adaptive_threshold=True,adaptive_resolution=True,smoothing_window=7,analysis_stride=10),lambda s:Empirical(s,p2,cal,r),visual,r)}

def runone(name,bundle,man,gtroot,prov):
    persist=OUT/name; done=OUT/f"{name}_DONE.json"
    if done.exists():
        print(f"[RESUME] {name} already complete."); return json.loads((persist/"uavdt_eval"/"aggregate.json").read_text()),json.loads((persist/"timing.json").read_text())
    if persist.exists(): shutil.move(str(persist),str(OUT/f"{name}_INTERRUPTED_{int(time.time())}"))
    local=Path(f"/content/uavdt_{name}"); shutil.rmtree(local,ignore_errors=True); local.mkdir()
    system,factory,vfn,res=bundle; atom(local/"configuration.json",{"systems":[system],"detector_classes":CLASSES,"selection_or_tuning_on_test":False}); atom(local/"dataset_manifest.json",man)
    oc,ov,od=pe.PresentationController,pe.visual,pe.detect_realtime; wm={640:res[0]} if len(res)==1 else {640:res[0],736:res[1],832:res[2]}
    def det(model,img,size,nms,conf,backend):
        if abs(float(nms)-.45)<1e-12 and abs(float(conf)-.19)<1e-12: size=wm.get(int(size),res[0])
        return detect(model,img,size,nms,conf,backend)
    try:
        pe.PresentationController=factory; pe.visual=vfn; pe.detect_realtime=det
        timing=pe.run_system(system,VIEW,man,WEIGHTS,Path("/content/weights/unused.engine"),25.,PROGRESS,"pytorch",16,local,1,1)
    finally: pe.PresentationController,pe.visual,pe.detect_realtime=oc,ov,od
    agg=evaluate(gtroot,local,name,man); atom(local/"timing.json",timing); atom(local/"RESULT.json",{"system":name,"metrics":{**agg,"FPS":timing["processing_fps"]},**prov,"tuning":False,"recalibration":False,"candidate_reselection":False})
    cp=OUT/f".{name}.copying"; shutil.rmtree(cp,ignore_errors=True); shutil.copytree(local,cp); cp.rename(persist); atom(done,json.loads((persist/"RESULT.json").read_text()))
    print(f"[DONE] {name}: MOTA={100*agg['MOTA']:.3f}% HOTA={100*agg['HOTA']:.3f}% IDF1={100*agg['IDF1']:.3f}% IDS={agg['IDS']} FPS={timing['processing_fps']:.2f}")
    return agg,timing

def main():
    print("="*94); print("UAVDT FROZEN EXTERNAL GENERALIZATION — BASELINE / V1 / V2"); print("="*94)
    lock,v1,p2,cal=preflight(); OUT.mkdir(parents=True,exist_ok=True); mark_access(lock)
    imroot,gtroot,man=discover()
    WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
    if not WEIGHTS.exists():
        old=os.getcwd(); os.chdir(WEIGHTS.parent)
        try: YOLO("yolov8n.pt")
        finally: os.chdir(old)
    prov={"adapter_sha256":ADAPTER_SHA,"systems_lock_sha256":sha(LOCK),"v1_freeze_sha256":sha(V1F),"v1_calibration_sha256":sha(CALF),"v2_freeze_sha256":sha(V2F),"repository_commit":BASE_COMMIT,"trackeval_commit":TE_COMMIT,"gpu":torch.cuda.get_device_name(0)}
    ev=unique(ROOT/"data"/"extracted","evaluateTracking.m")
    atom(OUT/"UAVDT_TEST_DATA_PROVENANCE.json",{**prov,"image_root":str(imroot),"gt_root":str(gtroot),"evaluateTracking_sha256":sha(ev),"test_sequences":TEST,"total_frames":16592,"ignore_sequences":sorted(IGNORE),"classes":CLASSES})
    S=specs(v1,p2,cal); rows=[]
    for name in ["Baseline_Frozen","V1_Trial24_Frozen","V2_Trial22_Frozen"]:
        a,t=runone(name,S[name],man,gtroot,prov); rows.append({"system":name,**a,"FPS":float(t["processing_fps"]),"mean_imgsz":float(t["mean_imgsz"]),"mean_conf":float(t["mean_conf"]),"mean_nms_iou":float(t["mean_nms_iou"])})
    with (OUT/"UAVDT_FINAL_COMPARISON.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    atom(OUT/"UAVDT_FINAL_COMPARISON.json",rows); atom(OUT/"UAVDT_EXTERNAL_TEST_DONE.json",{"status":"COMPLETE","completed_utc":now(),**prov,"systems":rows,"tuning":False,"recalibration":False,"candidate_reselection":False})
    print("\n"+"="*94); print("UAVDT EXTERNAL TEST COMPLETE"); print("="*94)
    for r in rows: print(f"{r['system']}: MOTA={100*r['MOTA']:.3f}% HOTA={100*r['HOTA']:.3f}% IDF1={100*r['IDF1']:.3f}% IDS={r['IDS']} FN={r['FN']} FP={r['FP']} FPS={r['FPS']:.2f}")
    print("Saved:",OUT)

if __name__=="__main__":
    try: main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True); atom(OUT/"UAVDT_EXTERNAL_TEST_FAILED.json",{"status":"FAILED","failed_utc":now(),"error":repr(e),"traceback":traceback.format_exc(),"tuning":False,"recalibration":False,"candidate_reselection":False}); raise
