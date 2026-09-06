"""Recompute IoU-gated MOT metrics from saved tracks, without YOLO inference.

This is a separate class-agnostic diagnostic using the research GT filter,
not the official VisDrone evaluator. It never replaces the legacy summaries.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
import motmetrics as mm


def distances(gt, pred, minimum_iou):
    if not len(gt) or not len(pred):
        return np.empty((len(gt),len(pred)))
    low=np.maximum(gt[:,None,:2],pred[None,:,:2])
    high=np.minimum(gt[:,None,2:],pred[None,:,2:])
    inter=np.prod(np.maximum(0,high-low),axis=2)
    ga=np.prod(np.maximum(0,gt[:,2:]-gt[:,:2]),axis=1)
    pa=np.prod(np.maximum(0,pred[:,2:]-pred[:,:2]),axis=1)
    union=ga[:,None]+pa[None,:]-inter
    iou=np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
    cost=1-iou
    cost[iou<minimum_iou]=np.nan
    return cost


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("--iou",type=float,default=.5)
    parser.add_argument("--dataset",type=Path)
    args=parser.parse_args()
    assert 0 < args.iou <= 1
    root=args.run
    cfg=json.loads((root/"configuration.json").read_text())
    dataset=args.dataset or Path(cfg["dataset"])
    rows=[]
    for system in cfg["systems"]:
        for record in sorted((root/system["name"]).glob("*.frames.jsonl.gz")):
            sequence=record.name.removesuffix(".frames.jsonl.gz")
            gt=pd.read_csv(dataset/"annotations"/f"{sequence}.txt",header=None,
                names=["frame","id","x","y","w","h","score","cat","trunc","occ"])
            gt=gt[gt.cat.isin([1,4,5,6,9]) & (gt.score==1) & (gt.trunc<2) & (gt.occ<2)]
            acc=mm.MOTAccumulator(auto_id=True)
            with gzip.open(record,"rt") as f:
                for line in f:
                    r=json.loads(line)
                    g=gt[gt.frame==r["frame"]]
                    boxes=np.column_stack([g.x,g.y,g.x+g.w,g.y+g.h])
                    pred=np.asarray(r["boxes_xyxy"],dtype=float).reshape(-1,4)
                    acc.update(g.id.tolist(),r["ids"],distances(boxes,pred,args.iou))
            values=mm.metrics.create().compute(acc,metrics=["mota","idf1","num_switches",
                "num_misses","num_false_positives","recall","precision"],name=sequence).iloc[0].to_dict()
            rows.append(dict(system=system["name"],sequence=sequence,minimum_iou=args.iou,**values))
    df=pd.DataFrame(rows)
    assert len(df)==85
    for _,g in df.groupby("system"):
        assert g.sequence.nunique()==17
    stem=f"diagnostic_iou{args.iou:g}"
    df.to_csv(root/f"{stem}_per_sequence.csv",index=False)
    summary=df.groupby("system",sort=False).agg(sequences=("sequence","count"),mota=("mota","mean"),
        idf1=("idf1","mean"),ids=("num_switches","sum"),fn=("num_misses","sum"),
        fp=("num_false_positives","sum"),recall=("recall","mean"),precision=("precision","mean"))
    summary.to_csv(root/f"{stem}_summary.csv")
    (root/f"{stem}_protocol.json").write_text(json.dumps(dict(minimum_iou=args.iou,
        source="Saved tracking outputs; no model inference",class_agnostic=True,
        gt_filter=cfg["ground_truth_filter"],official_visdrone=False,hota_computed=False,
        summary="Unweighted sequence means; IDS/FN/FP summed"),indent=2))
    print(summary.to_string())


if __name__=="__main__":
    main()
