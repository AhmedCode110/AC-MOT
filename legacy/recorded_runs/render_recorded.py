"""Offline presentation videos from saved AC-MOT outputs. No model inference.

python render_recorded.py /path/to/recorded_run [--dataset /path/to/dataset]
"""
import argparse
import gzip
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LABELS = ["A0 | Default ByteTrack", "A1 | Tuned ByteTrack", "A2 | + Adaptive threshold",
          "A3 | + Adaptive resolution (AC-MOT)", "A4 | + ReID (ablation only)"]
CHANGES = ["A0 to A1: tracker parameters", "A1 to A2: adaptive confidence / NMS",
           "A2 to A3: adaptive image resolution", "A3 to A4: ReID identity remapping"]


def text(img, content, xy, scale=.55, color=(235,235,235)):
    cv2.putText(img, str(content), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def load_records(path):
    with gzip.open(path, "rt") as f:
        return {int(r["frame"]): r for r in map(json.loads, f)}


def panel(image, rec, label, metric, width, height):
    out = np.full((height,width,3), (24,24,27), np.uint8)
    text(out,label,(14,25),.57,(80,220,255))
    h,w=image.shape[:2]
    factor=min((width-16)/w,(height-128)/h)
    nw,nh=int(w*factor),int(h*factor)
    ox,oy=(width-nw)//2,42
    out[oy:oy+nh,ox:ox+nw]=cv2.resize(image,(nw,nh))
    if rec is not None:
        for tid,box in zip(rec["ids"],rec["boxes_xyxy"]):
            x1,y1,x2,y2=box
            x1,x2=int(x1*factor)+ox,int(x2*factor)+ox
            y1,y2=int(y1*factor)+oy,int(y2*factor)+oy
            c=(int(80+(tid*37)%175),int(80+(tid*67)%175),int(80+(tid*97)%175))
            cv2.rectangle(out,(x1,y1),(x2,y2),c,1)
            text(out,f"ID {tid}",(x1,max(oy+10,y1-3)),.32,c)
        p=rec["settings"]
        text(out,f"Frame {rec['frame']} | scene={rec['scene']['scene']} | SCI={rec['scene']['sci']:.3f}",(12,height-63),.45)
        text(out,f"imgsz={p['imgsz']} conf={p['conf']:.3f} NMS IoU={p['iou']:.3f}",(12,height-42),.45)
    if metric is not None:
        text(out,f"SEQ: MOTA {metric.mota:.3f} | IDF1 {metric.idf1:.3f} | HOTA* {metric.hota:.3f}",(12,height-22),.43)
        text(out,f"IDS {int(metric.ids)} | FPS {metric.fps:.1f}",(width-230,25),.43)
    else:
        text(out,"Original dataset frame | colors encode track ID",(12,height-35),.45)
    return out


def render(path, frames, records, metrics, selected, title, start=0, length=240):
    # Videos are 20-fps illustrative playback; measured processing FPS is printed separately.
    chosen=frames[start:min(start+length,len(frames))]
    assert chosen
    width,height=1920,1080
    partial=path.with_name(path.stem+".partial.mp4")
    command=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-vcodec","rawvideo",
        "-pix_fmt","bgr24","-s",f"{width}x{height}","-r","20","-i","-","-an",
        "-c:v","libx264","-preset","veryfast","-crf","21","-pix_fmt","yuv420p",
        "-movflags","+faststart",str(partial)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for fp in chosen:
            frame=int(fp.stem)
            img=cv2.imread(str(fp))
            assert img is not None,fp
            canvas=np.full((height,width,3),(14,14,18),np.uint8)
            text(canvas,title,(24,32),.85,(80,220,255))
            text(canvas,"Legacy v10 evaluator | HOTA* proxy | sequence metrics, not clip metrics | playback 20 fps",(24,60),.58)
            if len(selected)==5:
                tiles=[panel(img,None,"Dataset input",None,640,490)]
                tiles += [panel(img,records[i][frame],LABELS[i],metrics[i],640,490) for i in selected]
                for j,tile in enumerate(tiles):
                    y=80+(j//3)*490; x=(j%3)*640
                    canvas[y:y+490,x:x+640]=tile
            else:
                for j,i in enumerate(selected):
                    canvas[80:1060,j*960:(j+1)*960]=panel(img,records[i][frame],LABELS[i],metrics[i],960,980)
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
    assert process.wait()==0,"ffmpeg failed"
    check=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-show_entries",
        "stream=codec_name,nb_read_frames,width,height","-of","json",str(partial)]))
    stream=check["streams"][0]
    assert stream["codec_name"]=="h264" and int(stream["nb_read_frames"])==len(chosen)
    partial.replace(path)
    return dict(file=path.name,start_frame=int(chosen[0].stem),end_frame=int(chosen[-1].stem),
        frames=len(chosen),playback_fps=20,systems=[LABELS[i] for i in selected],ffprobe=stream)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("run",type=Path)
    ap.add_argument("--dataset",type=Path)
    args=ap.parse_args()
    root=args.run
    cfg=json.loads((root/"configuration.json").read_text())
    dataset=args.dataset or Path(cfg["dataset"])
    systems=[s["name"] for s in cfg["systems"]]
    df=pd.read_csv(root/"per_sequence.csv")
    assert len(df)==85 and len(systems)==5
    video_dir=root/"presentation_videos"
    video_dir.mkdir(exist_ok=True)
    a3=df[df.system==systems[3]]
    # At least one real sequence for every observed dominant scene, no invented case labels.
    representative=set(a3.groupby("dominant_scene",sort=True).first().sequence)
    index=[]
    arabic=["# دليل شرح أمثلة التجربة", "",
        "الأرقام تخص الـsequence كاملة، والفيديو يعرض جزءًا متصلًا منها. التقييم هو بروتوكول v10 القديم وHOTA* تقريبية؛ ليست نتائج VisDrone الرسمية.",
        "ألوان الـboxes تميّز الهوية داخل النظام الواحد؛ اللون نفسه لا يثبت أن الهوية متطابقة بين نظامين.", ""]
    for seq in sorted(df.sequence.unique()):
        frames=sorted((dataset/"sequences"/seq).glob("*.jpg"))
        recs=[load_records(root/s/f"{seq}.frames.jsonl.gz") for s in systems]
        metrics=[df[(df.system==s)&(df.sequence==seq)].iloc[0] for s in systems]
        assert all(set(r)==set(int(f.stem) for f in frames) for r in recs)
        start=max(0,(len(frames)-240)//2)
        case=metrics[3].dominant_scene
        tasks=[(f"{seq}_A0-A4.mp4",list(range(5)),f"{seq} | A3 dominant scene: {case} | Five-system comparison")]
        if seq in representative:
            tasks += [(f"{seq}_A{i}-A{i+1}.mp4",[i,i+1],f"{seq} | {case} | {CHANGES[i]}") for i in range(4)]
            tasks += [(f"{seq}_A0-A3.mp4",[0,3],f"{seq} | {case} | Baseline vs adopted AC-MOT architecture")]
        for name,selected,title in tasks:
            target=video_dir/name
            sidecar=target.with_suffix(".json")
            if target.exists() and sidecar.exists():
                info=json.loads(sidecar.read_text())
            else:
                info=render(target,frames,recs,metrics,selected,title,start)
                info.update(sequence=seq,dominant_scene=case,title=title,
                    metrics_scope="whole sequence",metric_protocol=cfg["metric_protocol"])
                sidecar.write_text(json.dumps(info,indent=2))
            index.append(info)
            arabic.extend([f"## {name}", "", f"المثال من `{seq}`. نوع المشهد الغالب حسب محلل A3: `{case}`. الفريمات المعروضة: {info['start_frame']} إلى {info['end_frame']}.", ""])
            if len(selected)==2:
                first,last=metrics[selected[0]],metrics[selected[1]]
                arabic.append(f"المقارنة: {LABELS[selected[0]]} مقابل {LABELS[selected[1]]}. فرق MOTA: {last.mota-first.mota:+.4f}، فرق IDF1: {last.idf1-first.idf1:+.4f}، فرق IDS: {int(last.ids-first.ids):+d}. زيادة MOTA وIDF1 أفضل؛ انخفاض IDS أفضل. استخدم هذه الأرقام لشرح المكسب أو المقايضة الفعلية، ولا تفترض أن كل إضافة تحسّن كل المقاييس.")
            else:
                arabic.append("ابدأ بالفريم الأصلي، ثم قارن أماكن الـboxes واستمرار أرقام الهوية. A0 هو الأساس، A1 يضبط التراكر، A2 يضيف العتبات المتكيفة، A3 يضيف الدقة المتكيفة، وA4 يختبر ReID فقط. اقرأ إعدادات كل فريم أسفل صورته ثم اربط الملاحظة بمقاييس الـsequence المعروضة.")
            arabic.extend(["", f"[الفيديو]({name})", ""])
            print("VIDEO READY:",target,flush=True)
    (video_dir/"video_index.json").write_text(json.dumps(index,indent=2))
    lines=["# Recorded AC-MOT presentation examples", "",
        "All numbers shown in the videos are whole-sequence legacy-v10 metrics, not clip-only or official VisDrone scores.",
        "HOTA* is the saved proxy. Playback is 20 fps; processing FPS is printed separately.", "",
        "A0→A1: tracker tuning. A1→A2: adaptive thresholds. A2→A3: adaptive resolution. A3→A4: ReID ablation.",
        "A3 is the adopted architecture; A4 is an ablation only. Color identifies a track ID within a system, not a cross-system identity.", ""]
    for r in index:
        lines.append(f"- [{r['title']}]({r['file']}) — frames {r['start_frame']}–{r['end_frame']}; {r['dominant_scene']}.")
    (video_dir/"README.md").write_text("\n".join(lines)+"\n")
    (video_dir/"presentation_notes_ar.md").write_text("\n".join(arabic)+"\n")
    (root/"status.json").write_text(json.dumps(dict(status="complete",completed=85,total=85,videos=len(index))))
    print("ALL RECORDINGS AND VIDEOS COMPLETE:",root,flush=True)


if __name__=="__main__":
    main()
