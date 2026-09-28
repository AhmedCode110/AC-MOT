"""Verify a MOT17 val-half copy against BoostTrack's official split definition
(data/tools/convert_mot17_to_coco.py in github.com/vukasin-stanojevic/BoostTrack,
= ByteTrack convention) frame-for-frame, and the raw GT against the authors'
shipped evaluation GT (results/gt/MOT17-val). Read-only; writes a JSON report."""
import hashlib, json, sys
from pathlib import Path
import numpy as np

M = Path(sys.argv[1])            # .../MOT17 (train/, annotations/)
BT = Path(sys.argv[2])           # BoostTrack repo
OUT = Path(sys.argv[3])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expected_val_half():
    images, anns = [], []
    image_cnt = ann_cnt = video_cnt = tid_curr = 0
    tid_last = -1
    gt_lines = {}
    for seq in sorted(p.name for p in (M / "train").iterdir() if p.is_dir()):
        if "FRCNN" not in seq:
            continue
        video_cnt += 1
        sp = M / "train" / seq
        n = len([f for f in (sp / "img1").iterdir() if "jpg" in f.name])
        ini = dict(l.strip().split("=", 1) for l in (sp / "seqinfo.ini").read_text().splitlines() if "=" in l)
        h, w = int(ini["imHeight"]), int(ini["imWidth"])
        r0, r1 = n // 2 + 1, n - 1
        for i in range(n):
            if i < r0 or i > r1:
                continue
            images.append({"file_name": f"{seq}/img1/{i+1:06d}.jpg", "id": image_cnt + i + 1,
                           "frame_id": i + 1 - r0, "prev_image_id": image_cnt + i if i > 0 else -1,
                           "next_image_id": image_cnt + i + 2 if i < n - 1 else -1,
                           "video_id": video_cnt, "height": h, "width": w})
        a = np.loadtxt(sp / "gt/gt.txt", dtype=np.float32, delimiter=",")
        sel = a[(a[:, 0].astype(int) - 1 >= r0) & (a[:, 0].astype(int) - 1 <= r1)].copy()
        sel[:, 0] -= r0
        gt_lines[seq] = ["{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}".format(
            int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]), int(o[6]), int(o[7]), o[8]) for o in sel]
        for row in a:
            fid = int(row[0])
            if fid - 1 < r0 or fid - 1 > r1:
                continue
            tid, ann_cnt = int(row[1]), ann_cnt + 1
            if int(row[6]) != 1 or int(row[7]) in [3, 4, 5, 6, 9, 10, 11]:
                continue
            if int(row[7]) in [2, 7, 8, 12]:
                cat = -1
            else:
                cat = 1
                if tid != tid_last:
                    tid_curr, tid_last = tid_curr + 1, tid
            anns.append({"id": ann_cnt, "category_id": cat, "image_id": image_cnt + fid, "track_id": tid_curr,
                         "bbox": [float(x) for x in row[2:6]], "conf": float(row[6]), "iscrowd": 0,
                         "area": float(row[4] * row[5])})
        image_cnt += n
    return images, anns, gt_lines


img_e, ann_e, gt_e = expected_val_half()
dj = json.load(open(M / "annotations/val_half.json"))
rep = {"dataset_root": str(M), "val_half_json_sha256": sha(M / "annotations/val_half.json")}
rep["images_expected"], rep["images_drive"] = len(img_e), len(dj["images"])
key = lambda d: {k: d[k] for k in ("file_name", "id", "frame_id", "prev_image_id", "next_image_id", "video_id", "height", "width")}
rep["images_identical"] = [key(x) for x in dj["images"]] == img_e
rep["videos"] = dj.get("videos")
rep["annotations_expected"], rep["annotations_drive"] = len(ann_e), len(dj["annotations"])
def akey(d):
    return (d["id"], d["category_id"], d["image_id"], d["track_id"], tuple(round(v, 3) for v in d["bbox"]), round(d["conf"], 3))
rep["annotations_identical"] = [akey(x) for x in dj["annotations"]] == [akey(x) for x in ann_e]
rep["per_sequence"] = {}
for seq, lines in gt_e.items():
    ship = BT / "results/gt/MOT17-val" / seq / "gt/gt.txt"
    s = ship.read_text().splitlines()
    ini = dict(l.strip().split("=", 1) for l in (BT / "results/gt/MOT17-val" / seq / "seqinfo.ini").read_text().splitlines() if "=" in l)
    fr = sorted({x["frame_id"] for x in img_e if x["file_name"].startswith(seq)})
    rep["per_sequence"][seq] = dict(
        raw_gt_sha256=sha(M / "train" / seq / "gt/gt.txt"),
        val_frames=len(fr), first_original_frame=min(int(x["file_name"][-10:-4]) for x in img_e if x["file_name"].startswith(seq)),
        shipped_seqLength=int(ini["seqLength"]), seqLength_match=int(ini["seqLength"]) == len(fr),
        shipped_gt_sha256=sha(ship), derived_gt_equals_shipped=lines == s,
        n_gt_rows=len(lines))
rep["all_identical"] = rep["images_identical"] and rep["annotations_identical"] and all(
    v["derived_gt_equals_shipped"] and v["seqLength_match"] for v in rep["per_sequence"].values())
OUT.write_text(json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items() if k not in ("per_sequence", "videos")}, indent=1))
for s, v in rep["per_sequence"].items():
    print(s, v["val_frames"], v["first_original_frame"], v["seqLength_match"], v["derived_gt_equals_shipped"], v["n_gt_rows"])
