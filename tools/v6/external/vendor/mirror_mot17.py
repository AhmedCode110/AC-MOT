"""Byte-identical local mirror of the MOT17 val-half frames (+ seqinfo, gt,
annotations) from the authoritative Google Drive copy; sha256 manifest."""
import hashlib, json, shutil
from pathlib import Path
SRC = Path("/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@gmail.com/My Drive/AC-MOT-SparseTrack-NEW/data/MOT17")
DST = Path("/Users/ahmedgouda/Desktop/acmot_external/data_mirror/MOT17")
ann = json.load(open(SRC / "annotations/val_half.json"))
files = ["annotations/val_half.json", "annotations/train_half.json", "annotations/train.json"]
for v in ann["videos"]:
    s = v["file_name"]
    files += [f"train/{s}/seqinfo.ini", f"train/{s}/gt/gt.txt"]
files += ["train/" + im["file_name"] for im in ann["images"]]
man = {}
for i, f in enumerate(files):
    d = DST / f
    d.parent.mkdir(parents=True, exist_ok=True)
    if not d.exists():
        shutil.copyfile(SRC / f, d)
    man[f] = hashlib.sha256(d.read_bytes()).hexdigest()
    if i % 500 == 0:
        print(i, len(files), flush=True)
json.dump(man, open(DST.parent / "MOT17_mirror_manifest.json", "w"), indent=0)
print("MIRROR_DONE", len(man))
