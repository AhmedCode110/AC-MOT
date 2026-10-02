"""
Prepare the C-TWiX data layout (Guepardow/TWiX @ 3cff9cc) without images.
C-TWiX is coordinate-only: it reads no frame content, but lists frame
numbers from the image folders (and DanceTrack reads the size of frame 1). The folders are created with empty files
named exactly as the official frames (frame counts from the official
seqinfo.ini / KITTI seqmap), and the evaluation GT is built with the
authors' own tools/create_halves_mot17.py from the official gt.txt files.

  python ctwix_setup.py mot17 <MOT17 train dir with <seq>-FRCNN/{seqinfo.ini,gt/gt.txt}>
  python ctwix_setup.py kitti <KITTI dir with label_02/ and evaluate_tracking.seqmap.training>
  python ctwix_setup.py dancetrack <DanceTrack val dir with <seq>/{seqinfo.ini,gt/gt.txt}>
"""
import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path

TWIX = Path(os.environ.get("ACMOT_EXT", Path.home() / "acmot_work/acmot_external")) / "TWiX"
SEQ17 = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]


def _len(ini):
    c = configparser.ConfigParser()
    c.read(ini)
    return int(c["Sequence"]["seqLength"])


def mot17(src):
    src = Path(src)
    te = TWIX / "src/evaluation/TrackEval/data/gt/mot_challenge"
    for s in SEQ17:
        n = _len(src / f"{s}-FRCNN/seqinfo.ini")
        img = TWIX / f"data/MOT17/train/{s}-DPM/img1"
        img.mkdir(parents=True, exist_ok=True)
        for f in range(1, n + 1):
            (img / f"{f:06d}.jpg").touch()
        g = te / f"MOT17-train/{s}-DPM"
        (g / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy(src / f"{s}-FRCNN/gt/gt.txt", g / "gt/gt.txt")   # gt.txt is identical for DPM/FRCNN/SDP
        ini = (src / f"{s}-FRCNN/seqinfo.ini").read_text().replace(f"{s}-FRCNN", f"{s}-DPM")
        (g / "seqinfo.ini").write_text(ini)
    (te / "seqmaps").mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "create_halves_mot17.py", "-p", str(TWIX / "src/evaluation/TrackEval")],
                   cwd=TWIX / "tools", check=True)


def kitti(src):
    src = Path(src)
    root = TWIX / "data/KITTIMOT"
    for l in open(src / "evaluate_tracking.seqmap.training"):
        p = l.split()
        if not p:
            continue
        img = root / f"data_tracking_image_2/training/image_02/{p[0]}"
        img.mkdir(parents=True, exist_ok=True)
        for f in range(int(p[2]), int(p[3])):
            (img / f"{f:06d}.png").touch()
    lab = root / "data_tracking_label_2/training/label_02"
    lab.mkdir(parents=True, exist_ok=True)
    for f in (src / "label_02").glob("*.txt"):
        shutil.copy(f, lab / f.name)


def dancetrack(src):
    src = Path(src)
    for d in sorted(src.iterdir()):
        if not (d / "seqinfo.ini").exists():
            continue
        n = _len(d / "seqinfo.ini")
        out = TWIX / f"data/DanceTrack/val/{d.name}"
        (out / "img1").mkdir(parents=True, exist_ok=True)
        for f in range(1, n + 1):
            (out / f"img1/{f:08d}.jpg").touch()
        # DanceTrack.load_scene reads frame 1 only for its height and width: a black image of the
        # official size (seqinfo.ini imWidth / imHeight) is written there
        import cv2
        import numpy as np
        c = configparser.ConfigParser()
        c.read(d / "seqinfo.ini")
        h, w = int(c["Sequence"]["imHeight"]), int(c["Sequence"]["imWidth"])
        cv2.imwrite(str(out / "img1/00000001.jpg"), np.zeros((h, w, 3), np.uint8))
        shutil.copy(d / "seqinfo.ini", out / "seqinfo.ini")
        if (d / "gt/gt.txt").exists():
            (out / "gt").mkdir(exist_ok=True)
            shutil.copy(d / "gt/gt.txt", out / "gt/gt.txt")


if __name__ == "__main__":
    {"mot17": mot17, "kitti": kitti, "dancetrack": dancetrack}[sys.argv[1]](sys.argv[2])
