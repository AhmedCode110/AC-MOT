"""
Build a read-only VisDrone-layout *view* of the UAVDT MOT test split so the
same cache / replay / evaluation tools run unchanged. Canonical data are
never modified.

view/
  sequences/<seq> -> symlink to UAV-benchmark-M/<seq> (img000001.jpg ...)
  annotations/<seq>.txt : UAVDT <seq>_gt.txt rows with score != 0 and
      frame >= 1, written as VisDrone columns
      frame,id,x,y,w,h,1,4,0,0   (flag=1, category=4 "vehicle", trunc=0,
      occ=0 so the class-agnostic loader keeps every UAVDT GT box)
  ignore/<seq>.txt : <seq>_gt_ignore.txt boxes (frame,x,y,w,h)
  task.json : {"classes": [2, 5, 7]} — vehicle task (car/bus/truck)

Split and GT/ignore rules follow UAVDT_EXTERNAL_PROTOCOL_FREEZE.json and
UAVDT_ADAPTER_V1_FREEZE.json (frozen 2026-09-12, before this project).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ahmedgouda/Library/CloudStorage/"
            "GoogleDrive-a7medgoda1@gmail.com/My Drive/AC-MOT-shared/"
            "UAVDT_EXTERNAL_GENERALIZATION")
IMG = ROOT / "data/extracted/UAV-benchmark-M/UAV-benchmark-M"
GT = ROOT / ("data/extracted/UAV-benchmark-MOTD_v1.0/"
             "UAV-benchmark-MOTD_v1.0/GT")
VIEW = Path("outputs/uavdt_view")


def main():
    proto = json.loads((ROOT / "UAVDT_EXTERNAL_PROTOCOL_FREEZE.json")
                       .read_text())
    seqs = proto["test_sequences"]
    for sub in ("sequences", "annotations", "ignore"):
        (VIEW / sub).mkdir(parents=True, exist_ok=True)
    total = 0
    for s in seqs:
        link = VIEW / "sequences" / s
        if not link.exists():
            link.symlink_to(IMG / s, target_is_directory=True)
        n = len(list((IMG / s).glob("*.jpg")))
        assert n == proto["verified_frame_counts"][s], (s, n)
        total += n
        a = np.loadtxt(GT / f"{s}_gt.txt", delimiter=",", ndmin=2)
        a = a[(a[:, 6] != 0) & (a[:, 0] >= 1)]
        out = np.c_[a[:, :6], np.ones(len(a)), np.full(len(a), 4),
                    np.zeros(len(a)), np.zeros(len(a))]
        np.savetxt(VIEW / "annotations" / f"{s}.txt", out, delimiter=",",
                   fmt="%d")
        ign = GT / f"{s}_gt_ignore.txt"
        if ign.exists() and ign.stat().st_size:
            b = np.loadtxt(ign, delimiter=",", ndmin=2)
            np.savetxt(VIEW / "ignore" / f"{s}.txt", b[:, [0, 2, 3, 4, 5]],
                       delimiter=",", fmt="%d")
    (VIEW / "task.json").write_text(json.dumps({"classes": [2, 5, 7]}))
    assert total == proto["expected_test_frames"], total
    print("UAVDT test view:", len(seqs), "sequences,", total, "frames")


if __name__ == "__main__":
    main()
