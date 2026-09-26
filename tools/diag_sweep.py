"""Run named structural variants on the two diagnostic sequences for both
detectors (cache replay) and check the PREDECLARED diagnostic criteria:

  0137: HOTA >= V2cA HOTA - 2   (YOLO >= 39.31, RT-DETR >= 41.89)
  RT-DETR 0268: MOTA > 0, HOTA >= 35, tracks/frame <= 14
  YOLO 0268: HOTA >= 32.5
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from tools.run_policy_validation import run
from universal_policy_pipeline import POLICIES, replace

DATASET = "/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val"
SEQS = ["uav0000137_00458_v", "uav0000268_05773_v"]
OUT = Path("outputs/policy_runs")
FLOOR = {("yolov8", "uav0000137_00458_v"): ("HOTA", 39.31),
         ("rtdetr", "uav0000137_00458_v"): ("HOTA", 41.89),
         ("yolov8", "uav0000268_05773_v"): ("HOTA", 32.5)}


def check(det, seq, m, tracks):
    if (det, seq) in FLOOR:
        k, v = FLOOR[(det, seq)]
        return m[k] >= v
    return m["MOTA"] > 0 and m["HOTA"] >= 35 and tracks <= 14


def main(variants):
    for name, base, overrides in variants:
        policy = replace(POLICIES[base], name=name, **overrides)
        ok_all = True
        for det in ("yolov8", "rtdetr"):
            out = OUT / f"{det}_{name}_diag"
            if out.exists():
                summary = list(csv.DictReader(open(out / "summary.csv")))
                metrics = list(csv.DictReader(open(out / "metrics.csv")))
            else:
                summary, metrics = run(policy, DATASET,
                                       f"outputs/det_cache/{det}", out, SEQS,
                                       quiet=True)
            for s, m in zip(summary, metrics):
                m = {k: (float(v) if k not in ("sequence", "IDS_source")
                         else v) for k, v in m.items()}
                ok = check(det, s["sequence"], m, float(s["mean_tracks"]))
                ok_all &= ok
                print(f"{name:<14}{det:<7}{s['sequence'][:12]} "
                      f"MOTA {m['MOTA']:8.2f} HOTA {m['HOTA']:6.2f} "
                      f"IDF1 {m['IDF1']:6.2f} IDS {int(m['IDS']):4d} "
                      f"FP {int(m['FP']):6d} FN {int(m['FN']):6d} "
                      f"trk {float(s['mean_tracks']):5.1f} "
                      f"SCI {float(s['mean_sci']):.3f} "
                      f"rel {float(s['mean_reliability']):.3f} "
                      f"{'PASS' if ok else 'fail'}")
        print(f"{name:<14}=> {'ALL PASS' if ok_all else 'FAIL'}\n")


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
