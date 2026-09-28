"""
Collect every V7 development result into one JSON (reads cached evaluation
files only; no tracker is run). Output: research/final/V7_DEV_RESULTS.json
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXT = Path("/Users/ahmedgouda/Desktop/acmot_external/runs")


def visdrone(split, dets):
    from tools.v7 import dev
    out = {}
    base = ROOT / "outputs/v7" / split
    systems = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []
    refs = {"val7": ["V6:X5", "V6:V4", "V6:shared_static", "V6:static_default"],
            "dev40": ["V6:X5", "V6:V4"]}.get(split, [])
    for sy in systems + refs:
        for d in dets:
            try:
                with redirect_stdout(io.StringIO()):
                    r = dev.summary(split, sy, [d])[d]
            except Exception:
                continue
            out.setdefault(sy, {})[d] = {k: (round(v, 3) if isinstance(v, float) else v)
                                        for k, v in r.items() if k in
                                        ("MOTA", "HOTA", "IDF1", "IDS", "FP", "FN", "Precision", "Recall",
                                         "clean_frac")} | {"catastrophic": r["cat"]}
    return out


def mot17(host):
    sys.path.insert(0, str(ROOT / "tools/v6/external"))
    import mot17_eval as me
    root = EXT / host
    out = {}
    for p in sorted((root / "MOT17-val").iterdir()):
        if not (p / "data").exists():
            continue
        try:
            with redirect_stdout(io.StringIO()):
                per = me.evaluate(str(root), p.name)
                pooled = me.combine([per[s] for s in me.SEQS])
        except Exception as e:
            out[p.name] = {"error": str(e)[:200]}
            continue
        out[p.name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pooled.items()}
    return out


if __name__ == "__main__":
    res = {"visdrone_val7_yolo_rtdetr": visdrone("val7", ["yolov8", "rtdetr"]),
           "visdrone_val7_fasterrcnn": visdrone("val7", ["fasterrcnn"]),
           "visdrone_dev40": visdrone("dev40", ["yolov8", "rtdetr"]),
           "mot17_sparsetrack": mot17("sparsetrack"),
           "mot17_boosttrack": mot17("boosttrack")}
    dest = ROOT / "research/final/V7_DEV_RESULTS.json"
    json.dump(res, open(dest, "w"), indent=1)
    for k, v in res.items():
        print(k, len(v))
