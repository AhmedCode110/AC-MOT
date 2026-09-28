"""
Relocatable wrapper around the V6 MOT17 evaluator (tools/v6/external/
mot17_eval.py, used read-only): TrackEval and the MOT17 val-half GT are taken
from ACMOT_TRACKEVAL / ACMOT_MOT17_GT when set (cloud / Codex), otherwise the
original Mac paths.

  python tools/v7/mot17_eval_v7.py <runs_root> <name> [...]
  python tools/v7/mot17_eval_v7.py --boot <runs_root> <A> <B>
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_mot17_eval():
    te = os.environ.get("ACMOT_TRACKEVAL")
    if te:
        sys.path.insert(0, te)
    spec = importlib.util.spec_from_file_location("mot17_eval", ROOT / "tools/v6/external/mot17_eval.py")
    me = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(me)
    if os.environ.get("ACMOT_MOT17_GT"):
        me.GT = Path(os.environ["ACMOT_MOT17_GT"])
    return me


if __name__ == "__main__":
    me = load_mot17_eval()
    if sys.argv[1] == "--boot":
        print(json.dumps(me.bootstrap(*sys.argv[2:5]), indent=1))
    else:
        root, *names = sys.argv[1:]
        for n, v in me.table(root, names).items():
            p = v["pooled"]
            print(f"{n:<34} HOTA {p['HOTA']:6.3f} MOTA {p['MOTA']:6.3f} IDF1 {p['IDF1']:6.3f} "
                  f"IDS {p['IDS']:5d} FP {p['FP']:6d} FN {p['FN']:6d} P {p['Precision']:6.2f} "
                  f"R {p['Recall']:6.2f} AssA {p['AssA']:6.2f} DetA {p['DetA']:6.2f}")
