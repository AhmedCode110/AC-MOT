"""
FINAL HELD-OUT TEST, systems 3 and 4: AC-MOT (frozen as a no-op static
policy per FREEZE_MANIFEST.json: resolution=1088, NMS=0.45,
confidence>=0.40) + YOLO11m + {ByteTrack, OATrack}, on the untouched
7-sequence VisDrone-val. This is the ONLY read of val performed for
AC-MOT development; the policy was frozen before this script ever ran.

  python research/acmot_paper_v2/run_final_acmot_systems.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.acmot_paper_v2.controller_search.sim import (  # noqa: E402
    VAL_SEQS, run_static_on_sequence, score_sequence,
)
from tools.v6.eval_official import combine_official  # noqa: E402

FROZEN = dict(resolution=1088, nms=0.45, confidence_floor=0.40)


def run_system(name: str, host: str):
    per_seq = []
    t0 = time.time()
    for seq in VAL_SEQS:
        tr = run_static_on_sequence("visdrone_val", seq, FROZEN["resolution"], FROZEN["nms"], host,
                                     conf_floor=FROZEN["confidence_floor"])
        stats = score_sequence("visdrone_val", seq, tr)
        per_seq.append(stats)
        print(name, seq, "done", "elapsed", round(time.time() - t0, 1))
    agg = combine_official(per_seq)
    agg["wall_seconds"] = time.time() - t0
    print(name, "AGGREGATE:", json.dumps(agg, indent=1))
    return per_seq, agg


def main():
    results = {}
    seq3, agg3 = run_system("AC-MOT(frozen-nop)+YOLO11m+ByteTrack", "bytetrack")
    seq4, agg4 = run_system("AC-MOT(frozen-nop)+YOLO11m+OATrack", "oatrack")
    results["AC-MOT+YOLO11m+ByteTrack"] = dict(aggregate=agg3, per_sequence=dict(zip(VAL_SEQS, seq3)))
    results["AC-MOT+YOLO11m+OATrack"] = dict(aggregate=agg4, per_sequence=dict(zip(VAL_SEQS, seq4)))
    results["meta"] = dict(frozen_policy=FROZEN, val_sequences=VAL_SEQS,
                            freeze_manifest="research/acmot_paper_v2/controller_search/FREEZE_MANIFEST.json",
                            note="AC-MOT frozen as a documented no-op (constant operating point); "
                                 "systems 3/4 differ from systems 1/2 only in using the calibration-selected "
                                 "matched-static point (1088/0.45/conf>=0.40) instead of systems 1/2's "
                                 "independently predeclared default (1536/0.70/no confidence floor).")
    out_path = ROOT / "research/acmot_paper_v2/FINAL_ACMOT_SYSTEMS_RESULT.json"
    json.dump(results, open(out_path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
