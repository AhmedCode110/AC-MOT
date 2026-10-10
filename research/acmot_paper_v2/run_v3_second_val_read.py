"""
SECOND VAL READ for the frozen v3 genuinely-adaptive AC-MOT controller
(FREEZE_MANIFEST_V3_ADAPTIVE.json). VisDrone-val was already read once
for the static/no-op method (commit bfa5caf) -- this is explicitly the
SECOND read, never described as untouched held-out confirmation.

  python research/acmot_paper_v2/run_v3_second_val_read.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from acmot_sci import SceneSpec  # noqa: E402
from research.acmot_paper_v2.controller_search.sim import VAL_SEQS, score_sequence  # noqa: E402
from research.acmot_paper_v2.controller_search.run_v3_joint_search import run_policy_with_conf, spec_for  # noqa: E402
from tools.v6.eval_official import combine_official  # noqa: E402

ACTION_MAP = {"LOW": (1536, 0.70, 0.10), "MEDIUM": (1280, 0.45, 0.40), "HIGH": (1088, 0.60, 0.40)}
SPEC = spec_for(0.29747709701135766, 0.5114928923560124)
LABEL = "SECOND_VAL_READ"


def switches_per_100(levels):
    if len(levels) < 2:
        return 0.0
    return 100.0 * sum(1 for i in range(1, len(levels)) if levels[i] != levels[i - 1]) / len(levels)


def run_system(host: str):
    per_seq, per_seq_levels = [], {}
    t0 = time.time()
    for seq in VAL_SEQS:
        tr, levels = run_policy_with_conf("visdrone_val", seq, ACTION_MAP, host, SPEC)
        stats = score_sequence("visdrone_val", seq, tr)
        per_seq.append(stats)
        per_seq_levels[seq] = levels
        print(LABEL, host, seq, "done elapsed", round(time.time() - t0, 1))
    agg = combine_official(per_seq)
    agg["wall_seconds"] = time.time() - t0
    level_counts = {lv: sum(l.count(lv) for l in per_seq_levels.values()) for lv in ("LOW", "MEDIUM", "HIGH")}
    total = sum(level_counts.values())
    level_frac = {lv: c / total for lv, c in level_counts.items()}
    switches = {seq: switches_per_100(l) for seq, l in per_seq_levels.items()}
    print(LABEL, host, "AGGREGATE:", json.dumps(agg, indent=1))
    return per_seq, agg, level_frac, switches


def main():
    results = {"_label": LABEL, "_note": "VisDrone-val was already read once for the static/no-op method "
               "(commit bfa5caf). This is the SECOND read, for the frozen v3 adaptive controller."}
    seq3, agg3, lf3, sw3 = run_system("bytetrack")
    seq4, agg4, lf4, sw4 = run_system("oatrack")
    results["AC-MOT-v3+YOLO11m+ByteTrack"] = dict(aggregate=agg3, per_sequence=dict(zip(VAL_SEQS, seq3)),
                                                   level_frac=lf3, switches_per_100_frames=sw3)
    results["AC-MOT-v3+YOLO11m+OATrack"] = dict(aggregate=agg4, per_sequence=dict(zip(VAL_SEQS, seq4)),
                                                 level_frac=lf4, switches_per_100_frames=sw4)
    results["meta"] = dict(action_map=ACTION_MAP, val_sequences=VAL_SEQS,
                            freeze_manifest="research/acmot_paper_v2/controller_search/FREEZE_MANIFEST_V3_ADAPTIVE.json")
    out_path = ROOT / "research/acmot_paper_v2/SECOND_VAL_READ_V3_RESULT.json"
    json.dump(results, open(out_path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
