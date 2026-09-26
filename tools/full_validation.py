"""Full 7-sequence validation of FROZEN named policies for one detector
(cache replay). Writes each run to outputs/policy_runs/<det>_<name>_full."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.run_policy_validation import run
from universal_policy_pipeline import POLICIES, replace

DATASET = "/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val"

FROZEN = {
    "V1": ("V1", {}),
    "V2b": ("V2b", {}),
    "V2cA": ("V2cA", {}),
    # Leader-relative gate, rho=0.5 declared before diagnostics.
    "V2gA": ("V2cA", {"name": "V2gA", "leader_rho": 0.5}),
    "V1g": ("V1", {"name": "V1g", "leader_rho": 0.5,
                   "feedback": "accepted"}),
}


def main(det, names):
    for name in names:
        base, ov = FROZEN[name]
        policy = replace(POLICIES[base], **ov)
        out = Path(f"outputs/policy_runs/{det}_{name}_full")
        if out.exists():
            print("exists, skip:", out)
            continue
        summary, metrics = run(policy, DATASET, f"outputs/det_cache/{det}",
                               out, quiet=True)
        m = metrics[-1]
        print(f"{det:<7}{name:<6} OVERALL MOTA {m['MOTA']:8.3f} HOTA "
              f"{m['HOTA']:6.3f} IDF1 {m['IDF1']:6.3f} IDS {m['IDS']:5d} "
              f"FP {m['FP']:6d} FN {m['FN']:6d} P {m['Precision']:6.2f} "
              f"R {m['Recall']:6.2f} [{m['IDS_source']}]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
