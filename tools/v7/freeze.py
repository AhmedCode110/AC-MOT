"""
Write the V7 freeze artefacts for a registered system (single source of truth
for the frozen policy + integrity lock).

  python tools/v7/freeze.py <system>        # e.g. V7f

-> configs/universal_acmot_policy_v7.json   (resolved V7Spec, every field)
-> research/V7_POLICY_LOCK.json             (sha256 of the layer, the config,
                                             the registry and the runners)
The tag `universal-acmot-v7-freeze` is created on the commit that adds them.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LOCKED = ["acmot_v7.py", "online_calibration.py", "configs/universal_acmot_policy_v7.json",
          "tools/v7/systems.py", "tools/v7/dev.py", "tools/v7/external/boosttrack_v7.py",
          "tools/v7/external/sparsetrack_v7.py", "tools/v7/external/mot17_bytetrack_v7.py",
          "tools/v7/kitti/kitti_eval.py", "tools/v7/bootstrap.py"]


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main(system):
    from acmot_v7 import spec_from_dict
    from tools.v7.systems import SYSTEMS
    spec = spec_from_dict(dict(SYSTEMS[system], name=system))
    cfg = dict(policy="Universal AC-MOT V7", system=system, spec=asdict(spec),
               host_contract_fields=["assoc", "birth", "low", "match", "cmc"],
               notes="Training-free, online, causal. The layer reads only the candidate list, one "
                     "image motion cue (optional) and the host's output tracks, plus the host's "
                     "declared operating point. No detector, tracker, dataset or sequence names.")
    dst = ROOT / "configs/universal_acmot_policy_v7.json"
    dst.write_text(json.dumps(cfg, indent=1) + "\n")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    lock = dict(policy="configs/universal_acmot_policy_v7.json", system=system,
                status="FROZEN at tag universal-acmot-v7-freeze (commit = the tagged commit)",
                parent_commit=head,
                file_sha256={p: sha(p) for p in LOCKED},
                v6_lock_untouched=True)
    (ROOT / "research/V7_POLICY_LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
    print(json.dumps(cfg["spec"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
