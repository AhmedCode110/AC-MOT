"""Read-only data access for the JRTIP manuscript.

No evaluation is run here.  Values are loaded from frozen JSON/CSV records or
computed as explicit differences of their stored aggregates.
"""
from __future__ import annotations
import json
from pathlib import Path

# data.py lives at <repo>/research/acmot_paper_v2/manuscript_jrtip/scripts.
# Four parent steps reach the repository root; no experiment is run here.
ROOT = Path(__file__).resolve().parents[4]
PAPER = ROOT / "research" / "acmot_paper_v2"
LEGACY = ROOT / "research" / "paper_split" / "evidence" / "legacy"

def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def modern():
    return {
        "baseline": load_json(PAPER / "BASELINE_SYSTEMS_RESULT.json"),
        "v3": load_json(PAPER / "SECOND_VAL_READ_V3_RESULT.json"),
        "bootstrap": load_json(PAPER / "BOOTSTRAP_V3_RESULT.json"),
        "runtime": load_json(PAPER / "V3_RUNTIME_BENCHMARK_RESULT.json"),
        "uavdt": load_json(PAPER / "UAVDT_TRANSFER_RESULT.json"),
        "freeze": load_json(PAPER / "experiments" / "ACMOT_EXP_V3_2026-10-04" / "FREEZE_MANIFEST.json"),
    }

def legacy():
    return {
        "test": load_json(LEGACY / "FINAL_TEST_RESULTS_3WORKER.json"),
        "uavdt": load_json(LEGACY / "UAVDT_FINAL_COMPARISON.json"),
    }

if __name__ == "__main__":
    d = modern()
    print("Frozen v3 action map:", d["freeze"]["action_mapping"])
    print("Modern result records loaded:", ", ".join(sorted(d)))
