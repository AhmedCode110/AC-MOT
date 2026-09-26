"""Re-evaluate saved policy runs with the reference evaluator (motmetrics +
TrackEval HOTA) and print a compact table."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from tools.eval_local import evaluate

DATASET = Path("/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val")


def main(run_dirs):
    for run in map(Path, run_dirs):
        seqs = sorted(p.stem for p in (run / "tracks").glob("*.txt"))
        rows = evaluate(DATASET, run / "tracks", seqs)
        with open(run / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(f"{run.name:<28} {r['sequence'][:12]:<12} MOTA {r['MOTA']:8.3f} "
                  f"HOTA {r['HOTA']:6.3f} IDF1 {r['IDF1']:6.3f} IDS {r['IDS']:4d} "
                  f"FP {r['FP']:6d} FN {r['FN']:6d} P {r['Precision']:6.2f} "
                  f"R {r['Recall']:6.2f} [{r['IDS_source']}]")


if __name__ == "__main__":
    main(sys.argv[1:])
