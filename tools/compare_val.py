"""Validation comparison: baselines, ablations, frozen Universal AC-MOT and
the detector-specific oracle (in-sample and LOSO)."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from tools.seqstats import combine

DATASET = Path("/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val")
DETS = ["yolov8", "rtdetr"]
SEQS = sorted(p.name for p in (DATASET / "sequences").iterdir() if p.is_dir())


def load(root, cid, det, seq):
    return pickle.load(open(Path(root) / cid / det / f"{seq}.pkl", "rb"))


def m(root, cid, det, seqs):
    return combine([load(root, cid, det, s) for s in seqs])


def n_cat(root, cid, det, seqs):
    return sum(m(root, cid, det, [s])["MOTA"] < 0 for s in seqs)


def oracle_choice(det, seqs, root="outputs/baselines_val"):
    ids = [c["id"] for c in json.load(open(Path(root) / "configs.json"))
           if c["id"].startswith("oracle")]
    return max(ids, key=lambda c: (-n_cat(root, c, det, seqs),
                                   0.5 * (m(root, c, det, seqs)["HOTA"]
                                          + m(root, c, det, seqs)["IDF1"])))


def row(name, per):
    out = [f"{name:<34}"]
    for d in DETS:
        x = per[d]
        out.append(f"{x['MOTA']:7.2f} {x['HOTA']:6.2f} {x['IDF1']:6.2f} "
                   f"{x['IDS']:5d} {x['FP']:6d} {x['FN']:6d} "
                   f"{x['Precision']:5.1f} {x['Recall']:5.1f}")
    return " | ".join(out)


def main():
    O, B = "outputs/opt/stats", "outputs/baselines_val"
    print(f"{'system':<34}  " + " | ".join(
        f"{d}: MOTA   HOTA   IDF1   IDS     FP     FN     P     R"
        for d in DETS))
    rows = [
        ("Default (raw, ByteTrack defaults)", B, "default"),
        ("V1 normalization only", O, "gate_r0.0"),
        ("V2cA density Top-K (k=1)", O, "dens_k1.0_r0.0"),
        ("Gate + density (k=2, r=0.5)", O, "dens_k2.0_r0.5"),
        ("UNIVERSAL AC-MOT (gate r=0.6)", O, "gate_r0.6"),
    ]
    for name, root, cid in rows:
        print(row(name, {d: m(root, cid, d, SEQS) for d in DETS}))
    # V2b from full runs (separate pipeline outputs)
    # Oracle
    ins = {d: oracle_choice(d, SEQS) for d in DETS}
    print(row(f"ORACLE in-sample {ins['yolov8'][7:]}/{ins['rtdetr'][7:]}",
              {d: m(B, ins[d], d, SEQS) for d in DETS}))
    loso = {}
    for d in DETS:
        chosen = {s: oracle_choice(d, [t for t in SEQS if t != s])
                  for s in SEQS}
        loso[d] = combine([load(B, chosen[s], d, s) for s in SEQS])
        print(f"   oracle LOSO choices {d}: {sorted(set(chosen.values()))}")
    print(row("ORACLE detector-specific LOSO", loso))


if __name__ == "__main__":
    main()
