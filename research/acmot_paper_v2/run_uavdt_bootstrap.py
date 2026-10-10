"""
Paired sequence bootstrap for the frozen UAVDT transfer (v3 adaptive vs
r1536_n70), computed from the per-sequence counts already stored in
UAVDT_TRANSFER_RESULT.json. No tracker, detector or scorer is run.

Same procedure as the VisDrone bootstrap (BOOTSTRAP_V3_RESULT.json):
tools/v7/bootstrap.paired, 5000 resamples, numpy default_rng(42), the same
sequence indices for both systems, pooled metrics recomputed on every
resample, percentile 95% interval of (v3 - baseline).

Metrics: the count-based metrics of tools/seqstats.combine (MOTA, IDF1, IDS,
FP, FN, Precision, Recall). HOTA is not included: the transfer run stored
per-sequence motmetrics counts only, not the per-sequence HOTA results that
HOTA pooling needs.

  python research/acmot_paper_v2/run_uavdt_bootstrap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.v7.bootstrap import paired  # noqa: E402

SRC = ROOT / "research/acmot_paper_v2/UAVDT_TRANSFER_RESULT.json"
OUT = ROOT / "research/acmot_paper_v2/UAVDT_BOOTSTRAP_V3_RESULT.json"
KEYS = ["MOTA", "IDF1", "IDS", "FP", "FN", "Precision", "Recall"]
N, SEED = 5000, 42


def combine_counts(cells):
    """Count-based part of tools/seqstats.combine, same formulas."""
    c = {k: sum(x[k] for x in cells) for k in cells[0]}
    gt = c["num_objects"]
    tp = gt - c["num_misses"]
    return dict(
        MOTA=100 * (1 - (c["num_misses"] + c["num_false_positives"] + c["num_switches"]) / gt),
        IDF1=100 * 2 * c["idtp"] / max(1, 2 * c["idtp"] + c["idfp"] + c["idfn"]),
        IDS=c["num_switches"], FP=c["num_false_positives"], FN=c["num_misses"],
        Precision=100 * tp / max(1, tp + c["num_false_positives"]),
        Recall=100 * tp / max(1, gt),
    )


def main():
    src = json.load(open(SRC))
    res = dict(_label="UAVDT_TRANSFER_BOOTSTRAP",
               _note="paired sequence bootstrap of the frozen UAVDT transfer, computed from the stored "
                     "per-sequence counts of UAVDT_TRANSFER_RESULT.json (no rerun); HOTA not included "
                     "(per-sequence HOTA results were not stored)",
               source=str(SRC.relative_to(ROOT)), n=N, seed=SEED, metrics=KEYS)
    for host in ("bytetrack", "oatrack"):
        base = src[f"baseline_r1536_n70+{host}"]
        v3 = src[f"v3_adaptive+{host}"]
        seqs = sorted(base["per_sequence"])
        assert seqs == sorted(v3["per_sequence"]) and len(seqs) == 20
        A = [base["per_sequence"][s] for s in seqs]
        B = [v3["per_sequence"][s] for s in seqs]
        # the pooled counts must reproduce the stored aggregates exactly
        for cells, agg in ((A, base["aggregate"]), (B, v3["aggregate"])):
            m = combine_counts(cells)
            for k in KEYS:
                assert abs(m[k] - agg[k]) < 1e-9, (host, k, m[k], agg[k])
        res[f"{host}_v3_vs_r1536_n70"] = paired(A, B, combine_counts, N, SEED, KEYS)
        res[f"{host}_sequences"] = seqs
    json.dump(res, open(OUT, "w"), indent=1)
    for host in ("bytetrack", "oatrack"):
        print(host)
        for k, x in res[f"{host}_v3_vs_r1536_n70"].items():
            print(f"  {k:<9} {x['A']:10.3f} -> {x['B']:10.3f}  d {x['diff']:+10.3f} "
                  f"[{x['ci_lo']:+10.3f}, {x['ci_hi']:+10.3f}]  seq +{x['seq_wins']}/-{x['seq_losses']}")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
