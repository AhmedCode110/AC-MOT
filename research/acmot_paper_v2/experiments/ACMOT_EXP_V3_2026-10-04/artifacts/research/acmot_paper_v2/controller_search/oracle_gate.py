"""
STAGE 0 -- oracle/headroom gate. Calibration data only (8 sequences).
Evaluates all 9 static (resolution, NMS) points, ByteTrack host, official
evaluator. See PREDECLARATION.json for the exact procedure/decision rule.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.acmot_paper_v2.controller_search.sim import (  # noqa: E402
    CALIBRATION, RESOLUTIONS, NMS_VALUES, run_static_on_sequence, score_sequence,
)
from tools.v6.eval_official import combine_official  # noqa: E402

OUT = Path(__file__).resolve().parent / "ORACLE_GATE_RESULT.json"


def main():
    t0 = time.time()
    per_point_per_seq = {}  # (res,nms) -> {seq: pooled_hota_for_that_seq_alone}
    per_point_pooled = {}   # (res,nms) -> pooled HOTA across all 8 seqs
    raw_stats = {}          # (res,nms) -> {seq: official_sequence_stats output}
    for res in RESOLUTIONS:
        for nms in NMS_VALUES:
            key = f"r{res}_n{int(round(100*nms))}"
            raw_stats[key] = {}
            for seq in CALIBRATION:
                tr = run_static_on_sequence("visdrone_calib", seq, res, nms, "bytetrack")
                st = score_sequence("visdrone_calib", seq, tr)
                raw_stats[key][seq] = st
                print(key, seq, "HOTA(seq-alone)=", combine_official([st])["HOTA"], "elapsed", round(time.time()-t0, 1))
            per_point_per_seq[key] = {seq: combine_official([raw_stats[key][seq]])["HOTA"] for seq in CALIBRATION}
            per_point_pooled[key] = combine_official(list(raw_stats[key].values()))["HOTA"]

    matched_static_key = max(per_point_pooled, key=per_point_pooled.get)
    matched_static_pooled_hota = per_point_pooled[matched_static_key]
    matched_static_per_seq = per_point_per_seq[matched_static_key]

    oracle_per_seq = {seq: max(per_point_per_seq[k][seq] for k in per_point_per_seq) for seq in CALIBRATION}
    oracle_best_point_per_seq = {
        seq: max(per_point_per_seq, key=lambda k: per_point_per_seq[k][seq]) for seq in CALIBRATION
    }

    oracle_mean = sum(oracle_per_seq.values()) / len(oracle_per_seq)
    matched_static_mean_per_seq = sum(matched_static_per_seq.values()) / len(matched_static_per_seq)
    headroom = oracle_mean - matched_static_mean_per_seq

    result = dict(
        wall_seconds=time.time() - t0,
        per_point_pooled_hota=per_point_pooled,
        matched_static=dict(key=matched_static_key, pooled_hota=matched_static_pooled_hota,
                             per_seq_hota=matched_static_per_seq, mean_per_seq_hota=matched_static_mean_per_seq),
        oracle=dict(per_seq_hota=oracle_per_seq, best_point_per_seq=oracle_best_point_per_seq, mean_per_seq_hota=oracle_mean),
        headroom=headroom,
        decision="PROCEED to Stage 1/2 search" if headroom > 0 else "NULL RESULT -- no headroom, do not build an adaptive controller",
    )
    json.dump(result, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "per_point_pooled_hota"}, indent=1, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
