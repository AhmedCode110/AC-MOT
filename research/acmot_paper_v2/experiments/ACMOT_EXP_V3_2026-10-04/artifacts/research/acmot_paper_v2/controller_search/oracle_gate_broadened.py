"""
Broadened oracle/headroom gate: 15 static points (resolutions
{896,1088,1280,1536,1728} x NMS {0.45,0.60,0.70}), 8 calibration
sequences, ByteTrack host. Per PREDECLARATION_BROADENED.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.acmot_paper_v2.controller_search.sim import (  # noqa: E402
    CALIBRATION, NMS_VALUES, run_static_on_sequence, score_sequence,
)
from tools.v6.eval_official import combine_official  # noqa: E402

OUT = Path(__file__).resolve().parent / "ORACLE_GATE_BROADENED_RESULT.json"
RESOLUTIONS = [896, 1088, 1280, 1536, 1728]


def main():
    t0 = time.time()
    per_point_per_seq = {}
    per_point_pooled = {}
    raw_stats = {}
    for res in RESOLUTIONS:
        for nms in NMS_VALUES:
            key = f"r{res}_n{int(round(100*nms))}"
            raw_stats[key] = {}
            for seq in CALIBRATION:
                tr = run_static_on_sequence("visdrone_calib", seq, res, nms, "bytetrack")
                st = score_sequence("visdrone_calib", seq, tr)
                raw_stats[key][seq] = st
                print(key, seq, "HOTA=", combine_official([st])["HOTA"], "elapsed", round(time.time()-t0, 1))
            per_point_per_seq[key] = {seq: combine_official([raw_stats[key][seq]])["HOTA"] for seq in CALIBRATION}
            per_point_pooled[key] = combine_official(list(raw_stats[key].values()))["HOTA"]

    matched_static_key = max(per_point_pooled, key=per_point_pooled.get)
    matched_static_per_seq = per_point_per_seq[matched_static_key]
    oracle_per_seq = {seq: max(per_point_per_seq[k][seq] for k in per_point_per_seq) for seq in CALIBRATION}
    oracle_best_point_per_seq = {seq: max(per_point_per_seq, key=lambda k: per_point_per_seq[k][seq]) for seq in CALIBRATION}

    oracle_mean = sum(oracle_per_seq.values()) / len(oracle_per_seq)
    matched_static_mean = sum(matched_static_per_seq.values()) / len(matched_static_per_seq)
    headroom = oracle_mean - matched_static_mean
    n_seqs_with_gap = sum(1 for seq in CALIBRATION if oracle_per_seq[seq] - matched_static_per_seq[seq] > 1e-6)

    ORIGINAL_HEADROOM = 0.17760943168585897
    if headroom <= ORIGINAL_HEADROOM and n_seqs_with_gap <= 2:
        decision = ("NO RE-RUN of staged search on broadened space -- broadened headroom "
                    "({:.4f}) does not exceed the narrow-space headroom ({:.4f}) and remains "
                    "concentrated in <=2 sequences. Narrow-space staged results stand as the "
                    "complete controller-development evidence, per the predeclared rule.").format(headroom, ORIGINAL_HEADROOM)
    else:
        decision = "RE-RUN staged search (Stage1/2/3A/3B/Stage4 pattern) on the broadened action space."

    result = dict(
        wall_seconds=time.time() - t0, resolutions=RESOLUTIONS, nms=NMS_VALUES,
        per_point_pooled_hota=per_point_pooled,
        matched_static=dict(key=matched_static_key, per_seq_hota=matched_static_per_seq, mean_per_seq_hota=matched_static_mean),
        oracle=dict(per_seq_hota=oracle_per_seq, best_point_per_seq=oracle_best_point_per_seq, mean_per_seq_hota=oracle_mean),
        headroom=headroom, n_seqs_with_gap=n_seqs_with_gap, original_narrow_headroom=ORIGINAL_HEADROOM,
        decision=decision,
    )
    json.dump(result, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "per_point_pooled_hota"}, indent=1, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
