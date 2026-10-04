"""
Confidence-floor oracle check (calibration only): now that confidence-
floor filtering on the cache is confirmed exact (CONFIDENCE_EXACTNESS_CHECK_FIXED.json,
18/18), check whether adding confidence as a third detector-action
dimension changes the headroom picture found by the (confidence-blind)
Stages 1/2/3A/3B+4. Resolution/NMS fixed at matched-static (1088,0.45);
confidence floor swept over a predeclared grid.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.acmot_paper_v2.controller_search.sim import CALIBRATION, run_static_on_sequence, score_sequence  # noqa: E402
from tools.v6.eval_official import combine_official  # noqa: E402

CONF_FLOORS = [0.01, 0.10, 0.25, 0.40]  # 0.01 = cache floor, i.e. no extra filtering (current baseline)
RES, NMS = 1088, 0.45


def main():
    t0 = time.time()
    per_floor_per_seq = {}
    per_floor_pooled = {}
    for cf in CONF_FLOORS:
        stats = []
        per_seq = {}
        for seq in CALIBRATION:
            tr = run_static_on_sequence("visdrone_calib", seq, RES, NMS, "bytetrack", conf_floor=cf)
            st = score_sequence("visdrone_calib", seq, tr)
            stats.append(st)
            per_seq[seq] = combine_official([st])["HOTA"]
            print("conf_floor", cf, seq, "HOTA", per_seq[seq], "elapsed", round(time.time() - t0, 1))
        per_floor_per_seq[cf] = per_seq
        per_floor_pooled[cf] = combine_official(stats)["HOTA"]

    best_floor = max(per_floor_pooled, key=per_floor_pooled.get)
    baseline_floor = 0.01
    oracle_per_seq = {seq: max(per_floor_per_seq[cf][seq] for cf in CONF_FLOORS) for seq in CALIBRATION}
    oracle_mean = sum(oracle_per_seq.values()) / len(oracle_per_seq)
    baseline_mean = sum(per_floor_per_seq[baseline_floor].values()) / len(CALIBRATION)
    headroom = oracle_mean - baseline_mean

    result = dict(conf_floors=CONF_FLOORS, res=RES, nms=NMS, per_floor_pooled_hota=per_floor_pooled,
                  per_floor_per_seq_hota=per_floor_per_seq, best_floor_pooled=best_floor,
                  baseline_floor=baseline_floor, baseline_mean_per_seq=baseline_mean,
                  oracle_per_seq=oracle_per_seq, oracle_mean_per_seq=oracle_mean, headroom=headroom,
                  wall_seconds=time.time() - t0)
    OUT = Path(__file__).resolve().parent / "CONFIDENCE_ORACLE_RESULT.json"
    json.dump(result, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if "per_seq" not in k}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
