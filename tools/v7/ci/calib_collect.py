"""
Collector for tools/v7/ci/calib_boot.sh (Paper 2, post-freeze analysis only).
For each of the 14 calibration-shift conditions of ledger STRESS-L: pooled
and per-sequence TrackEval metrics of NATIVE (host alone on the transformed
stream) and V7f, a replay identity check against the pooled values recorded
in research/final/V7_DEV_RESULTS.json, the paired sequence bootstrap of
(V7f - NATIVE) (10,000 resamples, seed 42, percentile 95% CI; the frozen
project protocol of tools/v6/external/mot17_eval.py), and sequence
wins / ties / losses on HOTA (tie: |dHOTA| < 0.01).

  python calib_collect.py <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/v7"))
from mot17_eval_v7 import load_mot17_eval  # noqa: E402

EXT = Path(os.environ["ACMOT_EXT"])
BY_ROOT = EXT / "runs/bytetrack_mot17"
BT_ROOT = EXT / "runs/boosttrack"
# (host label, runs root, NATIVE run, V7f run, recorded key group, transform)
CONDITIONS = []
for label, pre in (("ByteTrack (official setting)", "BY_official"),
                   ("ByteTrack (ultralytics setting)", "BY_ultra"), ("OC-SORT", "OC")):
    for tf in ("pow3", "scale05", "temp2", "temp05"):
        CONDITIONS.append((label, BY_ROOT, f"{pre}_st_NATIVE_t_{tf}", f"{pre}_st_V7f_t_{tf}",
                           "mot17_bytetrack_ocsort_c1", tf))
for tf in ("pow3", "temp2"):
    CONDITIONS.append(("BoostTrack online", BT_ROOT, f"BT7C_NATIVE_{tf}_pf", f"BT7C_V7f_{tf}_pf",
                       "mot17_boosttrack", tf))
CHECK = ["HOTA", "MOTA", "IDF1", "IDS", "FP", "FN"]


def main():
    out = Path(sys.argv[1])
    me = load_mot17_eval()
    rec = json.load(open(ROOT / "research/final/V7_DEV_RESULTS.json"))
    rows = []
    for label, root, a, b, grp, tf in CONDITIONS:
        t = me.table(str(root), [a, b])
        ident = {}
        for n in (a, b):
            got, want = t[n]["pooled"], rec[grp][n]
            ident[n] = all(abs(round(got[k], 3) - want[k]) <= 1e-3 if isinstance(want[k], float)
                           else got[k] == want[k] for k in CHECK)
        boot = me.bootstrap(str(root), a, b)
        per = {s: t[b]["per_seq"][s]["HOTA"] - t[a]["per_seq"][s]["HOTA"] for s in me.SEQS}
        w = sum(v >= 0.01 for v in per.values())
        l = sum(v <= -0.01 for v in per.values())
        same = all(t[a]["per_seq"][s] == t[b]["per_seq"][s] for s in me.SEQS)
        h = boot["HOTA"]
        verdict = ("identical output" if same else "significant recovery" if h["ci_lo"] > 0
                   else "significant degradation" if h["ci_hi"] < 0 else "no significant change")
        rows.append(dict(host=label, transform=tf, native_run=a, v7f_run=b,
                         native=t[a]["pooled"], v7f=t[b]["pooled"],
                         replay_identical_to_record=ident, bootstrap=boot,
                         per_seq_dHOTA=per, wins_ties_losses=[w, 7 - w - l, l], verdict=verdict))
        print(f"{label:<32} {tf:<8} HOTA {t[a]['pooled']['HOTA']:6.2f} -> {t[b]['pooled']['HOTA']:6.2f} "
              f"d {h['diff']:+6.2f} [{h['ci_lo']:+6.2f}, {h['ci_hi']:+6.2f}] "
              f"W/T/L {w}/{7 - w - l}/{l} {verdict:<24} replay-identical {all(ident.values())}", flush=True)
    n_rec = sum(r["verdict"] == "significant recovery" for r in rows)
    n_deg = sum(r["verdict"] == "significant degradation" for r in rows)
    summary = dict(conditions=len(rows), significant_recovery=n_rec, significant_degradation=n_deg,
                   all_replays_identical=all(all(r["replay_identical_to_record"].values()) for r in rows))
    print("SUMMARY", json.dumps(summary))
    json.dump(dict(policy="V7f", freeze_commit="488df9a", protocol="tools/v7/ci/calib_boot.sh (declared rule)",
                   bootstrap=dict(resamples=10000, seed=42, ci="percentile 95%", tie="|dHOTA| < 0.01"),
                   summary=summary, conditions=rows),
              open(out / "calib_boot.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
