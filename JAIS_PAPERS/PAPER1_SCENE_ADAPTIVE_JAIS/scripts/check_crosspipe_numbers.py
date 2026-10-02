"""Rebuilds every cross-pipeline number quoted in Sec. XIII and Table 8 from the source JSON files and checks
that each string occurs verbatim in manuscript.tex or tables/tab8_crosspipeline.tex. Exit status 1 on any miss."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
M = (HERE / "manuscript.tex").read_text() + (HERE / "tables/tab8_crosspipeline.tex").read_text()
RS = ROOT / "research/transfer_legacy/rescore"
u = json.load(open(RS / "u2mot/u2mot_testdev_rescore.json"))
s = json.load(open(RS / "sparsetrack/sparsetrack_val_half_rescore.json"))
a = json.load(open(ROOT / "research/paper_split/evidence/legacy/MATCHED_STATIC_A0_TESTDEV.json"))
assert u["reproduction_all_match"] is True and all(x["all_match"] is True for x in s["reproduction"].values())

uh = u["paired_controller_vs_baseline"]["metrics"]["HOTA"]
sh = s["pairs"]["adaptive_vs_static075"]["metrics"]["HOTA"]
sd = s["pairs"]["adaptive_vs_static070"]["metrics"]["HOTA"]
ah = a["results_V1_minus_A0_pp"]["HOTA"]
U, S = u["systems"], s["systems"]


def txt(x):
    return f"$\\Delta={x['delta']:+.2f}$, $[{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]$"


def tab(x):
    return f"${x['delta']:+.2f}$ & [${x['ci95'][0]:+.2f}$, ${x['ci95'][1]:+.2f}$] & {x['wins']}/{x['ties']}/{x['losses']}"


checks = {
    "U2MOT dHOTA (text)": txt(uh),
    "U2MOT dHOTA (table)": tab(uh),
    "U2MOT LOO": f"${uh['loo_min']:+.2f}$ to ${uh['loo_max']:+.2f}$",
    "U2MOT HOTA": f"{U['controller']['trackeval_overall']['HOTA']:.2f}\\% against {U['baseline']['trackeval_overall']['HOTA']:.2f}\\%",
    "U2MOT motmetrics MOTA": f"MOTA was {U['controller']['motmetrics_overall']['MOTA']:.2f}\\% against {U['baseline']['motmetrics_overall']['MOTA']:.2f}\\%",
    "U2MOT motmetrics IDF1": f"IDF1 {U['controller']['motmetrics_overall']['IDF1']:.2f}\\% against {U['baseline']['motmetrics_overall']['IDF1']:.2f}\\%",
    "SparseTrack dHOTA vs 0.75 (text)": txt(sh),
    "SparseTrack dHOTA vs 0.75 (table)": tab(sh),
    "SparseTrack LOO": f"${sh['loo_min']:+.2f}$ to ${sh['loo_max']:+.2f}$",
    "SparseTrack HOTA": f"{S['adaptive']['trackeval_overall']['HOTA']:.2f}\\% against {S['static075']['trackeval_overall']['HOTA']:.2f}\\%",
    "SparseTrack motmetrics MOTA": f"MOTA was {S['adaptive']['motmetrics_overall']['MOTA']:.2f}\\% against {S['static075']['motmetrics_overall']['MOTA']:.2f}\\%",
    "SparseTrack motmetrics IDF1": f"IDF1 {S['adaptive']['motmetrics_overall']['IDF1']:.2f}\\% against {S['static075']['motmetrics_overall']['IDF1']:.2f}\\%",
    "SparseTrack default 0.70 HOTA": f"reached {S['static070']['trackeval_overall']['HOTA']:.2f}\\%",
    "SparseTrack vs 0.70 (text)": txt(sd) + f"; {sd['wins']} wins, no ties, {sd['losses']} losses",
    "SparseTrack vs 0.70 (note c)": f"${sd['delta']:+.2f}$ [${sd['ci95'][0]:+.2f}$, ${sd['ci95'][1]:+.2f}$], W/T/L {sd['wins']}/{sd['ties']}/{sd['losses']}",
    "Row 1 dHOTA": f"${ah['delta']:+.2f}$ & [${ah['ci_low']:+.2f}$, ${ah['ci_high']:+.2f}$]",
    "Row 1 settings": f"conf.\\ {a['A0_system']['mean_conf']:.2f}, NMS {a['A0_system']['mean_nms_iou']:.2f}, {a['A0_system']['mean_imgsz']:.0f} px",
}
miss = 0
for k, v in checks.items():
    ok = v in M
    miss += not ok
    print("OK  " if ok else "MISS", k, "|", v)
sys.exit(1 if miss else 0)
