"""
LaTeX tables of Paper 2 (frozen V7f), generated from scripts/data.py only.
Run from any directory:

  python JAIS_PAPERS/PAPER2_UNIVERSAL_ACMOT_JAIS/scripts/make_tables.py
Also writes tables/text_numbers.json (counts quoted in the text).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

OUT = HERE.parent / "tables"
OUT.mkdir(exist_ok=True)
NUM = {}


def n(x, p=2, sign=True):
    if round(x, p) == 0:
        return f"${0:.{p}f}$"
    s = f"{x:+.{p}f}" if sign else f"{x:.{p}f}"
    return f"${s}$"


def ci(t, p=2, bold=True):
    """(diff, lo, hi) -> '$+0.61$ [$+0.36$, $+1.24$]' (bold when the CI excludes 0)."""
    d, lo, hi = t
    if lo is None:
        return n(d, 0) if float(d).is_integer() else n(d, p)
    if any(0 < abs(v) < 0.5 * 10 ** -p for v in (d, lo, hi)):
        p += 1            # keep the sign of bounds that would round to zero
    s = f"{n(d, p)} [{n(lo, p)}, {n(hi, p)}]"
    return f"{{\\bfseries\\boldmath {s}}}" if bold and (lo > 0 or hi < 0) else s


def bt(b, k, p=2):
    return (b[k]["diff"], b[k]["ci_lo"], b[k]["ci_hi"])


def cnt(t):
    d, lo, hi = t
    return f"{n(d, 0)} [{n(lo, 0)}, {n(hi, 0)}]"


def wtl(per_delta):
    w = sum(v >= 0.01 for v in per_delta)
    l = sum(v <= -0.01 for v in per_delta)
    return f"{w}/{len(per_delta) - w - l}/{l}"


def write(name, s):
    (OUT / name).write_text(s)
    print("wrote", name)


def table(label, caption, cols, header, rows, size="\\footnotesize", resize=True):
    body = "\n".join(rows)
    inner = f"\\begin{{tabular}}{{{cols}}}\n\\hline\n{header} \\\\\n\\hline\n{body}\n\\hline\n\\end{{tabular}}"
    if resize:
        inner = "\\resizebox{\\textwidth}{!}{%\n" + inner + "}"
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n"
            f"{inner}\n\\end{{table}}\n")


# ------------------------------------------------------------------ Table 3: development hosts, MOT17
def tab_development():
    rows = []
    sp = D.MAIN["sparsetrack"]
    b = sp["bootstrap_V7f_minus_reproduction"]
    r0, r1 = sp["reproduction"], sp["V7f"]
    per = [v["v7f"]["HOTA"] - v["base"]["HOTA"] for v in sp["per_sequence"].values()]
    rows.append(" & ".join([
        "SparseTrack", "0.01", "69.2/76.8/81.4", f"{r0['HOTA']:.2f}/{r0['MOTA']:.2f}/{r0['IDF1']:.2f}",
        ci(bt(b, "HOTA")), ci(bt(b, "DetA")), ci(bt(b, "AssA")), ci(bt(b, "MOTA")), ci(bt(b, "IDF1")),
        f"{n(r1['FP'] - r0['FP'], 0)}/{n(r1['FN'] - r0['FN'], 0)}", wtl(per)]) + " \\\\")
    NUM["sparsetrack_wtl"] = wtl(per)
    per_keys = {"ByteTrack official, floor 0.01": "ByteTrack_st", "ByteTrack official, floor 0.1": "ByteTrack_bt",
                "OC-SORT, floor 0.01": "OC-SORT_st", "OC-SORT, floor 0.1": "OC-SORT_bt",
                "BoostTrack online, floor 0.1": "BoostTrack_online"}
    ref = {"ByteTrack official": "--/76.6/79.3", "ByteTrack ultralytics": "n/a", "OC-SORT": "66.5/74.9/77.7",
           "BoostTrack online": "68.37/75.56/81.35"}
    for name, v in D.STAT_MOT17.items():
        host, floor = name.split(", floor ")
        grp = "boost" if v.get("group") == "boost" else "mot"
        p0, p1 = D.pooled(grp, v["base"]), D.pooled(grp, v["v7"])
        if name in per_keys:
            ps = D.MAIN["per_sequence"][per_keys[name]]["per_seq"]
            w = wtl([x["v7f"]["HOTA"] - x["base"]["HOTA"] for x in ps.values()])
        else:
            w = "0/7/0"
        NUM[f"wtl::{name}"] = w
        base = f"{p0['HOTA']:.2f}/{p0['MOTA']:.2f}/{p0['IDF1']:.2f}"
        if v.get("identical"):
            cells = ["identical output"] + [""] * 4 + ["0/0"]
            rows.append(" & ".join([host, floor, ref[host], base, "\\multicolumn{5}{c}{identical output (all sequences)}",
                                    "0/0", w]) + " \\\\")
            continue
        dd = lambda k: (p1[k] - p0[k], None, None)
        rows.append(" & ".join([
            host, floor, ref[host], base, ci(v["HOTA"]), n(p1["DetA"] - p0["DetA"]), n(p1["AssA"] - p0["AssA"]),
            ci(v["MOTA"]), ci(v["IDF1"]), f"{n(p1['FP'] - p0['FP'], 0)}/{n(p1['FN'] - p0['FN'], 0)}", w]) + " \\\\")
    g0, g1 = D.BOOST["BT7C_BASELINE_pf_post_gbi"], D.BOOST["BT7C_V7f_pf_post_gbi"]
    same = all(g0[k] == g1[k] for k in g0)
    NUM["boost_gbi_identical"] = same
    rows.append(" & ".join(["BoostTrack + GBI", "0.1", "71.33/80.55/83.84",
                            f"{g0['HOTA']:.2f}/{g0['MOTA']:.2f}/{g0['IDF1']:.2f}",
                            "\\multicolumn{5}{c}{identical output (all sequences)}", "0/0", "0/7/0"]) + " \\\\")
    cap = ("Development hosts on MOT17 val-half (7 sequences, published YOLOX-X detections, emission floor 0.01 or 0.1). "
           "Host = our reproduction; $\\Delta$ = host + V7f $-$ host. Bracketed intervals: paired sequence bootstrap, "
           "10,000 resamples, seed 42, 95\\% percentile; bold = interval excludes 0. $\\Delta$DetA and $\\Delta$AssA "
           "without interval were not bootstrapped in the development record. W/T/L: sequences with "
           "$\\Delta$HOTA $\\ge 0.01$ / $|\\Delta| < 0.01$ / $\\le -0.01$. These hosts were visible while V7f was designed; "
           "they are development evidence, not held-out evidence.")
    hdr = ("Host & Floor & Paper HOTA/MOTA/IDF1 & Host HOTA/MOTA/IDF1 & $\\Delta$HOTA & $\\Delta$DetA & $\\Delta$AssA & "
           "$\\Delta$MOTA & $\\Delta$IDF1 & $\\Delta$FP/$\\Delta$FN & W/T/L")
    write("tab3_development.tex", table("tab:dev", cap, "llllllllll l", hdr, rows))


# ------------------------------------------------------------------ Table 4: external
def tab_external():
    rows_a, rows_b = [], []

    def add(sysname, data, cls, base, b, per, w=None):
        w = w or wtl(per)
        rows_a.append(" & ".join([sysname, data, cls, f"{base['HOTA']:.2f}", ci(bt(b, "HOTA")),
                                  ci(bt(b, "DetA")), ci(bt(b, "AssA")), w]) + " \\\\")
        rows_b.append(" & ".join([sysname, data, ci(bt(b, "MOTA")), ci(bt(b, "IDF1")), cnt(bt(b, "IDS")),
                                  cnt(bt(b, "FP")), cnt(bt(b, "FN"))]) + " \\\\")
        NUM[f"ext::{sysname}::{data}"] = dict(HOTA=bt(b, "HOTA"), wtl=w)

    pd = D.EXT["PD-SORT (IEEE TCE 2025)"]
    per = [v["with_v7f"]["HOTA"] - v["reproduction"]["HOTA"] for v in pd["per_seq"].values()]
    add("PD-SORT$^{a}$", "MOT17", "EXACT", pd["reproduction"], pd["bootstrap_10k_seed42"], per)
    hs = D.EXT["Hybrid-SORT (AAAI 2024)"]
    add("Hybrid-SORT$^{a}$", "MOT17", "CLOSE", hs["reproduction"], hs["bootstrap_10k_seed42"], [0.0] * 7)
    for run, cls, lab, rc in (("MOT17", "pedestrian", "MOT17", "CLOSE"), ("KITTIMOT", "car", "KITTI car", "EXACT"),
                              ("KITTIMOT", "pedestrian", "KITTI ped.", "CLOSE"),
                              ("DanceTrack", "pedestrian", "DanceTrack", "CLOSE")):
        r = D.recent("ctwix", run, cls)
        d = {k: dict(diff=v["diff"], ci_lo=v["ci_lo"], ci_hi=v["ci_hi"]) for k, v in r["delta"].items()}
        w, t, l = r["seq_wins_ties_losses"]
        add("C-TWiX$^{b}$", lab, rc, r["baseline"]["pooled"], d, None, f"{w}/{t}/{l}")
    r = D.recent("tracktrack", "DanceTrack_post", "pedestrian")
    d = {k: dict(diff=v["diff"], ci_lo=v["ci_lo"], ci_hi=v["ci_hi"]) for k, v in r["delta"].items()}
    w, t, l = r["seq_wins_ties_losses"]
    add("TrackTrack$^{b}$", "DanceTrack", "CLOSE", r["baseline"]["pooled"], d, None, f"{w}/{t}/{l}")
    cap = ("External systems: $^{a}$ selected and declared in the freeze commit; $^{b}$ selected by a protocol "
           "preregistered after the freeze and before any run. Every system was run once with the frozen V7f; "
           "nothing was retuned. Host = our reproduction (class per the preregistered rule, Table~\\ref{tab:audit}); "
           "$\\Delta$ = host + V7f $-$ host; paired sequence bootstrap, 10,000 resamples, seed 42, 95\\% interval; "
           "bold = interval excludes 0. MOT17 = val-half (7 sequences); KITTI = KITTIMOTS validation sequences "
           "(9); DanceTrack = validation (25). TrackTrack is reported on its post-processed output, as in its paper. "
           "TOPICTrack is excluded (failed reproduction, Table~\\ref{tab:audit}).")
    s = ("\\begin{table}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{tab:ext}\n\\footnotesize\n"
         "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lllllllc}\n\\hline\n"
         "\\multicolumn{8}{l}{(a) Higher-order tracking accuracy and its components} \\\\\n\\hline\n"
         "System & Data & Reproduction & Host HOTA & $\\Delta$HOTA & $\\Delta$DetA & $\\Delta$AssA & W/T/L \\\\\n\\hline\n"
         + "\n".join(rows_a) + "\n\\hline\n\\end{tabular}}\n\n\\vspace{4pt}\n\\resizebox{\\textwidth}{!}{%\n"
         "\\begin{tabular}{lllllll}\n\\hline\n\\multicolumn{7}{l}{(b) CLEAR and identity metrics} \\\\\n\\hline\n"
         "System & Data & $\\Delta$MOTA & $\\Delta$IDF1 & $\\Delta$IDS & $\\Delta$FP & $\\Delta$FN \\\\\n\\hline\n"
         + "\n".join(rows_b) + "\n\\hline\n\\end{tabular}}\n\\end{table}\n")
    write("tab4_external.tex", s)


# ------------------------------------------------------------------ Table 5: KITTI
def tab_kitti():
    rows = []
    for (h, det), v in D.STAT_KITTI.items():
        b = v["base"]
        rows.append(" & ".join([det, h, str(v["n"]), f"{b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f} ({b[3]})", ci(v["HOTA"]),
                                ci(v["MOTA"]), ci(v["IDF1"]), cnt(v["IDS"])]) + " \\\\")
    cap = ("Detector transfer on KITTI tracking training (development data): COCO-pretrained RT-DETR-L and "
           "YOLOv8n with three hosts at their published or library-default operating points; official KITTI HOTA, "
           "car and pedestrian averaged. Same bootstrap as Table~\\ref{tab:dev} over sequences. BoT-SORT cells "
           "exclude sequence 0020, where the unmodified host raised a numerical error on V7f-filtered input.")
    hdr = ("Detector & Host & Seq. & Host HOTA/MOTA/IDF1 (IDS) & $\\Delta$HOTA & $\\Delta$MOTA & $\\Delta$IDF1 & $\\Delta$IDS")
    write("tab5_kitti.tex", table("tab:kitti", cap, "llllllll", hdr, rows))


# ------------------------------------------------------------------ Table 6: calibration shift
def tab_calibration():
    if D.CALIB is None:
        print("tab6 skipped: calibration bootstrap not available")
        return
    lab = {"pow3": "$s^3$", "scale05": "$0.5\\,s$", "temp2": "temp.\\ 2", "temp05": "temp.\\ 0.5"}
    rows = []
    for r in D.CALIB["conditions"]:
        a, b, bo = r["native"], r["v7f"], r["bootstrap"]
        v = r["verdict"]
        cell = ("identical output" if v == "identical output" else ci(bt(bo, "HOTA")))
        w, t, l = r["wins_ties_losses"]
        rows.append(" & ".join([r["host"], lab[r["transform"]], f"{a['HOTA']:.2f}", f"{b['HOTA']:.2f}", cell,
                                ci(bt(bo, "MOTA")) if v != "identical output" else "--",
                                ci(bt(bo, "IDF1")) if v != "identical output" else "--",
                                f"{w}/{t}/{l}", v.replace("significant ", "sig.\\ ")]) + " \\\\")
    s = D.CALIB["summary"]
    NUM["calib"] = s
    cap = ("Detector-score calibration shift (development hosts, MOT17 val-half). The transform is applied to every "
           "detection score before the host; the host keeps its published thresholds. Host = host alone on the "
           "transformed stream; paired bootstrap as in Table~\\ref{tab:dev}. The significance rule was fixed in the "
           "analysis script before the intervals were computed (after the freeze; the pooled values were recorded "
           f"before the freeze). Significant recovery in {s['significant_recovery']} of {s['conditions']} "
           f"conditions; significant degradation in {s['significant_degradation']}.")
    hdr = ("Host & Transform & Host HOTA & + V7f HOTA & $\\Delta$HOTA & $\\Delta$MOTA & $\\Delta$IDF1 & W/T/L & Verdict")
    write("tab6_calibration.tex", table("tab:calib", cap, "lllllllll", hdr, rows))


# ------------------------------------------------------------------ Table 7: runtime
def tab_runtime():
    a = D.RT["arms"]
    st = [("read", "Frame decode"), ("det", "Detector (YOLOv8n, 736 px, fp32)"),
          ("ctrl", "V7f controller (motion cue, step, observe)"), ("trk", "Tracker (ByteTrack)"),
          ("pipe", "Detector + controller + tracker"), ("e2e", "End to end")]
    rows = []
    for k, lab in st:
        b0 = "--" if k == "ctrl" else f"{a['baseline'][k]['mean_ms']:.2f} / {a['baseline'][k]['p95_ms']:.2f}"
        rows.append(f"{lab} & {b0} & {a['v7'][k]['mean_ms']:.2f} / {a['v7'][k]['p95_ms']:.2f} \\\\")
    rows.append(f"Frames per second, end to end & {a['baseline']['fps_e2e']:.2f} & {a['v7']['fps_e2e']:.2f} \\\\")
    d = a["v7"]["e2e"]["mean_ms"] - a["baseline"]["e2e"]["mean_ms"]
    NUM["runtime"] = dict(ctrl=a["v7"]["ctrl"]["mean_ms"], ctrl_p95=a["v7"]["ctrl"]["p95_ms"], e2e_delta=d,
                          e2e_pct=100 * d / a["baseline"]["e2e"]["mean_ms"], frames=a["v7"]["frames"])
    cap = ("Runtime after the freeze on one fixed device (cloud virtual machine, Intel Xeon at 2.10 GHz, 4 vCPU, "
           f"no graphics processor, PyTorch on the CPU with 4 threads), live YOLOv8n and ByteTrack on KITTI sequences 0001, 0009 and 0019, "
           f"{a['v7']['frames']} timed frames per arm, both arms on the same frames. Mean / 95th percentile in "
           "milliseconds per frame. The tracker is faster with V7f because fewer candidates reach it.")
    write("tab7_runtime.tex", table("tab:runtime", cap, "lll", "Stage & Host alone & Host + V7f", rows,
                                    resize=False, size="\\small"))


# ------------------------------------------------------------------ Table 8: reproduction / failure audit
def tab_audit():
    R = [
        ("SparseTrack", "A: development", "MOT17 val-half", "69.2/76.8/81.4", "68.88/77.85/81.97", "CLOSE",
         "HOTA $-0.32$, MOTA $+1.05$; official code, checkpoint and detections"),
        ("BoostTrack (online)", "A: development", "MOT17 val-half", "68.37/75.56/81.35", "68.49/75.50/81.41",
         "EXACT", "within 0.2/0.3 of the authors' corrected numbers"),
        ("ByteTrack (official setting)", "A: development", "MOT17 val-half", "--/76.6/79.3", "67.70/77.60/79.47",
         "not comparable", "paper ran its own detector inference; published detections replayed here"),
        ("OC-SORT", "A: development", "MOT17 val-half", "66.5/74.9/77.7", "66.43/74.67/78.05", "CLOSE",
         "HOTA $-0.07$, MOTA $-0.23$, IDF1 $+0.35$; published detections replayed"),
        ("PD-SORT", "B: predeclared", "MOT17 val-half", "authors' released output", "metric-identical",
         "EXACT", "reproduces the authors' released run file for file"),
        ("Hybrid-SORT", "B: predeclared", "MOT17 val-half", "67.1/75.8/78.0 (README)", "66.70/75.52/77.56",
         "CLOSE", "fp16 released detections vs fp32 in the README; Python 3.11 environment"),
        ("C-TWiX", "C: post-freeze", "MOT17 val-half", "77.8 (README)", "77.55", "CLOSE", "CPU float16 autocast"),
        ("C-TWiX", "C: post-freeze", "KITTI val car / ped.", "89.3 / 71.4", "89.27 / 70.77", "EXACT / CLOSE", "CPU run"),
        ("C-TWiX", "C: post-freeze", "DanceTrack val", "60.4", "59.62", "CLOSE", "CPU run"),
        ("TrackTrack", "C: post-freeze", "DanceTrack val", "63.3 (paper)", "62.96", "CLOSE",
         "ReID features re-extracted on CPU float32"),
        ("TOPICTrack", "C: post-freeze", "MOT17 val-half", "69.6/79.8/81.2 (README)", "67.54/79.07/78.63",
         "FAILED", "$|\\Delta$HOTA$| > 1.0$; fewer detections (FP $-$1,187, FN +1,660); cause not diagnosed; "
         "excluded from all claims"),
        ("TOPICTrack", "C: post-freeze", "MOT20 val-half", "57.5 (README)", "not run", "--", "not attempted"),
        ("LG-MOT", "candidate", "--", "test set only", "not run", "--", "no verifiable validation reference"),
        ("MOTIP", "candidate", "--", "--", "not run", "--", "architecture audit not completed"),
        ("CAMELTrack", "candidate", "--", "--", "not run", "--", "peer-reviewed venue not verified; exploratory only"),
    ]
    rows = [" & ".join(r) + " \\\\" for r in R]
    cap = ("Reproduction audit. Reference = the authors' number for the same split and setting (paper or official "
           "README), recorded before any run for the post-freeze systems; HOTA/MOTA/IDF1. Classes follow the "
           "preregistered rule: EXACT, $|\\Delta$HOTA$| \\le 0.2$ and $|\\Delta$MOTA$|$, $|\\Delta$IDF1$| \\le 0.3$ or "
           "released outputs reproduced; CLOSE, $|\\Delta$HOTA$| \\le 1.0$ with a documented, non-tuned cause; "
           "otherwise FAILED. V7f was run only on EXACT or CLOSE baselines for claims; the TOPICTrack arm is "
           "recorded as exploratory.")
    cols = ("@{}>{\\raggedright\\arraybackslash}p{2.2cm}>{\\raggedright\\arraybackslash}p{1.7cm}"
            ">{\\raggedright\\arraybackslash}p{1.9cm}>{\\raggedright\\arraybackslash}p{2.3cm}"
            ">{\\raggedright\\arraybackslash}p{2.0cm}>{\\raggedright\\arraybackslash}p{1.4cm}"
            ">{\\raggedright\\arraybackslash}p{3.9cm}@{}")
    s = table("tab:audit", cap, cols, "System & Role & Split & Reference & Reproduced & Class & Note", rows,
              resize=False, size="\\scriptsize\\setlength{\\tabcolsep}{3pt}")
    write("tab8_audit.tex", s)


# ------------------------------------------------------------------ Table 9: mechanism ablation (V6 -> V7f)
def tab_ablation():
    m = D.MOT
    b = D.BOOST
    R = [
        ("V6-TF emulated (always-noisy bands, IoU duplicate rule, rank scores)",
         f"{m['BY_official_bt_V6EMU']['HOTA']:.2f}", f"{m['OC_bt_V6EMU']['HOTA']:.2f}",
         f"{b['BT7C_V6EMU_pf']['HOTA']:.2f}"),
        ("V7d: $\\rho$ regime (100 frames), host-anchored clean regime, crowd-safe duplicates",
         f"{m['BY_official_bt_V7d']['HOTA']:.2f}" if "BY_official_bt_V7d" in m else "--",
         f"{m['OC_bt_V7d']['HOTA']:.2f}" if "OC_bt_V7d" in m else "--", f"{b['BT7C_V7d_pf']['HOTA']:.2f}"),
        ("+ regime from the whole stream", f"{m['BY_official_bt_V7d_cum']['HOTA']:.2f}" if "BY_official_bt_V7d_cum" in m else "--",
         f"{m['OC_bt_V7d_cum']['HOTA']:.2f}" if "OC_bt_V7d_cum" in m else "--", f"{b['BT7C_V7d_cum_pf']['HOTA']:.2f}"),
        ("+ foreground track-consistent rescue (V7e)", f"{m['BY_official_bt_V7e']['HOTA']:.2f}" if "BY_official_bt_V7e" in m else "--",
         f"{m['OC_bt_V7e']['HOTA']:.2f}" if "OC_bt_V7e" in m else "--", f"{b['BT7C_V7e_pf']['HOTA']:.2f}"),
        ("+ interpretability check of the split (V7f, frozen)", f"{m['BY_official_bt_V7f']['HOTA']:.2f}",
         f"{m['OC_bt_V7f']['HOTA']:.2f}", f"{b['BT7C_V7f_pf']['HOTA']:.2f}"),
        ("Host alone", f"{m['BY_official_bt_BASELINE']['HOTA']:.2f}", f"{m['OC_bt_BASELINE']['HOTA']:.2f}",
         f"{b['BT7C_BASELINE_pf']['HOTA']:.2f}"),
    ]
    rows = [" & ".join(r) + " \\\\" for r in R]
    cap = ("Mechanism ablation on development data, one change at a time (HOTA, MOT17 val-half, emission floor 0.1, "
           "no image motion cue). Each row adds one mechanism to the previous row; the first row places the prior "
           "V6-TF rules inside the same framework. Values from the development record; no interval is claimed.")
    write("tab9_ablation.tex", table("tab:ablation", cap, "llll", "Variant & ByteTrack (official) & OC-SORT & BoostTrack (online)",
                                     rows, resize=True))


if __name__ == "__main__":
    tab_development()
    tab_external()
    tab_kitti()
    tab_calibration()
    tab_runtime()
    tab_audit()
    tab_ablation()
    (OUT / "text_numbers.json").write_text(json.dumps(NUM, indent=1, default=str))
    print("wrote text_numbers.json")
    # LaTeX macros for numbers quoted in the running text (traceable to the same files)
    m = []
    rt = NUM["runtime"]
    m += [f"\\newcommand{{\\RTctrl}}{{{rt['ctrl']:.2f}}}", f"\\newcommand{{\\RTctrlP}}{{{rt['ctrl_p95']:.2f}}}",
          f"\\newcommand{{\\RTdelta}}{{{rt['e2e_delta']:.2f}}}", f"\\newcommand{{\\RTpct}}{{{rt['e2e_pct']:.1f}}}"]
    if D.CALIB is not None:
        s = D.CALIB["summary"]
        m += [f"\\newcommand{{\\CalibN}}{{{s['conditions']}}}", f"\\newcommand{{\\CalibRec}}{{{s['significant_recovery']}}}",
              f"\\newcommand{{\\CalibDeg}}{{{s['significant_degradation']}}}",
              f"\\newcommand{{\\CalibIdent}}{{{sum(r['verdict'] == 'identical output' for r in D.CALIB['conditions'])}}}",
              f"\\newcommand{{\\CalibNS}}{{{sum(r['verdict'] == 'no significant change' for r in D.CALIB['conditions'])}}}"]
    else:
        m += ["\\newcommand{\\CalibN}{14}", "\\newcommand{\\CalibRec}{[PENDING]}", "\\newcommand{\\CalibDeg}{[PENDING]}",
              "\\newcommand{\\CalibIdent}{[PENDING]}", "\\newcommand{\\CalibNS}{[PENDING]}"]
    (OUT / "numbers.tex").write_text("\n".join(m) + "\n")
    print("wrote numbers.tex")
