"""
LaTeX tables for Paper 1, generated only from the evidence snapshot
research/paper_split/evidence/legacy/ (machine-readable files) and, where no
machine-readable file exists, from the numbered sections of the freeze record
ACMOT_FINAL_SCIENTIFIC_FREEZE_2026-09-12.md (cited inline below).
Run from anywhere:

  python JAIS_PAPERS/PAPER1_SCENE_ADAPTIVE_JAIS/scripts/make_tables.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EV = ROOT / "research/paper_split/evidence/legacy"
OUT = Path(__file__).resolve().parents[1] / "tables"
OUT.mkdir(exist_ok=True)


def rows(name):
    return list(csv.DictReader(open(EV / name)))


def pct(x):
    return f"{100 * float(x):.2f}"


def write(name, body):
    (OUT / name).write_text(body)
    print("wrote", OUT / name)


cfg = json.load(open(EV / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"))
Q = cfg["optimized_parameters"]
# Balanced profile (Trial 22): freeze record section 12 (no separate JSON in the snapshot).
B = dict(weight_crowd=.16464526567145857, weight_tiny=.17462652795045444, weight_edge=.5076112530333374,
         weight_night=.12069564693725379, weight_blur=.03242130640749567, conf_easy=.4, conf_hard=.4,
         nms_easy=.6, nms_hard=.35, threshold_mid=.2927135841069045, threshold_high=.6661671600900015)

# ---------------------------------------------------------------- Table 1: settings
t = r"""\begin{table}[t]
\centering
\caption{Frozen configurations compared in this study. Confidence and NMS entries give the value at $S=0$ and at $S=1$ (linear in between); resolution is the short-side input size in pixels. All adaptive controllers analyze every $K=10$ frames with a moving mean over $W=7$ analyses.}
\label{tab:settings}
\small
\begin{tabular}{lccccc}
\hline
Configuration & Confidence & NMS IoU & Resolution & Tracker profile & Selection \\
\hline
Static default & 0.25 & 0.45 & 640 & default & none (project reference) \\
Heuristic controller & $0.245-0.05S$ (clipped 0.19--0.28) & $0.49-0.05S$ (clipped 0.40--0.52) & 640/736/832 & tuned & hand-designed \\
"""
t += (f"Quality profile Q (trial 24) & {Q['conf_easy']:.2f}$\\to${Q['conf_hard']:.2f} & {Q['nms_easy']:.2f}$\\to${Q['nms_hard']:.2f} "
      f"& 512/928/960 at $S\\ge{Q['threshold_mid']:.3f}/{Q['threshold_high']:.3f}$ & tuned & max MOTA s.t. IDS gate \\\\\n")
t += (f"Balanced profile B (trial 22) & {B['conf_easy']:.2f}$\\to${B['conf_hard']:.2f} & {B['nms_easy']:.2f}$\\to${B['nms_hard']:.2f} "
      f"& 512/928/960 at $S\\ge{B['threshold_mid']:.3f}/{B['threshold_high']:.3f}$ & tuned & balanced Pareto score \\\\\n")
a = cfg["operating_ablation"]["cue_calibration_anchor"]
t += (f"Matched static anchor & {a['confidence']:.2f} & {a['nms_iou']:.2f} & {a['resolution']} & tuned & post hoc attribution \\\\\n")
t += r"""\hline
\multicolumn{6}{l}{Tracker profiles (ByteTrack): default = high 0.25, low 0.10, new 0.25, buffer 30, match 0.80;}\\
\multicolumn{6}{l}{tuned = high 0.18, low 0.04, new 0.20, buffer 45, match 0.86.}\\
\hline
\end{tabular}
\end{table}
"""
write("tab1_settings.tex", t)

# ---------------------------------------------------------------- Table 2: data
t = r"""\begin{table}[t]
\centering
\caption{Data splits and their role. No test sequence was used for any selection, calibration, or tuning decision.}
\label{tab:data}
\small
\begin{tabular}{llrrl}
\hline
Dataset & Split & Sequences & Frames & Role \\
\hline
VisDrone2019-MOT & validation & 7 & 2,846 & cue calibration, sweeps, ablations, optimization \\
VisDrone2019-MOT & test-dev & 17 & 6,635 & locked held-out comparison \\
UAVDT & test & 20 & 16,592 & zero-tuning cross-dataset test \\
\hline
\end{tabular}
\end{table}
"""
write("tab2_data.tex", t)

# ---------------------------------------------------------------- Table 3: held-out comparison
done = json.load(open(EV / "FINAL_TEST_DONE.json"))["results"]
b2 = json.load(open(EV / "V2_TRIAL22_TESTDEV_RESULT.json")) if (EV / "V2_TRIAL22_TESTDEV_RESULT.json").exists() else None
m = json.load(open(EV / "MATCHED_STATIC_A0_TESTDEV.json"))
ua = {r["system"]: r for r in json.load(open(EV / "UAVDT_FINAL_COMPARISON.json"))}


def line(name, r, fps=True, dag=""):
    f = f"{float(r['FPS']):.2f}" if fps and "FPS" in r else "--"
    return (f"{name} & {pct(r['HOTA'])} & {pct(r['MOTA'])} & {pct(r['IDF1'])} & {int(r['IDS'])} & "
            f"{int(r['FN'])} & {int(r['FP'])} & {f}{dag} \\\\\n")


def v2r():
    d = b2 if isinstance(b2, dict) else None
    if d is None:
        return None
    for k in ("metrics", "result", "final"):
        if k in d and isinstance(d[k], dict):
            d = d[k]
    return d


t = r"""\begin{table}[t]
\centering
\caption{Held-out results (percent for HOTA, MOTA, IDF1; counts for IDS, FN, FP; processing frames per second on one NVIDIA Tesla T4, each system measured in its own session). No entry is marked as best because the profiles target different trade-offs.}
\label{tab:main}
\small
\begin{tabular}{lrrrrrrr}
\hline
System & HOTA & MOTA & IDF1 & IDS & FN & FP & FPS \\
\hline
\multicolumn{8}{l}{\textit{VisDrone2019-MOT test-dev (17 sequences, 6,635 frames)}}\\
"""
t += line("Static default", done["Baseline_Default"])
t += line("Heuristic controller", done["Old_ACMOT_Frozen"])
t += line("Quality profile Q", done["New_ACMOT_Frozen"])
vr = v2r()
if vr:
    t += line("Balanced profile B", vr)
t += line("Matched static anchor", dict(m["observed_A0"], FPS=m["A0_system"]["processing_fps"]), dag="$^{a}$")
t += r"\multicolumn{8}{l}{\textit{UAVDT test (20 sequences, 16,592 frames), no tuning on UAVDT}}\\" + "\n"
t += line("Static default", ua["Baseline_Frozen"])
t += line("Quality profile Q", ua["V1_Trial24_Frozen"])
t += line("Balanced profile B", ua["V2_Trial22_Frozen"])
t += r"""\hline
\multicolumn{8}{l}{$^{a}$Run after the freeze for attribution only; frames per second measured in a separate T4 session.}\\
\hline
\end{tabular}
\end{table}
"""
write("tab3_main.tex", t)

# ---------------------------------------------------------------- Table 4: paired bootstrap
bs = {(r["comparison"], r["metric"]): r for r in rows("V1_PAIRED_BOOTSTRAP_95CI.csv")}


def ci(d, lo, hi, digits=2, sign=True):
    f = f"{{:+.{digits}f}}" if sign else f"{{:.{digits}f}}"
    return f"{f.format(d)} [{f.format(lo)}, {f.format(hi)}]"


t = r"""\begin{table}[t]
\centering
\caption{Paired sequence-level bootstrap (5,000 resamples, seed 42): observed difference and 95\% percentile interval. HOTA, MOTA and IDF1 in percentage points; IDS reduction in switches (positive = fewer switches for the first system).}
\label{tab:boot}
\small
\begin{tabular}{llllll}
\hline
Data & Comparison & $\Delta$HOTA & $\Delta$MOTA & $\Delta$IDF1 & IDS reduction \\
\hline
"""
for comp, lab in (("New vs Baseline", "Q vs static default"), ("New vs Old", "Q vs heuristic")):
    g = lambda k: bs[(comp, k)]
    t += f"VisDrone & {lab} & " + " & ".join(
        ci(float(g(k)["observed_pp"]), float(g(k)["CI95_low_pp"]), float(g(k)["CI95_high_pp"]), 0 if k == "IDS_reduction" else 2)
        for k in ("HOTA", "MOTA", "IDF1", "IDS_reduction")) + " \\\\\n"
r = m["results_V1_minus_A0_pp"]
t += "VisDrone & Q vs matched static anchor & " + " & ".join(
    ci(r[k]["delta"], r[k]["ci_low"], r[k]["ci_high"], 0 if k == "IDS_reduction" else 2)
    for k in ("HOTA", "MOTA", "IDF1", "IDS_reduction")) + " \\\\\n"
# Freeze record sections 14 and 19 (no machine-readable CSV in the snapshot).
FR = [("VisDrone", "B vs static default", (2.788, .747, 4.660), (4.063, .952, 6.886), (5.146, 2.028, 8.080), (316, 175, 470)),
      ("VisDrone", "B vs Q", (-2.617, -4.004, -1.694), (-3.157, -6.145, -1.346), (-3.676, -5.882, -2.241), (265, 82, 502)),
      ("UAVDT", "Q vs static default", (4.305, 2.740, 5.709), (3.558, 1.452, 5.807), (6.565, 3.808, 8.822), (237, 104, 377)),
      ("UAVDT", "B vs static default", (2.845, 1.238, 4.348), (2.277, .720, 4.164), (4.127, 1.630, 6.485), (250, 110, 404)),
      ("UAVDT", "B vs Q", (-1.460, -2.190, -.908), (-1.281, -2.441, -.202), (-2.439, -3.737, -1.366), (13, -20, 50))]
for d, lab, h, mo, i1, ids in FR:
    t += f"{d} & {lab} & {ci(*h)} & {ci(*mo)} & {ci(*i1)} & {ci(*ids, digits=0)} \\\\\n"
t += r"""\hline
\end{tabular}
\end{table}
"""
write("tab4_bootstrap.tex", t)

# ---------------------------------------------------------------- Table 5: ablations (validation)
old = {r["stage"]: r for r in rows("OLD_ACMOT_COMPONENT_ABLATION.csv")}
new = {r["stage"]: r for r in rows("NEW_ACMOT_COMPONENT_ABLATION.csv")}
t = r"""\begin{table}[t]
\centering
\caption{Component ablations on VisDrone2019-MOT validation (7 sequences). FPS is processing throughput on a T4; mean resolution, confidence and NMS are averaged over frames.}
\label{tab:ablation}
\small
\begin{tabular}{llrrrrrrrr}
\hline
Stage & Components & HOTA & MOTA & IDF1 & IDS & FPS & Res. & Conf. & NMS \\
\hline
\multicolumn{10}{l}{\textit{Heuristic controller}}\\
"""
lab = {"OLD-A0": ("H0", "static default, default tracker"), "OLD-A1": ("H1", "+ tuned tracker"),
       "OLD-A2": ("H2", "+ adaptive conf./NMS (640 px)"), "OLD-A2R": ("H2R", "tuned tracker + adaptive resolution only"),
       "OLD-A3": ("H3", "full heuristic controller")}
for k in ("OLD-A0", "OLD-A1", "OLD-A2", "OLD-A2R", "OLD-A3"):
    r = old[k]
    t += (f"{lab[k][0]} & {lab[k][1]} & {pct(r['HOTA'])} & {pct(r['MOTA'])} & {pct(r['IDF1'])} & {r['IDS']} & "
          f"{float(r['FPS']):.1f} & {float(r['mean_imgsz']):.0f} & {float(r['mean_conf']):.3f} & {float(r['mean_nms_iou']):.3f} \\\\\n")
t += r"\multicolumn{10}{l}{\textit{Optimized controller (quality profile Q)}}\\" + "\n"
lab = {"A0": ("C0", "static anchor 960/0.35/0.35, tuned tracker"), "A1": ("C1", "+ adaptive confidence"),
       "A2": ("C2", "+ adaptive NMS"), "A3": ("C3", "+ adaptive resolution (full Q)")}
for k in ("A0", "A1", "A2", "A3"):
    r = new[k]
    t += (f"{lab[k][0]} & {lab[k][1]} & {pct(r['HOTA'])} & {pct(r['MOTA'])} & {pct(r['IDF1'])} & {r['IDS']} & "
          f"{float(r['FPS']):.1f} & {float(r['mean_imgsz']):.0f} & {float(r['mean_conf']):.3f} & {float(r['mean_nms_iou']):.3f} \\\\\n")
t += r"""\hline
\end{tabular}
\end{table}
"""
write("tab5_ablation.tex", t)

# ---------------------------------------------------------------- Table 6: operating sweeps / accuracy-speed
res = rows("OPERATING_RESOLUTION_SWEEP.csv")
t = r"""\begin{table}[t]
\centering
\caption{Accuracy--throughput trade-off of the detector input resolution on VisDrone2019-MOT validation (confidence 0.25, NMS IoU 0.70, tuned tracker; T4 processing throughput and 95th-percentile per-frame latency).}
\label{tab:speed}
\small
\begin{tabular}{rrrrrrr}
\hline
Resolution & HOTA & MOTA & IDF1 & IDS & FPS & P95 latency (ms) \\
\hline
"""
for r in res:
    if int(r["resolution"]) in (512, 576, 640, 704, 768, 832, 896, 928, 960):
        t += (f"{int(r['resolution'])} & {pct(r['HOTA'])} & {pct(r['MOTA'])} & {pct(r['IDF1'])} & {r['IDS']} & "
              f"{float(r['FPS']):.1f} & {float(r['p95_ms']):.1f} \\\\\n")
t += r"""\hline
\end{tabular}
\end{table}
"""
write("tab6_speed.tex", t)

# ---------------------------------------------------------------- Table 7: temporal ablation (compact)
tem = rows("TEMPORAL_ABLATION_FULL.csv")
t = r"""\begin{table}[t]
\centering
\caption{Temporal settings of the heuristic controller on VisDrone2019-MOT validation: HOTA (\%) / processing FPS for smoothing window $W$ (analyses) and analysis stride $K$ (frames). The selected setting is $W=7$, $K=10$.}
\label{tab:temporal}
\small
\begin{tabular}{r""" + "c" * 5 + r"""}
\hline
$W$ \textbackslash{} $K$ & 1 & 5 & 10 & 15 & 20 \\
\hline
"""
g = defaultdict(dict)
for r in tem:
    g[int(r["smoothing_window"])][int(r["analysis_stride"])] = r
for w in sorted(g):
    cells = []
    for k in (1, 5, 10, 15, 20):
        r = g[w][k]
        s = f"{pct(r['HOTA'])} / {float(r['FPS']):.1f}"
        cells.append(r"\textbf{" + s + "}" if (w, k) == (7, 10) else s)
    t += f"{w} & " + " & ".join(cells) + " \\\\\n"
t += r"""\hline
\end{tabular}
\end{table}
"""
write("tab7_temporal.tex", t)

# ---------------------------------------------------------------- per-sequence summary numbers used in the text
v = defaultdict(dict)
for r in rows("V1_PER_SEQUENCE_METRICS.csv"):
    v[r["sequence"]][r["system"]] = r
u = defaultdict(dict)
for r in rows("UAVDT_PER_SEQUENCE.csv"):
    u[r["sequence"]][r["system"]] = r


def wtl(d, a, b, eps=1e-9):
    w = t_ = l = 0
    for s in d.values():
        x = float(s[a]["HOTA"]) - float(s[b]["HOTA"])
        if x > eps: w += 1
        elif x < -eps: l += 1
        else: t_ += 1
    return w, t_, l


stats = dict(
    testdev_Q_vs_default=wtl(v, "New_ACMOT_Frozen", "Baseline_Default"),
    testdev_Q_vs_heuristic=wtl(v, "New_ACMOT_Frozen", "Old_ACMOT_Frozen"),
    uavdt_Q_vs_default=wtl(u, "V1_Trial24_Frozen", "Baseline_Frozen"),
    uavdt_B_vs_default=wtl(u, "V2_Trial22_Frozen", "Baseline_Frozen"),
    testdev_frames=sum(int(s["Baseline_Default"]["frames"]) for s in v.values()),
    uavdt_zero_hota_all=[k for k, s in u.items() if all(float(s[x]["HOTA"]) < 0.005 for x in s)],
)
(OUT / "text_numbers.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats))
