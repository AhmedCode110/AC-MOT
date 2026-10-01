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
\caption{Frozen configurations compared in this study. Confidence and NMS entries give the value at $S=0$ and at $S=1$ (linear in between); resolution is the long-side input size in pixels. All adaptive controllers analyze every $K=10$ frames with a moving mean over $W=7$ analyses.}
\label{tab:settings}
\small
\begin{tabular}{p{3.3cm}p{3.0cm}p{3.0cm}p{3.8cm}p{1.2cm}}
\hline
Configuration & Confidence & NMS IoU & Resolution & Tracker \\
\hline
Static default & 0.25 & 0.45 & 640 & default \\
Hand-designed controller & $0.245-0.05S$, clipped to 0.19--0.28 & $0.49-0.05S$, clipped to 0.40--0.52 & 640/736/832 (rules in Sec.~\ref{sec:control}) & tuned \\
"""
t += (f"Quality profile Q (trial 24) & {Q['conf_easy']:.2f}$\\to${Q['conf_hard']:.2f} & {Q['nms_easy']:.2f}$\\to${Q['nms_hard']:.2f} "
      f"& 512/928/960 at $S\\ge{Q['threshold_mid']:.3f}/{Q['threshold_high']:.3f}$ & tuned \\\\\n")
t += (f"Balanced profile B (trial 22) & {B['conf_easy']:.2f}$\\to${B['conf_hard']:.2f} & {B['nms_easy']:.2f}$\\to${B['nms_hard']:.2f} "
      f"& 512/928/960 at $S\\ge{B['threshold_mid']:.3f}/{B['threshold_high']:.3f}$ & tuned \\\\\n")
a = cfg["operating_ablation"]["cue_calibration_anchor"]
t += (f"Matched static anchor & {a['confidence']:.2f} & {a['nms_iou']:.2f} & {a['resolution']} & tuned \\\\\n")
t += r"""\hline
\multicolumn{5}{l}{Tracker profiles (ByteTrack): default = high 0.25, low 0.10, new 0.25, buffer 30, match 0.80;}\\
\multicolumn{5}{l}{tuned = high 0.18, low 0.04, new 0.20, buffer 45, match 0.86.}\\
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
t += line("Hand-designed controller", done["Old_ACMOT_Frozen"])
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
    return f"${f.format(d)}$ [${f.format(lo)}$, ${f.format(hi)}$]"   # math mode: typographic minus


t = r"""\begin{table}[t]
\centering
\caption{Paired sequence-level bootstrap (5,000 resamples, seed 42): observed difference and 95\% percentile interval. HOTA, MOTA and IDF1 in percentage points; IDS reduction in switches (positive = fewer switches for the first system).}
\label{tab:boot}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{llllll}
\hline
Data & Comparison & $\Delta$HOTA & $\Delta$MOTA & $\Delta$IDF1 & IDS reduction \\
\hline
"""
for comp, lab in (("New vs Baseline", "Q vs static default"), ("New vs Old", "Q vs hand-designed")):
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
\end{tabular}}
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
\multicolumn{10}{l}{\textit{Hand-designed controller}}\\
"""
lab = {"OLD-A0": ("H0", "static default, default tracker"), "OLD-A1": ("H1", "+ tuned tracker"),
       "OLD-A2": ("H2", "+ adaptive conf./NMS (640 px)"), "OLD-A2R": ("H2R", "tuned tracker + adaptive resolution only"),
       "OLD-A3": ("H3", "full hand-designed controller")}
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
\caption{Temporal settings of the hand-designed controller on VisDrone2019-MOT validation: HOTA (\%) / processing FPS for smoothing window $W$ (analyses) and analysis stride $K$ (frames). The selected setting is $W=7$, $K=10$.}
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

# ---------------------------------------------------------------- Table 8: cross-pipeline consistency (HOTA, TrackEval)
# Row 1: matched static anchor record (5,000 resamples, seed 42; per-sequence anchor values not archived).
# Rows 2-3: post hoc re-scoring of the archived frozen outputs, research/transfer_legacy/rescore/ (commit 97853c6;
# 10,000 resamples, seed 0); used only because every archived motmetrics number was reproduced exactly.
RS = ROOT / "research/transfer_legacy/rescore"
u2 = json.load(open(RS / "u2mot/u2mot_testdev_rescore.json"))
st = json.load(open(RS / "sparsetrack/sparsetrack_val_half_rescore.json"))
assert u2["reproduction_all_match"] is True
assert all(x["all_match"] is True for x in st["reproduction"].values())
assert u2["paired_controller_vs_baseline"]["bootstrap_resamples"] == 10000 and u2["paired_controller_vs_baseline"]["seed"] == 0
uh = u2["paired_controller_vs_baseline"]["metrics"]["HOTA"]
sh = st["pairs"]["adaptive_vs_static075"]["metrics"]["HOTA"]
sd = st["pairs"]["adaptive_vs_static070"]["metrics"]["HOTA"]
H070 = st["systems"]["static070"]["trackeval_overall"]["HOTA"]
assert H070 == max(st["systems"][k]["trackeval_overall"]["HOTA"] for k in ("static070", "static075", "static080"))
assert st["pairs"]["adaptive_vs_static070"]["bootstrap_resamples"] == 10000 and st["pairs"]["adaptive_vs_static070"]["seed"] == 0
# Note b (U2MOT pipeline) transcribes the frozen controller and runtime, not the rescore JSON. Drive sources
# (FINAL_U2MOT_ACMOT_FREEZE_2026-09-14): 05_VALIDATED_CONTROLLER_FREEZE/FROZEN_CONFIG.json (1WMdWrMPc-gJc4FdZNEh1TkwjOPcSiYKE)
# and acmot_full_policy_calibrated.py (sha256 985491bb..., listed in that folder's SHA256SUMS.txt) for the boundaries and
# operating points; 02_SCI_CALIBRATION/SCI_CALIBRATION.txt (1k0wiYVDJRONBL-PctxA7GqE5ubljzcDb) for the label-free terciles;
# 07_CODE_SNAPSHOT/acmot_v1.py (1v44BZrDZRu4pg7MMc3L7_Q3lN_izUA2B) for the cue weights and transforms;
# 01_A0_BASELINE/BASELINE_RUN_SUMMARY.txt (1x0QwAyBRtjHMzUIjQRojz8B-W2T-ioi6) and upstream u2mot tools/track.py::parse_benchmark
# (commit 7411211) for the tracker settings; the evaluation filter is u2mot tools/utils/eval_visdrone.py at that commit.
# Note c (SparseTrack pipeline): FROZEN_CONTROLLER_MANIFEST.json (1IhAmHmoHxKMRk1xcI14bgDN81rqz-_wo) and
# frozen_adaptive_edge_v1_runner.py (1ybeR6-xMWcmhDYt1B8FbhKShhbT9MRQj) for the rule, threshold, confidence and tracker;
# MOT20 EXPERIMENT_PROTOCOL.json (14unHmvv-hsiDT4CyBQyJEWNL-UxBLxQ5) for the input size and checkpoint provenance.
# Both are summarized in research/transfer_legacy/U2MOT_SPARSETRACK_EVIDENCE_AUDIT.md.
ah = json.load(open(EV / "MATCHED_STATIC_A0_TESTDEV.json"))["results_V1_minus_A0_pp"]["HOTA"]


def dci(d, lo, hi):
    return f"${d:+.2f}$", f"[${lo:+.2f}$, ${hi:+.2f}$]"


def wtl3(x):
    return f"{x['wins']}/{x['ties']}/{x['losses']}"


L = r">{\raggedright\arraybackslash}p"
t = r"""\begin{table}[t]
\centering
\caption{Cross-pipeline consistency: scene switching versus a static operating point. $\Delta$HOTA is adaptive minus static (TrackEval, percentage points) with its 95\% paired sequence bootstrap interval (row 1: 5,000 resamples, seed 42; rows 2--3: 10,000 resamples, seed 0); W/T/L counts the sequences won, tied, and lost by the adaptive system. Detectors, trackers, ground-truth filters, and policies differ between rows, so absolute accuracies are not comparable across rows.}
\label{tab:crosspipe}
\footnotesize
\setlength\tabcolsep{3pt}
\begin{tabular}{""" + L + "{1.6cm}" + L + "{1.75cm}" + L + "{1.9cm}" + L + "{2.0cm}ccc" + L + "{1.95cm}" + L + r"""{1.6cm}}
\hline
Pipeline & Dataset/split & Adaptive mechanism & Static comparator & $\Delta$HOTA & 95\% CI & W/T/L & Evidence status & Interpretation \\
\hline
"""
d, c = dci(ah["delta"], ah["ci_low"], ah["ci_high"])
t += (r"YOLOv8n + ByteTrack (this study) & VisDrone2019 test-dev, 17 sequences & Quality profile Q: index $\to$ confidence, NMS, resolution "
      r"& Matched static anchor (conf.\ 0.35, NMS 0.35, 960 px, tuned tracker) & " + d + " & " + c + r" & n/a$^{a}$ "
      r"& Held-out; Q frozen before test; anchor run post hoc & No measurable change \\" + "\n")
d, c = dci(uh["delta"], *uh["ci95"])
t += (r"YOLOX-X + U2MOT & VisDrone2019 test-dev, 17 sequences & Recalibrated index tiers $\to$ confidence, NMS, input size$^{b}$ "
      r"& Author-calibrated static operating point (conf.\ 0.09, NMS 0.70, 1600$\times$896 px) & " + d + " & " + c + " & " + wtl3(uh) +
      r" & Held-out; controller frozen before test; post hoc TrackEval HOTA rescore of the frozen outputs & No measurable change \\" + "\n")
d, c = dci(sh["delta"], *sh["ci95"])
t += (r"YOLOX + SparseTrack & MOT17 validation half, 7 sequences & Edge cue $\to$ NMS 0.70 or 0.80$^{c}$ "
      r"& Static NMS 0.75 (named in the freeze record) & " + d + " & " + c + " & " + wtl3(sh) +
      r" & In-sample validation; threshold selected on the same 7 sequences & Small in-sample gain over NMS 0.75, none over default NMS 0.70$^{c}$; not held-out evidence \\" + "\n")
t += r"""\hline
\end{tabular}

\smallskip
\parbox{\textwidth}{\raggedright
$^{a}$Per-sequence values of the matched static anchor were not archived.\\
$^{b}$YOLOX-X checkpoint released by the U2MOT authors; tracker thresholds 0.5 and 0.1, matching threshold 0.8, buffer 15 frames in both runs. Index: cue weights of Q with the cue transforms of Eq.~\eqref{eq:heur}; boundaries 0.519 and 0.590 (terciles of the index on VisDrone validation, no labels). Operating points (confidence, NMS, input): (0.15, 0.70, 1280$\times$704), (0.12, 0.65, 1440$\times$800), (0.09, 0.60, 1600$\times$896); no record of how these values were chosen. Evaluation: pedestrian, car, van, truck, and bus merged class-agnostically, ignore regions removed by box center, overlap 0.5.\\
$^{c}$YOLOX ablation checkpoint released with ByteTrack \cite{zhang2022bytetrack} (training data include the first half of each MOT17 training sequence), input 1440$\times$800, confidence 0.01; tracker unchanged. Every tenth frame, the fraction of Canny edge pixels of a quarter-scale image divided by 0.14 and capped at 1, averaged over the last seven analyses, sets NMS 0.80 above the selected threshold and 0.70 otherwise. HOTA with the standard MOT17 pedestrian preprocessing on the validation-half ground truth. Against the default static NMS 0.70 (""" + f"{H070:.2f}" + r"""\% HOTA, highest of the static settings 0.70, 0.75, 0.80): $\Delta$HOTA """ + dci(sd["delta"], *sd["ci95"])[0] + " " + dci(sd["delta"], *sd["ci95"])[1] + ", W/T/L " + wtl3(sd) + r""".}
\end{table}
"""
write("tab8_crosspipeline.tex", t)

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
