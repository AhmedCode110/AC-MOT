"""
LaTeX tables for Paper 2, generated from scripts/evidence.py only.
  python JAIS_PAPERS/PAPER2_UNIVERSAL_ACMOT_JAIS/scripts/make_tables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence as E  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tables"
OUT.mkdir(exist_ok=True)


def ci(t, bold=True):
    if t is None:
        return "0 (identical)"
    d, lo, hi = t
    k = 3 if any(0 < abs(x) < 0.01 for x in (d, lo, hi)) or max(abs(d), abs(lo), abs(hi)) < 0.1 else 2
    s = f"${d:+.{k}f}$ [${lo:+.{k}f}$, ${hi:+.{k}f}$]"   # math mode: typographic minus
    return r"{\boldmath " + s + "}" if bold and (lo > 0 or hi < 0) else s


def hmi(t):
    return " / ".join(f"{x:.2f}" for x in t)


def write(name, s):
    (OUT / name).write_text(s)
    print("wrote", name)


# ---------------------------------------------------------------- Table 1: host contracts
t = r"""\begin{table}[t]
\centering
\caption{Host contracts: the operating point each published tracker declares, as used by the layer. $\tau$ is the tracker's own published threshold, which differs between trackers and, for TrackTrack, between sequences.}
\label{tab:contracts}
\small
\begin{tabular}{p{3.4cm}cccccp{3.6cm}}
\hline
Tracker (setting) & Type & assoc & birth & low & match & Decision mapped to \\
\hline
ByteTrack (official MOT17) & two-stage & 0.6 & 0.7 & 0.1 & 0.8 & thresholds, candidates, scores \\
ByteTrack (library default) & two-stage & 0.25 & 0.25 & 0.1 & 0.8 & same \\
SparseTrack & two-stage & $\tau$ & $\tau+0.1$ & 0.1 & published & same \\
BoostTrack & two-stage & $\tau$ & $\tau$ & 0.1 & $1-$IoU gate & same \\
Hybrid-SORT & two-stage & $\tau$ & $\tau$ & 0.1 & $1-$IoU gate & same \\
OC-SORT & single-stage & 0.6 & 0.6 & 0.6 & 0.7 & same \\
PD-SORT & single-stage & $\tau$ & $\tau$ & $\tau$ & $1-$IoU gate & same \\
C-TWiX (MOT17 / KITTI / DanceTrack) & single-stage & 0.5 & 0.7 / 0.5 / 0.9 & 0.5 & not mapped & candidates, scores, two score thresholds \\
TrackTrack & two-view & $\tau_{\mathrm{det}}$ & $\tau_{\mathrm{init}}$ & 0.1 & not mapped & candidates in both views, two thresholds \\
\hline
\end{tabular}
\end{table}
"""
write("tab1_contracts.tex", t)

# ---------------------------------------------------------------- Table 2: data and roles
t = r"""\begin{table}[t]
\centering
\caption{Evaluation cells and their role. Development cells were visible while the layer was designed; predeclared and post-freeze cells were run once with the frozen policy.}
\label{tab:data}
\small
\begin{tabular}{p{3.0cm}p{3.6cm}p{4.6cm}p{3.2cm}}
\hline
Data & Detector & Trackers & Role \\
\hline
MOT17 val-half (7 sequences) & published YOLOX-X detections & ByteTrack (2 settings), OC-SORT, BoostTrack, SparseTrack & development \\
KITTI tracking training (21 sequences) & YOLOv8n, RT-DETR-L & ByteTrack, BoT-SORT, OC-SORT & development (detector transfer) \\
MOT17 val-half, score transforms & published YOLOX-X, transformed scores & ByteTrack (2 settings), OC-SORT, BoostTrack & development (robustness) \\
MOT17 val-half & released PD-SORT / Hybrid-SORT detections & PD-SORT, Hybrid-SORT & predeclared external \\
MOT17 val-half, KITTI MOTS val (9 sequences), DanceTrack val (25 sequences) & authors' released detections & C-TWiX, TrackTrack (DanceTrack), TOPICTrack (MOT17) & post-freeze external \\
\hline
\end{tabular}
\end{table}
"""
write("tab2_data.tex", t)

# ---------------------------------------------------------------- Table 3: MOT17 development
t = r"""\begin{table}[t]
\centering
\caption{Development hosts on MOT17 val-half (published YOLOX-X detections). Host = our reproduction; $\Delta$ = host with the frozen layer minus host, with 95\% paired-bootstrap interval (10,000 resamples, seed 42); bold intervals exclude zero. The floor is the lowest detector score emitted.}
\label{tab:mot17}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{llllccc}
\hline
Tracker & Setting & Paper HOTA / MOTA / IDF1 & Host HOTA / MOTA / IDF1 & $\Delta$HOTA & $\Delta$MOTA & $\Delta$IDF1 \\
\hline
"""
ref = {"SparseTrack": E.PAPER_REF["SparseTrack"], "BoostTrack online": E.PAPER_REF["BoostTrack online"],
       "BoostTrack + GBI": E.PAPER_REF["BoostTrack + GBI"], "ByteTrack": E.PAPER_REF["ByteTrack"], "OC-SORT": E.PAPER_REF["OC-SORT"]}
for h, v, host, dh, dm, di in E.MOT17_DEV:
    r = ref.get(f"{h} {v}".strip(), ref.get(h, "--"))
    t += f"{h} & {v or '--'} & {r} & {hmi(host)} & {ci(dh)} & {ci(dm)} & {ci(di)} \\\\\n"
t += r"""\hline
\end{tabular}}
\end{table}
"""
write("tab3_mot17.tex", t)

# ---------------------------------------------------------------- Table 4: KITTI
t = r"""\begin{table}[t]
\centering
\caption{Detector transfer on KITTI tracking training (official HOTA, car and pedestrian averaged). BoT-SORT excludes one sequence on which the unmodified tracker fails numerically.}
\label{tab:kitti}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccc}
\hline
Tracker & Detector & Host HOTA / MOTA / IDF1 & $\Delta$HOTA & $\Delta$MOTA & $\Delta$IDF1 \\
\hline
"""
for h, det, n, host, dh, dm, di in E.KITTI:
    t += f"{h} & {det} & {hmi(host)} & {ci(dh)} & {ci(dm)} & {ci(di)} \\\\\n"
t += r"""\hline
\end{tabular}}
\end{table}
"""
write("tab4_kitti.tex", t)

# ---------------------------------------------------------------- Table 5: external
t = r"""\begin{table}[t]
\centering
\caption{External systems evaluated once with the frozen layer. Reproduction class against the authors' reported number (EXACT: $|\Delta\mathrm{HOTA}|\le0.2$; CLOSE: $\le1.0$ with a documented, untuned cause). TOPICTrack failed reproduction and is excluded from the analysis.}
\label{tab:external}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{llllccc}
\hline
System & Data & Reference / reproduced HOTA & Class & $\Delta$HOTA & $\Delta$MOTA & $\Delta$IDF1 \\
\hline
\multicolumn{7}{l}{\textit{Predeclared before the runs}}\\
"""
for n, refs, host, dh, dm, di in E.EXT_PRE:
    cls = "EXACT (released output)" if n == "PD-SORT" else "CLOSE"
    t += f"{n} & MOT17 val-half & {refs.split(' ')[0]} / {host[0]:.2f} & {cls} & {ci(dh)} & {ci(dm)} & {ci(di)} \\\\\n"
t += r"\multicolumn{7}{l}{\textit{Post-freeze, protocol registered before the runs}}\\" + "\n"
labds = {"MOT17": "MOT17 val-half", "KITTIMOT": "KITTI MOTS val", "DanceTrack": "DanceTrack val", "DanceTrack_post": "DanceTrack val", "MOT17_post": "MOT17 val-half"}
for key, s in E.RECENT["systems"].items():
    for ds, r in s["runs"].items():
        if key == "tracktrack" and not ds.endswith("_post"):
            continue
        if key == "topictrack" and not ds.endswith("_post"):
            continue
        for c, v in r["classes"].items():
            b = v["baseline"]["pooled"]["HOTA"]
            refv = r["reference"].get(c, r["reference"]).get("HOTA") if isinstance(r["reference"].get(c), dict) else r["reference"].get("HOTA")
            cls = r["reproduction"].split(":")[0]
            if key == "ctwix" and ds == "KITTIMOT":
                cls = "EXACT" if c == "car" else "CLOSE"
            d = {m: (v["delta"][m]["diff"], v["delta"][m]["ci_lo"], v["delta"][m]["ci_hi"]) for m in ("HOTA", "MOTA", "IDF1")}
            name = s["name"] + (f" ({c})" if len(r["classes"]) > 1 else "")
            if cls == "FAILED":
                t += f"{name} & {labds[ds]} & {refv} / {b:.2f} & FAILED & \\multicolumn{{3}}{{c}}{{not analyzed}} \\\\\n"
            else:
                t += f"{name} & {labds[ds]} & {refv} / {b:.2f} & {cls} & {ci(d['HOTA'])} & {ci(d['MOTA'])} & {ci(d['IDF1'])} \\\\\n"
t += r"""\hline
\end{tabular}}
\end{table}
"""
write("tab5_external.tex", t)

# ---------------------------------------------------------------- Table 6: ablation
t = r"""\begin{table}[t]
\centering
\caption{Design steps from the prior always-intervening design to the frozen policy (development data, HOTA \%), and variants tested and rejected.}
\label{tab:ablation}
\small
\begin{tabular}{p{3.1cm}p{4.8cm}p{6.9cm}}
\hline
Step & Change & Effect \\
\hline
"""
for a, b, c in E.ABLATION:
    t += f"{a} & {b} & {c} \\\\\n"
t += r"\hline" + "\n" + r"\multicolumn{2}{l}{\textit{Rejected variant}} & \textit{Evidence} \\" + "\n"
for a, b in E.REJECTED:
    t += f"\\multicolumn{{2}}{{p{{8.0cm}}}}{{{a}}} & {b} \\\\\n"
t += r"""\hline
\end{tabular}
\end{table}
"""
write("tab6_ablation.tex", t)

# ---------------------------------------------------------------- Table 7: runtime
t = r"""\begin{table}[t]
\centering
\caption{Per-frame time on a 4-vCPU Intel Xeon (2.10 GHz, no graphics processor), live YOLOv8n (736 pixels) and ByteTrack on three KITTI sequences (2,279 timed frames per arm), mean / 95th percentile in milliseconds.}
\label{tab:runtime}
\small
\begin{tabular}{lcc}
\hline
Stage & Host alone & Host + V7f \\
\hline
"""
for st, b, v in E.RUNTIME:
    f = lambda x: f"{x[0]:.2f} / {x[1]:.2f}" if x else "--"
    t += f"{st} & {f(b)} & {f(v)} \\\\\n"
t += r"""\hline
\end{tabular}
\end{table}
"""
write("tab7_runtime.tex", t)
