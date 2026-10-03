"""
Tables, value macros and machine-readable number list of the General AC-MOT manuscript.

Every cell comes from the evidence registry (scripts/evidence.py, pinned commit) or from
figures/fig_data.json (descriptive statistics of the frozen release caches, written by
make_figures.py). Writes:

  tables/*.tex            LaTeX tables input by manuscript.tex
  tables/numbers.tex      \\V{key} / \\CI{key} value macros for the running text
  tables/text_numbers.json  every registered value with its source
  tables/central_table.csv  full native-vs-V7f matrix ("n/m" = not measured, "n/r" = not reported)
  tables/table_keys.json  registry keys used by each table (consumed by check_numbers.py)

    python scripts/make_tables.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence as E  # noqa: E402

R = E.build()
OUT = E.PAPER / "tables"
OUT.mkdir(exist_ok=True)
FIG = json.loads((E.PAPER / "figures" / "fig_data.json").read_text())
USED: dict[str, set] = {}
_current = None


def v(key, ci=False):
    USED.setdefault(_current, set()).add(key)
    x = R[key]
    return f"{x.text} {x.citext}" if ci and x.citext else x.text


def c(key):
    USED.setdefault(_current, set()).add(key)
    return R[key].citext


def write(name, body):
    (OUT / name).write_text(body)


def table(label, caption, cols, header, rows, size="\\footnotesize", star=False, notes=None, side=False):
    # The class wraps table/sidewaystable in threeparttable, which cannot hold a width-limited box;
    # the original environments (saved by the class as tableorg/sidewaystableorg) are used instead.
    env = "sidewaystableorg" if side else "tableorg"
    s = [f"\\begin{{{env}}}" + ("" if side else "[!htbp]"), "\\centering", f"\\caption{{{caption}}}",
         f"\\label{{{label}}}", size,
         "\\setlength\\tabcolsep{3.5pt}", "\\begin{adjustbox}{max width=\\linewidth}",
         f"\\begin{{tabular}}{{{cols}}}", "\\toprule", header + " \\\\", "\\midrule"]
    for r in rows:
        s.append(r if r.startswith("\\") else r + " \\\\")
    s.append("\\botrule")
    s.append("\\end{tabular}")
    s.append("\\end{adjustbox}")
    if notes:
        s.append("\\par\\smallskip\\parbox{\\linewidth}{\\scriptsize " + notes + "}")
    s.append(f"\\end{{{env}}}")
    return "\n".join(s) + "\n"


def sec(title, ncol):
    return f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{title}}}}} \\\\"


def pm(x, nd=2):
    return E.num(x, nd, signed=True)


# ============================================================================ Stage 1
def tab_stage1():
    global _current
    _current = "tab_stage1"
    rows = [sec("VisDrone2019-MOT test-dev, 17 sequences, locked one-shot comparison", 8)]
    for k, lab in [("base", "Static operating point, default tracker"), ("heur", "Hand-designed SCI controller"),
                   ("q", "Calibrated SCI controller (AC-MOT)")]:
        rows.append(f"{lab} & {v(f's1.{k}.HOTA')} & {v(f's1.{k}.MOTA')} & {v(f's1.{k}.IDF1')} & {v(f's1.{k}.IDS')} & "
                    f"{v(f's1.{k}.FP')} & {v(f's1.{k}.FN')} & {v(f's1.{k}.FPS')}")
    rows.append(f"$\\Delta$ AC-MOT $-$ static [95\\% CI] & {v('s1.qb.dHOTA', True)} & {v('s1.qb.dMOTA', True)} & "
                f"{v('s1.qb.dIDF1', True)} & \\multicolumn{{4}}{{l}}{{IDS reduction {v('s1.qb.dIDS_reduction', True)}; "
                f"W/T/L {v('s1.qb.wtl')}}}")
    rows.append(f"$\\Delta$ AC-MOT $-$ hand-designed [95\\% CI] & {v('s1.qh.dHOTA', True)} & {v('s1.qh.dMOTA', True)} & "
                f"{v('s1.qh.dIDF1', True)} & \\multicolumn{{4}}{{l}}{{IDS reduction {v('s1.qh.dIDS_reduction', True)}; "
                f"W/T/L {v('s1.qh.wtl')}}}")
    rows.append("\\midrule")
    rows.append(sec("UAVDT test, 20 sequences, frozen controller, no retuning", 8))
    for k, lab in [("base", "Static operating point, default tracker"), ("q", "Calibrated SCI controller (AC-MOT)")]:
        rows.append(f"{lab} & {v(f'uav.{k}.HOTA')} & {v(f'uav.{k}.MOTA')} & {v(f'uav.{k}.IDF1')} & {v(f'uav.{k}.IDS')} & "
                    f"{v(f'uav.{k}.FP')} & {v(f'uav.{k}.FN')} & {v(f'uav.{k}.FPS')}")
    rows.append(f"$\\Delta$ AC-MOT $-$ static [95\\% CI] & {v('uav.qb.dHOTA', True)} & {v('uav.qb.dMOTA', True)} & "
                f"{v('uav.qb.dIDF1', True)} & \\multicolumn{{4}}{{l}}{{IDS reduction {v('uav.qb.dIDS_reduction', True)}; "
                f"W/T/L {v('uav.qb.wtl')}}}")
    cap = ("Stage 1: the scene-adaptive AC-MOT controller with YOLOv8n and ByteTrack. Custom class-agnostic protocol "
           "(TrackEval), not the official VisDrone toolkit. Intervals: paired sequence bootstrap, 5,000 resamples, "
           "seed 42. FPS is processing throughput on decoded frames on a Tesla T4, each system in its own session.")
    write("tab_stage1.tex", table("tab:stage1", cap, "p{4.6cm}rrrrrrr",
                                  "System & HOTA & MOTA & IDF1 & IDS & FP & FN & FPS", rows, star=True))


def tab_attribution():
    global _current
    _current = "tab_attribution"
    rows = [sec("Held-out test-dev (post hoc attribution, 17 sequences)", 5),
            f"Calibrated SCI controller (AC-MOT) & {v('s1.q.HOTA')} & {v('s1.q.MOTA')} & {v('s1.q.IDF1')} & {v('s1.q.IDS')}",
            f"Matched static point ({v('ms.a0.res')} px, conf {v('ms.a0.conf')}, NMS {v('ms.a0.nms')}) & {v('ms.a0.HOTA')} & "
            f"{v('ms.a0.MOTA')} & {v('ms.a0.IDF1')} & {v('ms.a0.IDS')}",
            f"$\\Delta$ controller $-$ matched static [95\\% CI] & {v('ms.dHOTA', True)} & {v('ms.dMOTA', True)} & "
            f"{v('ms.dIDF1', True)} & {v('ms.dIDS_reduction', True)}$^{{a}}$",
            "\\midrule",
            sec("Validation (VisDrone2019-MOT val, 7 sequences): decomposition of the gain, HOTA", 5),
            f"Static 640 px, conf 0.25, default tracker & {v('val.static_default.HOTA')} & \\multicolumn{{3}}{{l}}{{reference}}",
            f"+ tuned tracker profile & {v('val.static_tuned.HOTA')} & \\multicolumn{{3}}{{l}}{{{v('val.d_tracker')}}}",
            f"+ calibrated static operating point (960 px, 0.35, 0.35) & {v('val.a0.HOTA')} & \\multicolumn{{3}}{{l}}{{{v('val.d_operating')}}}",
            f"+ scene switching (calibrated controller) & {v('val.q.HOTA')} & \\multicolumn{{3}}{{l}}{{{v('val.d_switch')}}}"]
    notes = (f"$^{{a}}$ IDS reduction (positive = fewer switches for the controller). The calibrated controller ran at a mean "
             f"input of {v('val.q.res')} px, mean confidence {v('val.q.conf')} and constant NMS {v('val.q.nms')} on "
             f"validation; the matched static point was run on 2026-09-18, after the freeze, and is used only for attribution.")
    cap = ("Attribution of the Stage-1 gain. The matched static operating point reproduces the calibrated controller "
           "within the resolution of the test; on validation, almost all of the gain over the static default comes from the "
           "tracker profile and the calibrated operating point, not from frame-to-frame switching.")
    write("tab_attribution.tex", table("tab:attribution", cap, "p{6.2cm}rrrr", "System & HOTA & MOTA & IDF1 & IDS",
                                       rows, star=True, notes=notes))


# ============================================================================ Score scales
def tab_rawscale():
    global _current
    _current = "tab_rawscale"
    rows = []
    for det, lab in [("yolov8", "YOLOv8n"), ("rtdetr", "RT-DETR-L"), ("fasterrcnn", "Faster R-CNN R50-FPN v2"),
                     ("retinanet", "RetinaNet R50-FPN v2")]:
        f = FIG["fig3"][det]
        rows.append(f"{lab} & {f['per_frame']:.0f} & {100 * f['share_ge_025']:.1f} & {f['t1_median_score']:.2f} & "
                    f"{f['t2_median_score']:.2f} & {v(f'raw.{det}.HOTA')} & {v(f'raw.{det}.MOTA')} & "
                    f"{v(f'raw.{det}.Precision')} & {v(f'raw.{det}.Recall')} & {v(f'raw.{det}.cat')}")
    cap = ("One raw threshold, four detectors: VisDrone2019-MOT val (7 sequences, 2,846 frames), 736 px input, score "
           "floor 0.01, ByteTrack with its library default thresholds (0.25). Columns 2--5 describe the cached detector "
           "streams: candidates per frame, share of candidates with score $\\geq 0.25$, and the median over frames of the "
           "self-calibrated thresholds $t_1$ and $t_2$ (nested Otsu on the logits of the ten previous frames, shown as scores). "
           "Columns 6--10: tracking with the raw threshold (internal protocol); `Cat.' counts sequences with MOTA $<0$.")
    notes = ("Stream statistics are computed from the published detector caches (release assets "
             "\\texttt{v7-dev-assets-1} and \\texttt{sci-v7f-unseen-1}, checked by SHA-256) with the frozen thresholding "
             "function; duplicate handling is not applied in these statistics. RetinaNet entered the project only after the "
             "V7f freeze.")
    write("tab_rawscale.tex", table("tab:rawscale", cap, "lrrrrrrrrr",
                                    "Detector & Cand./frame & $\\geq0.25$ (\\%) & $\\sigma(t_1)$ & $\\sigma(t_2)$ & HOTA & "
                                    "MOTA & Prec. & Rec. & Cat.", rows, star=True, notes=notes))


def tab_calib():
    global _current
    _current = "tab_calib"
    host = {"byoff": "ByteTrack (official setting)", "byul": "ByteTrack (library default)", "oc": "OC-SORT",
            "bt": "BoostTrack (online)"}
    tr = {"pow3": "$s^3$", "scale05": "$0.5\\,s$", "temp2": "logit $/2$", "temp05": "logit $\\times 2$"}
    rows = []
    for h in ("byoff", "byul", "oc", "bt"):
        for t in ("pow3", "scale05", "temp2", "temp05"):
            k = f"cal.{h}.{t}"
            if k + ".d" not in R:
                continue
            verdict = R[k + ".verdict"].value
            rows.append(f"{host[h]} & {tr[t]} & {v(k + '.native')} & {v(k + '.v7f')} & {v(k + '.d', True)} & "
                        f"{v(k + '.wtl')} & {verdict}")
    cap = (f"Detector-score calibration shift on MOT17 val-half (published YOLOX-X detections, development hosts). The "
           f"detector scores are transformed before the tracker; the frozen V7f layer is not changed. HOTA of the host alone "
           f"and with V7f; paired sequence bootstrap, {v('cal.resamples')} resamples, seed 42. {v('cal.n')} conditions: "
           f"{v('cal.rec')} significant recoveries, {v('cal.deg')} significant degradations, {v('cal.ident')} identical outputs.")
    write("tab_calib.tex", table("tab:calib", cap, "llrrlcl",
                                 "Host & Transform & Host alone & + V7f & $\\Delta$HOTA [95\\% CI] & W/T/L & Verdict",
                                 rows, star=True))


# ============================================================================ Method tables
def tab_contracts():
    global _current
    _current = "tab_contracts"
    src = "JAIS_PAPERS/PAPER2_SIVP_SPRINGER/tables/tab1_contracts.tex"
    rows_src = [
        ("ByteTrack (library default; also BoT-SORT)", "two-stage", "0.25", "0.25", "0.1", "0.8",
         "ByteTrack (library default) & two-stage & 0.25 & 0.25 & 0.1 & 0.8"),
        ("ByteTrack (official MOT17)", "two-stage", "0.6", "0.7", "0.1", "0.8",
         "ByteTrack (official MOT17) & two-stage & 0.6 & 0.7 & 0.1 & 0.8"),
        ("SparseTrack", "two-stage", "$\\tau$", "$\\tau+0.1$", "0.1", "published",
         "SparseTrack & two-stage & $\\tau$ & $\\tau+0.1$ & 0.1 & published"),
        ("BoostTrack", "two-stage", "$\\tau$", "$\\tau$", "0.1", "$1-$IoU gate",
         "BoostTrack & two-stage & $\\tau$ & $\\tau$ & 0.1 & $1-$IoU gate"),
        ("Hybrid-SORT", "two-stage", "$\\tau$", "$\\tau$", "0.1", "$1-$IoU gate",
         "Hybrid-SORT & two-stage & $\\tau$ & $\\tau$ & 0.1 & $1-$IoU gate"),
        ("OC-SORT", "single-stage", "0.6", "0.6", "0.6", "0.7", "OC-SORT & single-stage & 0.6 & 0.6 & 0.6 & 0.7"),
        ("PD-SORT", "single-stage", "$\\tau$", "$\\tau$", "$\\tau$", "$1-$IoU gate",
         "PD-SORT & single-stage & $\\tau$ & $\\tau$ & $\\tau$ & $1-$IoU gate"),
        ("C-TWiX (MOT17 / KITTI / DanceTrack)", "single-stage", "0.5", "0.7 / 0.5 / 0.9", "0.5", "not mapped",
         "C-TWiX (MOT17 / KITTI / DanceTrack) & single-stage & 0.5 & 0.7 / 0.5 / 0.9 & 0.5 & not mapped"),
        ("TrackTrack", "two-view", "$\\tau_{\\mathrm{det}}$", "$\\tau_{\\mathrm{init}}$", "0.1", "not mapped",
         "TrackTrack & two-view & $\\tau_{\\mathrm{det}}$ & $\\tau_{\\mathrm{init}}$ & 0.1 & not mapped"),
    ]
    rows = []
    for i, (lab, typ, a, b, lo, m, frag) in enumerate(rows_src):
        E.transcribed(f"contract.{i}", lab, src, frag)
        USED.setdefault(_current, set()).add(f"contract.{i}")
        rows.append(f"{lab} & {typ} & {a} & {b} & {lo} & {m}")
    cap = ("Host contracts: the operating point each tracker declares to the layer (association threshold, birth threshold, "
           "lowest score used by any association stage, matching tolerance). $\\tau$ is the tracker's own published "
           "threshold. A host is single-stage when its lowest stage equals its association/birth threshold. The contract "
           "carries no tracker identity.")
    write("tab_contracts.tex", table("tab:contracts", cap, "p{5.0cm}lcccl",
                                     "Tracker (setting) & Type & assoc & birth & low & match", rows, star=True))


def tab_constants():
    global _current
    _current = "tab_constants"
    code = "acmot_v7.py"
    cfg = E.J("configs/universal_acmot_policy_v7.json")["spec"]
    E.put("const.window", cfg["window"], "configs/universal_acmot_policy_v7.json::spec.window", nd=0)
    E.put("const.dup_iou", cfg["dup_iou"], "configs/universal_acmot_policy_v7.json::spec.dup_iou", nd=1)
    E.put("const.hist", cfg["hist"], "configs/universal_acmot_policy_v7.json::spec.hist", nd=0)
    E.put("const.warmup", cfg["warmup"], "configs/universal_acmot_policy_v7.json::spec.warmup", nd=0)
    frag = [("const.regime", "$\\bar\\rho_t \\geq 0.5$", "(s.regime == \"rho\" and rho_bar >= 0.5))"),
            ("const.minlogits", "3", "if len(H) < 3:"),
            ("const.remap", "$0.5+0.5u$ / $0.1+0.4u$", "out = np.where(Lp >= ta, 0.5 + 0.5 * u, 0.1 + 0.4 * u)"),
            ("const.ecdfwin", "20", "self.ecdf_samples = deque(maxlen=20)"),
            ("const.ecdfstride", "10", "(self.frame == 1 or self.frame % 10 == 0)"),
            ("const.ecdfclip", "$10^{-6}$", "return np.clip((lo + hi) / (2.0 * len(ref)), 1e-6, 1 - 1e-6)"),
            ("const.margin", "$10^{-3}$", "else floor_pass + 1e-3 if s.rescue_band == \"fg\""),
            ("const.cap", "0.95", "match = min(0.95, 1.0 - (1.0 - h.match) / max(1.0, r))"),
            ("const.logitclip", "$10^{-9}$", "s = np.clip(np.asarray(s, dtype=np.float64), 1e-9, 1 - 1e-9)")]
    for k, val, f in frag:
        E.transcribed(k, val, code, f)
    for k in ["const.window", "const.dup_iou", "const.hist", "const.warmup"] + [f[0] for f in frag]:
        USED.setdefault(_current, set()).add(k)
    rows = [
        f"History window $W$ (frames) & {R['const.window'].text} & structural; insensitivity checked in development (5/20) & config \\texttt{{window}}",
        f"Minimum pooled logits for valid bands & {R['const.minlogits'].text} & structural (Otsu needs a split) & \\texttt{{\\_bands}}",
        "Thresholds $t_1$, $t_2$ & data & online, nested exact Otsu & \\texttt{nested\\_otsu}",
        f"Regime boundary & {R['const.regime'].text} & design constant chosen on development data & \\texttt{{step}}",
        f"Overlap rule (duplicates, track claim, rescue) & IoU {R['const.dup_iou'].text} & conventional correspondence threshold & config \\texttt{{dup\\_iou}}",
        f"Rank remap (noisy regime) & {R['const.remap'].text} & structural (order-preserving, two ranges) & \\texttt{{step}}",
        f"ECDF reference: stride / samples & {R['const.ecdfstride'].text} / {R['const.ecdfwin'].text} & structural & \\texttt{{step}}",
        f"Motion history / warm-up & {R['const.hist'].text} / {R['const.warmup'].text} & structural & config \\texttt{{hist}}, \\texttt{{warmup}}",
        f"Matching-tolerance cap & {R['const.cap'].text} & safety bound & \\texttt{{step}}",
        f"Rescue margin above the host's lowest usable score & {R['const.margin'].text} & numerical margin & \\texttt{{step}}",
        f"Clipping (ECDF / logit) & {R['const.ecdfclip'].text} / {R['const.logitclip'].text} & numerical & \\texttt{{\\_ecdf}}, \\texttt{{logit}}",
        "Host thresholds and matching tolerance & declared & host contract (Table~\\ref{tab:contracts}) & host adapter",
    ]
    cap = ("Constants of the frozen V7f layer (\\texttt{acmot\\_v7.py} and \\texttt{configs/universal\\_acmot\\_policy\\_v7.json} "
           "at tag \\texttt{universal-acmot-v7-freeze}). No constant names a detector, tracker, dataset or sequence; none is "
           "fitted at deployment. The design constants were fixed during development, which used labelled development data.")
    write("tab_constants.tex", table("tab:constants", cap, "p{4.6cm}lp{5.0cm}l", "Quantity & Value & Nature & Location",
                                     rows, star=True))


def tab_protocol():
    global _current
    _current = "tab_protocol"
    rows = [
        "VisDrone2019-MOT test-dev & Stage 1 & YOLOv8n & ByteTrack & held-out, locked one-shot (Stage 1) & custom class-agnostic",
        "UAVDT test & Stage 1 & YOLOv8n & ByteTrack & frozen transfer, no retuning (Stage 1) & UAVDT adapter, IoU 0.5",
        "VisDrone2019-MOT val (7 seq.) & V7f, General AC-MOT & YOLOv8n, RT-DETR-L & ByteTrack, BoT-SORT, OC-SORT & development & internal and official-compatible",
        "VisDrone2019-MOT val (7 seq.) & General AC-MOT & RetinaNet & ByteTrack, BoT-SORT & unseen detector (locked), development sequences & internal and official-compatible",
        "KITTI tracking training (21 seq.) & V7f & YOLOv8n, RT-DETR-L & ByteTrack, BoT-SORT, OC-SORT & development & KITTI HOTA (car/pedestrian mean)",
        "MOT17 val-half (7 seq.) & V7f & YOLOX-X (published) & SparseTrack, BoostTrack, ByteTrack, OC-SORT & development & TrackEval, internal",
        "MOT17 val-half (7 seq.) & V7f & YOLOX-X (published) & PD-SORT, Hybrid-SORT & predeclared external & TrackEval, internal",
        "MOT17 val-half / KITTIMOTS val / DanceTrack val & V7f & released detections (incl. PermaTrack) & C-TWiX, TrackTrack & post-freeze external & TrackEval, internal",
    ]
    cap = ("Evaluation settings and the role of each. `Development' data were visible while the corresponding system was "
           "designed; `external' systems were run once with the frozen policy. No result in this paper is a test-server or "
           "leaderboard submission.")
    write("tab_protocol.tex", table("tab:protocol", cap, "p{2.9cm}p{1.8cm}p{2.0cm}p{2.4cm}p{2.6cm}p{2.3cm}",
                                    "Data & System & Detector(s) & Tracker(s) & Role & Evaluation", rows, star=True))


# ============================================================================ Central table
CENTRAL = [
    # (group, detector, tracker, status, keys)  keys: dict(base, v7f, d, dprefix, fp/fn/p/r prefixes)
    ("VisDrone val-7", "YOLOv8n", "ByteTrack", "D", dict(b="vd.yolov8.base.int", g="vd.yolov8.g1.int", d="vd.yolov8.int")),
    ("VisDrone val-7", "RT-DETR-L", "ByteTrack", "D", dict(b="vd.rtdetr.base.int", g="vd.rtdetr.g1.int", d="vd.rtdetr.int")),
    ("VisDrone val-7", "YOLOv8n", "BoT-SORT", "D", dict(b="bot.yolov8.base", g="bot.yolov8.g1", d="bot.yolov8.int")),
    ("VisDrone val-7", "RT-DETR-L", "BoT-SORT", "D", dict(b="bot.rtdetr.base", g="bot.rtdetr.g1", d="bot.rtdetr.int")),
    ("VisDrone val-7", "YOLOv8n", "OC-SORT", "D", dict(b="oc.yolov8.base", g="oc.yolov8.g1", d="oc.yolov8.int")),
    ("VisDrone val-7", "RT-DETR-L", "OC-SORT", "D", dict(b="oc.rtdetr.base", g="oc.rtdetr.g1", d="oc.rtdetr.int")),
    ("VisDrone val-7", "RetinaNet", "ByteTrack", "U", dict(b="rn.by.int.base", g="rn.by.int.g1", d="rn.by.736.int",
                                                           cnt_b="rn.by.base", cnt_g="rn.by.g1")),
    ("VisDrone val-7", "RetinaNet", "BoT-SORT", "U", dict(b="bot.retinanet.base", g="bot.retinanet.g1", d="bot.retinanet.int")),
    ("KITTI train", "YOLOv8n", "ByteTrack", "D", dict(b="kit.yolov8.base", g="kit.yolov8.v7f", kd="kitd.yolov8.by")),
    ("KITTI train", "YOLOv8n", "BoT-SORT", "D", dict(b0="kit.botbase.yolov8.HOTA", kd="kitd.yolov8.bot")),
    ("KITTI train", "YOLOv8n", "OC-SORT", "D", dict(b="kit.yolov8.ocbase", g="kit.yolov8.ocv7f", kd="kitd.yolov8.oc")),
    ("KITTI train", "RT-DETR-L", "ByteTrack", "D", dict(b="kit.rtdetr.base", g="kit.rtdetr.v7f", kd="kitd.rtdetr.by")),
    ("KITTI train", "RT-DETR-L", "BoT-SORT", "D", dict(b0="kit.botbase.rtdetr.HOTA", kd="kitd.rtdetr.bot")),
    ("KITTI train", "RT-DETR-L", "OC-SORT", "D", dict(b="kit.rtdetr.ocbase", g="kit.rtdetr.ocv7f", kd="kitd.rtdetr.oc")),
    ("MOT17 val-half", "YOLOX-X", "SparseTrack", "D", dict(b="mot.st.base", g="mot.st.v7f", md="mot.st")),
    ("MOT17 val-half", "YOLOX-X", "BoostTrack (online)", "D", dict(b="mot.bt.base", g="mot.bt.v7f", ident=True)),
    ("MOT17 val-half", "YOLOX-X", "ByteTrack (official)", "D", dict(b="mot.byoff.base", g="mot.byoff.v7f", only_h="mot.byoff.dHOTA")),
    ("MOT17 val-half", "YOLOX-X", "ByteTrack (library)", "D", dict(b="mot.byul.base", g="mot.byul.v7f", ident=True)),
    ("MOT17 val-half", "YOLOX-X", "OC-SORT", "D", dict(b="mot.oc.base", g="mot.oc.v7f", md3="mot.oc")),
    ("MOT17 val-half", "YOLOX-X", "PD-SORT", "P", dict(b="ext.pd.base", g="ext.pd.v7f", md="ext.pd")),
    ("MOT17 val-half", "YOLOX-X", "Hybrid-SORT", "P", dict(b="ext.hy.base", g="ext.hy.v7f", ident=True)),
    ("MOT17 val-half", "YOLOX-X", "C-TWiX", "X", dict(b="ctx.mot.base", g="ctx.mot.v7f", md="ctx.mot")),
    ("KITTIMOTS val", "PermaTrack", "C-TWiX (car)", "X", dict(b="ctx.kcar.base", g="ctx.kcar.v7f", md="ctx.kcar")),
    ("KITTIMOTS val", "PermaTrack", "C-TWiX (ped.)", "X", dict(b="ctx.kped.base", g="ctx.kped.v7f", md="ctx.kped")),
    ("DanceTrack val", "released", "C-TWiX", "X", dict(b="ctx.dance.base", g="ctx.dance.v7f", md="ctx.dance")),
    ("DanceTrack val", "released", "TrackTrack", "X", dict(b="tt.dance.base", g="tt.dance.v7f", md="tt.dance")),
]


def _cell(prefix, m):
    k = f"{prefix}.{m}"
    return v(k) if k in R else "n/r"


def _arrow(kb, kg, m):
    a = f"{kb}.{m}" if kb else None
    b = f"{kg}.{m}" if kg else None
    if a in R and b in R:
        return f"{v(a)}$\\to${v(b)}"
    return "n/r"


def central_cells(row):
    g, det, trk, st, k = row
    out = dict(group=g, detector=det, tracker=trk, status=st)
    kb, kg = k.get("b"), k.get("g")
    out["base"] = v(k["b0"]) if "b0" in k else _cell(kb, "HOTA")
    out["v7f"] = _cell(kg, "HOTA") if kg else "n/r"
    if k.get("ident"):
        out.update(dH="0 (identical)", dM="0", dI="0", dS="0")
    elif "d" in k:
        p = k["d"]
        out.update(dH=v(p + ".dHOTA", True), dM=v(p + ".dMOTA"), dI=v(p + ".dIDF1"), dS=v(p + ".dIDS"))
    elif "kd" in k:
        p = k["kd"]
        out.update(dH=v(p + ".dHOTA", True), dM=v(p + ".dMOTA"), dI=v(p + ".dIDF1"), dS=v(p + ".dIDS"))
    elif "md" in k:
        p = k["md"]
        out.update(dH=v(p + ".dHOTA", True), dM=v(p + ".dMOTA"), dI=v(p + ".dIDF1"), dS=v(p + ".dIDS"))
    elif "md3" in k:
        p = k["md3"]
        out.update(dH=v(p + ".dHOTA", True), dM=v(p + ".dMOTA"), dI=v(p + ".dIDF1"),
                   dS=E.num(R[kg + ".IDS"].value - R[kb + ".IDS"].value, 0, True))
        USED[_current].update({kg + ".IDS", kb + ".IDS"})
    else:
        out.update(dH=v(k["only_h"], True), dM=E.num(R[kg + ".MOTA"].value - R[kb + ".MOTA"].value, 3, True),
                   dI=E.num(R[kg + ".IDF1"].value - R[kb + ".IDF1"].value, 3, True),
                   dS=E.num(R[kg + ".IDS"].value - R[kb + ".IDS"].value, 0, True))
        USED[_current].update({kg + ".MOTA", kb + ".MOTA", kg + ".IDF1", kb + ".IDF1", kg + ".IDS", kb + ".IDS"})
    cb, cg = k.get("cnt_b", kb), k.get("cnt_g", kg)
    for m, name in [("FP", "FP"), ("FN", "FN"), ("Precision", "P"), ("Recall", "R")]:
        out[name] = _arrow(cb, cg, m)
    return out


def tab_central():
    global _current
    _current = "tab_central"
    rows, csvrows = [], []
    group = None
    for row in CENTRAL:
        x = central_cells(row)
        if x["group"] != group:
            rows.append(sec(x["group"], 12))
            group = x["group"]
        rows.append(f"{x['detector']} & {x['tracker']} & {x['status']} & {x['base']} & {x['v7f']} & {x['dH']} & {x['dM']} & "
                    f"{x['dI']} & {x['dS']} & {x['FP']} & {x['FN']} & {x['P']} / {x['R']}")
        csvrows.append({k: (val.replace(E.MINUS, "-").replace("$\\to$", " -> ") if isinstance(val, str) else val)
                        for k, val in x.items()})
    notes = ("Status: D development (data or host visible while V7f was designed), U unseen detector evaluated under a transfer "
             "lock on development sequences, P predeclared external system, X post-freeze external system. VisDrone rows use "
             "the frozen General AC-MOT configuration (V7f at 736 px) and the internal protocol; official-compatible values "
             "are in Tables~\\ref{tab:visdrone_official} and~\\ref{tab:retinanet}. KITTI: official KITTI HOTA script, "
             "car/pedestrian mean; BoT-SORT cells exclude sequence 0020 (host numerical failure); FP, FN, precision and recall "
             "are not reported as pooled values (n/r). MOT17 ByteTrack (official) and OC-SORT use the 0.01 score floor. "
             "Intervals: paired sequence bootstrap, 10,000 resamples, seed 42. Runtime is reported separately "
             "(Table~\\ref{tab:runtime}) because it was measured on three configurations only.")
    cap = ("Central result: the host alone versus the host with the frozen layer, for every detector--tracker--dataset "
           "combination with a recorded result. HOTA, MOTA, IDF1 in points; $\\Delta$ = host + V7f $-$ host alone.")
    write("tab_central.tex", table("tab:central", cap, "lllrrlrrrlll",
                                   "Detector & Tracker & St. & Host & +V7f & $\\Delta$HOTA [95\\% CI] & $\\Delta$MOTA & "
                                   "$\\Delta$IDF1 & $\\Delta$IDS & FP & FN & Prec. / Rec.", rows, size="\\scriptsize",
                                   notes=notes, side=True))
    with open(OUT / "central_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csvrows[0].keys()) + ["mean_latency_ms", "p95_latency_ms", "fps", "controller_ms"])
        w.writeheader()
        for r in csvrows:
            lat = {"mean_latency_ms": "n/m", "p95_latency_ms": "n/m", "fps": "n/m", "controller_ms": "n/m"}
            if r["group"] == "VisDrone val-7" and r["tracker"] == "ByteTrack":
                d = {"YOLOv8n": "yolov8", "RT-DETR-L": "rtdetr", "RetinaNet": "retinanet"}[r["detector"]]
                lat = {"mean_latency_ms": f"{R[f'g1rt.{d}.g1.total'].value:.2f}",
                       "p95_latency_ms": f"{R[f'g1rt.{d}.g1.total95'].value:.2f}",
                       "fps": f"{R[f'g1rt.{d}.g1.fps'].value:.3f}", "controller_ms": f"{R[f'g1rt.{d}.g1.score'].value:.2f}"}
                USED[_current].update({f"g1rt.{d}.g1.total", f"g1rt.{d}.g1.total95", f"g1rt.{d}.g1.fps", f"g1rt.{d}.g1.score"})
            r.update(lat)
            w.writerow(r)


def tab_visdrone_official():
    global _current
    _current = "tab_visdrone_official"
    rows = []
    for det, lab in [("yolov8", "YOLOv8n"), ("rtdetr", "RT-DETR-L")]:
        for prot, pl in [("int", "internal"), ("off", "official-compatible")]:
            p = f"vd.{det}.{prot}"
            rows.append(f"{lab} & {pl} & {v(f'vd.{det}.base.{prot}.HOTA')} & {v(f'vd.{det}.g1.{prot}.HOTA')} & "
                        f"{v(p + '.dHOTA', True)} & {v(p + '.dMOTA', True)} & {v(p + '.dIDF1', True)} & {v(p + '.dIDS')}")
    for prot, pl in [("int", "internal"), ("off", "official-compatible")]:
        p = f"vd.pool.{prot}"
        rows.append(f"Pooled (14 cells) & {pl} & -- & -- & {v(p + '.dHOTA', True)} & {v(p + '.dMOTA', True)} & "
                    f"{v(p + '.dIDF1', True)} & {v(p + '.dIDS')}")
    cap = ("Frozen General AC-MOT (V7f at 736 px) versus ByteTrack alone on VisDrone2019-MOT val (development detectors). "
           "Internal: custom class-agnostic protocol. Official-compatible: port of the VisDrone MOT toolkit (class-aware, "
           "ignored regions removed) with HOTA added; it is not the official evaluation server. Paired bootstrap over "
           "sequences (pooled: over detector--sequence cells), 10,000 resamples, seed 42.")
    write("tab_visdrone_official.tex", table("tab:visdrone_official", cap, "llrrllll",
                                             "Detector & Protocol & Host & + V7f & $\\Delta$HOTA & $\\Delta$MOTA & "
                                             "$\\Delta$IDF1 & $\\Delta$IDS", rows, star=True))


def tab_retinanet():
    global _current
    _current = "tab_retinanet"
    rows = []
    for trk, lab, pre in [("by", "ByteTrack", "rn.by.736"), ("bot", "BoT-SORT", "bot.retinanet")]:
        for prot, pl in [("int", "internal"), ("off", "official-compatible")]:
            p = f"{pre}.{prot}"
            rows.append(f"{lab} & {pl} & 736 & {v(p + '.dHOTA', True)} & {v(p + '.dMOTA', True)} & {v(p + '.dIDF1', True)} & "
                        f"{v(p + '.dIDS')} & {v(p + '.dHOTA.wl')}")
    for res in ("640", "832"):
        for prot, pl in [("int", "internal"), ("off", "official-compatible")]:
            p = f"rn.by.{res}.{prot}"
            rows.append(f"ByteTrack & {pl} & {res} & {v(p + '.dHOTA', True)} & {v(p + '.dMOTA', True)} & "
                        f"{v(p + '.dIDF1', True)} & {v(p + '.dIDS')} & {v(p + '.dHOTA.wl')}")
    cap = ("Unseen detector: RetinaNet ResNet50-FPN v2 with the frozen General AC-MOT configuration, VisDrone2019-MOT val. "
           "The transfer lock (\\texttt{research/TRANSFER\\_LOCK\\_RETINANET\\_G1.json}) was committed before any RetinaNet "
           "tracking metric existed; only the detector adapter is new. The sequences are development sequences of V7f. "
           "Rows at 640 and 832 px are the pre-declared compute-curve arms. $\\Delta$ = General AC-MOT $-$ host alone; "
           "+/$-$ counts sequences with higher/lower HOTA.")
    write("tab_retinanet.tex", table("tab:retinanet", cap, "llcllllc",
                                     "Host & Protocol & px & $\\Delta$HOTA [95\\% CI] & $\\Delta$MOTA & $\\Delta$IDF1 & "
                                     "$\\Delta$IDS & +/$-$", rows, star=True))


# ============================================================================ Runtime
def tab_runtime():
    global _current
    _current = "tab_runtime"
    rows = [sec(f"V7f layer, YOLOv8n + ByteTrack, KITTI frames, {R['rt.cpu'].value} ({v('rt.threads')} threads), "
                f"{v('rt.res')} px, {v('rt.v7f.frames')} frames per arm", 8)]
    rows.append(f"Host alone & {v('rt.base.det')} & {v('rt.base.ctrl')} & {v('rt.base.trk')} & {v('rt.base.e2e')} & "
                f"{v('rt.base.e2e95')} & {v('rt.base.fps')} & --")
    rows.append(f"+ V7f & {v('rt.v7f.det')} & {v('rt.v7f.ctrl')} (P95 {v('rt.v7f.ctrl95')}) & {v('rt.v7f.trk')} & "
                f"{v('rt.v7f.e2e')} & {v('rt.v7f.e2e95')} & {v('rt.v7f.fps')} & {v('rt.e2e_delta')} ms ({v('rt.e2e_pct')}\\%)")
    rows.append("\\midrule")
    rows.append(sec(f"General AC-MOT, VisDrone val frames, {R['g1rt.cpu'].value} ({v('g1rt.yolov8.threads')} threads), "
                    f"736 px, {v('g1rt.yolov8.g1.frames')} frames, decode excluded", 8))
    for det, lab in [("yolov8", "YOLOv8n"), ("rtdetr", "RT-DETR-L"), ("retinanet", "RetinaNet")]:
        for arm, al in [("base", "host alone"), ("g1", "General AC-MOT")]:
            last = f"{v(f'g1rt.{det}.pct')}\\%" if arm == "g1" else "--"
            rows.append(f"{lab}, {al} & {v(f'g1rt.{det}.{arm}.det')} & {v(f'g1rt.{det}.{arm}.score')} & "
                        f"{v(f'g1rt.{det}.{arm}.trk')} & {v(f'g1rt.{det}.{arm}.total')} & {v(f'g1rt.{det}.{arm}.total95')} & "
                        f"{v(f'g1rt.{det}.{arm}.fps')} & {last}")
    notes = ("End-to-end time in the first block includes frame reading; the overhead relative to the pipeline without "
             f"reading is {v('rt.pipe_pct')}\\%. The control-layer time includes its quarter-resolution phase-correlation "
             "motion cue. The tracker is faster with the layer because fewer candidates reach it. CPU measurements only; no "
             "GPU timing of V7f or General AC-MOT was available, and no FPS is converted to other hardware.")
    cap = "Runtime per frame (ms). Mean unless marked P95."
    write("tab_runtime.tex", table("tab:runtime", cap, "p{3.6cm}rlrrrrl",
                                   "Configuration & Detector & Control layer & Tracker & Total & P95 total & FPS & Overhead",
                                   rows, star=True, notes=notes))


# ============================================================================ Ablations
def tab_ablation():
    global _current
    _current = "tab_ablation"
    steps = [("native", "Native host through the layer and adapters, no intervention (pass-through)"),
             ("v6", "Self-calibration only: nested-Otsu bands always applied, IoU duplicate rule, rank scores (prior design)"),
             ("v7d", "+ host-aware regime ($\\rho$ over 100 frames), clean regime anchored to the host, crowd-safe duplicates"),
             ("cum", "+ regime estimated from the whole stream"),
             ("v7e", "+ foreground track-consistent continuation"),
             ("v7f", "+ interpretability check of the split = frozen V7f")]
    rows = []
    for s, lab in steps:
        kit = v(f"abl.{s}.kitoc") if f"abl.{s}.kitoc" in R else "--"
        rows.append(f"{lab} & {v(f'abl.{s}.by')} & {v(f'abl.{s}.oc')} & {v(f'abl.{s}.bt')} & {kit}")
    rows.append(f"Host alone (original driver) & {v('abl.host.by')} & {v('abl.host.oc')} & {v('abl.host.bt')} & {v('abl.host.kitoc')}")
    notes = (f"Pass-through: on {v('abl.passthrough.identical')} of {v('abl.passthrough.cells')} MOT17 host/floor cells the "
             "pass-through output equals the host alone in every metric, and the SparseTrack and BoostTrack pass-through "
             "track files are byte-identical to the official replays (development ledger, E0). KITTI column: YOLOv8n + "
             "OC-SORT, KITTI HOTA. All rows are development evidence; no interval is claimed.")
    cap = ("Mechanism ablation, one change at a time (HOTA). MOT17 val-half with published YOLOX-X detections at the 0.1 "
           "score floor, no image motion cue.")
    write("tab_ablation.tex", table("tab:ablation", cap, "p{7.2cm}rrrr",
                                    "Variant & ByteTrack (official) & OC-SORT & BoostTrack & KITTI OC-SORT", rows, star=True,
                                    notes=notes))


def tab_factorial():
    global _current
    _current = "tab_factorial"
    comps = [("c_a", "V7f only $-$ native (C$-$A)"), ("b_a", "historical SCI only $-$ native (B$-$A)"),
             ("d_c", "SCI + V7f $-$ V7f only (D$-$C)"), ("d_b", "SCI + V7f $-$ SCI only (D$-$B)"),
             ("d_p1", "SCI + V7f $-$ budget-matched shuffled schedule, seed 1"),
             ("d_p2", "SCI + V7f $-$ shuffled schedule, seed 2"), ("d_p3", "SCI + V7f $-$ shuffled schedule, seed 3"),
             ("high", "V7f at 832 px $-$ V7f at 736 px"), ("low", "V7f at 640 px $-$ V7f at 736 px")]
    rows = []
    for k, lab in comps:
        rows.append(f"{lab} & {v(f'fw.{k}.int.dHOTA', True)} & {v(f'fw.{k}.int.dMOTA')} & {v(f'fw.{k}.off.dHOTA', True)} & "
                    f"{v(f'fw.{k}.off.dMOTA', True)}")
    notes = (f"Relative compute of the scene-adaptive arms: SCI only {v('fw.comp.NATIVE+SCI.yolov8')} (YOLOv8n) / "
             f"{v('fw.comp.NATIVE+SCI.rtdetr')} (RT-DETR-L); SCI + V7f {v('fw.comp.V7f+SCI.yolov8')} / "
             f"{v('fw.comp.V7f+SCI.rtdetr')}; 832 px {v('fw.comp.V7f+HIGH.yolov8')}; 640 px {v('fw.comp.V7f+LOW.yolov8')} "
             "(736 px = 1). The decision rule was committed before the first result.")
    cap = ("Pre-registered factorial on VisDrone2019-MOT val (ByteTrack, YOLOv8n and RT-DETR-L, pooled over 14 "
           "detector--sequence cells). The historical scene controller is compared with and without V7f, and with "
           "budget-matched scene-blind schedules. Paired bootstrap, 10,000 resamples, seed 42.")
    write("tab_factorial.tex", table("tab:factorial", cap, "p{5.6cm}llll",
                                     "Comparison & $\\Delta$HOTA internal & $\\Delta$MOTA internal & $\\Delta$HOTA official & "
                                     "$\\Delta$MOTA official", rows, star=True, notes=notes))


# ============================================================================ Published methods
def tab_published():
    global _current
    _current = "tab_published"
    rows = [
        f"SparseTrack (TCSVT 2025) & {v('ref.st.HOTA')} / {v('ref.st.MOTA')} / {v('ref.st.IDF1')} (paper, val) & "
        f"{v('mot.st.base.HOTA')} & {v('mot.st.v7f.HOTA')} & {v('mot.st.dHOTA', True)} & D",
        f"BoostTrack (MVA 2024), online & {v('ref.bt.HOTA')} / {v('ref.bt.MOTA')} / {v('ref.bt.IDF1')} (authors' corrected) & "
        f"{v('mot.bt.base.HOTA')} & {v('mot.bt.v7f.HOTA')} & 0 (identical) & D",
        f"ByteTrack (ECCV 2022) & -- / {v('ref.by.MOTA')} / {v('ref.by.IDF1')} (own detector run) & {v('mot.byoff.base.HOTA')} & "
        f"{v('mot.byoff.v7f.HOTA')} & {v('mot.byoff.dHOTA', True)} & D",
        f"OC-SORT (CVPR 2023) & {v('ref.oc.HOTA')} / {v('ref.oc.MOTA')} / {v('ref.oc.IDF1')} (paper, val-half) & "
        f"{v('mot.oc.base.HOTA')} & {v('mot.oc.v7f.HOTA')} & {v('mot.oc.dHOTA', True)} & D",
        f"PD-SORT (TCE 2025) & {v('ext.pd.base.HOTA')} (authors' released output) & {v('ext.pd.base.HOTA')} & "
        f"{v('ext.pd.v7f.HOTA')} & {v('ext.pd.dHOTA', True)} & P",
        f"Hybrid-SORT (AAAI 2024) & {v('ref.hy.HOTA')} / {v('ref.hy.MOTA')} / {v('ref.hy.IDF1')} (README) & "
        f"{v('ext.hy.base.HOTA')} & {v('ext.hy.v7f.HOTA')} & 0 (identical) & P",
        f"C-TWiX (PR 2025), MOT17 & {v('ctx.mot.refHOTA')} (README) & {v('ctx.mot.base.HOTA')} & {v('ctx.mot.v7f.HOTA')} & "
        f"{v('ctx.mot.dHOTA', True)} & X",
        f"C-TWiX, KITTIMOTS car & {v('ctx.kcar.refHOTA')} (README) & {v('ctx.kcar.base.HOTA')} & {v('ctx.kcar.v7f.HOTA')} & "
        f"{v('ctx.kcar.dHOTA', True)} & X",
        f"C-TWiX, DanceTrack & {v('ctx.dance.refHOTA')} (README) & {v('ctx.dance.base.HOTA')} & {v('ctx.dance.v7f.HOTA')} & "
        f"{v('ctx.dance.dHOTA', True)} & X",
        f"TrackTrack (CVPR 2025), DanceTrack & {v('tt.dance.refHOTA')} (paper, post-processed) & {v('tt.dance.base.HOTA')} & "
        f"{v('tt.dance.v7f.HOTA')} & {v('tt.dance.dHOTA', True)} & X",
        f"TOPICTrack (TIP 2025), MOT17 & {v('tp.mot.refHOTA')} (paper) & {v('tp.mot.base.HOTA')} & -- & reproduction failed; excluded & X",
    ]
    notes = ("All values are validation-split evaluations with TrackEval in our environment, on the same split the "
             "published number refers to; none is a test-server number, and none is comparable with leaderboard results. "
             "VisDrone and UAVDT results in this paper use a custom or official-compatible protocol that differs from the "
             "published VisDrone/UAVDT benchmarks; they are not directly comparable with published tracker numbers on those "
             "benchmarks and are not compared here.")
    cap = ("Published trackers: number reported by the authors, our reproduction (host alone), and the reproduction with "
           "the frozen V7f layer (HOTA; reported number as HOTA / MOTA / IDF1 where available). The comparison measures "
           "what the layer adds to each published system, not a ranking between systems.")
    write("tab_published.tex", table("tab:published", cap, "p{3.8cm}p{4.2cm}rrlc",
                                     "System & Reported & Reproduction & + V7f & $\\Delta$HOTA [95\\% CI] & St.", rows,
                                     star=True, notes=notes))


# ============================================================================ outputs
def numbers_tex():
    lines = ["% Generated by scripts/make_tables.py from the evidence registry; do not edit by hand.",
             "\\makeatletter",
             "\\newcommand{\\V}[1]{\\@ifundefined{acv@#1}{\\PackageError{numbers}{Undefined value #1}{}}{\\@nameuse{acv@#1}}}",
             "\\newcommand{\\CI}[1]{\\@ifundefined{acc@#1}{\\PackageError{numbers}{Undefined interval #1}{}}{\\@nameuse{acc@#1}}}"]
    for k, x in sorted(R.items()):
        t = x.text
        if isinstance(x.value, str) and not t.startswith("$"):
            t = t.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&") \
                 .replace("#", "\\#")
        lines.append(f"\\@namedef{{acv@{k}}}{{{t}}}")
        if x.citext:
            lines.append(f"\\@namedef{{acc@{k}}}{{{x.citext}}}")
    for det, f in FIG.get("fig3", {}).items():
        lines.append(f"\\@namedef{{acv@fig3.{det}.perframe}}{{{f['per_frame']:.0f}}}")
        lines.append(f"\\@namedef{{acv@fig3.{det}.ge025}}{{{100 * f['share_ge_025']:.1f}}}")
        lines.append(f"\\@namedef{{acv@fig3.{det}.t1}}{{{f['t1_median_score']:.2f}}}")
        lines.append(f"\\@namedef{{acv@fig3.{det}.t2}}{{{f['t2_median_score']:.2f}}}")
    f4 = FIG.get("fig4")
    if f4:
        for k in ("pooled", "n_reject", "n_ext", "n_primary"):
            lines.append(f"\\@namedef{{acv@fig4.{k}}}{{{f4[k]:,}}}")
        lines.append(f"\\@namedef{{acv@fig4.t1score}}{{{f4['t1_score']:.2f}}}")
        lines.append(f"\\@namedef{{acv@fig4.t2score}}{{{f4['t2_score']:.2f}}}")
    f8 = FIG["fig8"]
    lines.append(f"\\@namedef{{acv@fig8.noisy}}{{{f8['noisy']}}}")
    lines.append(f"\\@namedef{{acv@fig8.frames}}{{{f8['frames']}}}")
    lines.append(f"\\@namedef{{acv@fig8.switches}}{{{f8['regime_switches']}}}")
    lines.append(f"\\@namedef{{acv@fig8.rhomin}}{{{f8['rho_bar_min']:.3f}}}")
    lines.append(f"\\@namedef{{acv@fig8.rhomax}}{{{f8['rho_bar_max_after_first_50']:.3f}}}")
    lines.append("\\makeatother")
    write("numbers.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    for fn in (tab_stage1, tab_attribution, tab_rawscale, tab_calib, tab_contracts, tab_constants, tab_protocol,
               tab_central, tab_visdrone_official, tab_retinanet, tab_runtime, tab_ablation, tab_factorial, tab_published):
        fn()
    numbers_tex()
    (OUT / "text_numbers.json").write_text(json.dumps(
        {k: dict(value=x.value, text=x.text.replace(E.MINUS, "-"), ci=list(x.ci) if x.ci else None, source=x.src,
                 kind=x.kind) for k, x in sorted(R.items())}, indent=1, default=str))
    (OUT / "table_keys.json").write_text(json.dumps({k: sorted(s) for k, s in USED.items()}, indent=1))
    print(f"wrote {len(USED)} tables, {len(R)} registered values")
