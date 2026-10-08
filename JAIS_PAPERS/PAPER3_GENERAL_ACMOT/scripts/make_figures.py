"""
Figures 3-8 of the General AC-MOT manuscript (Figures 1-2 are TikZ in manuscript.tex).

Result figures read the evidence registry (scripts/evidence.py, pinned commit).
Figures 3 and 4 are descriptive statistics of frozen detector caches published as
release assets; set ACMOT_CACHE_DIR to the directory holding the extracted assets:

  v7-dev-assets-1/acmot_detcache_val_native.tar   -> outputs/det_cache_val_native/<det>/<seq>.npz
  sci-v7f-unseen-1/sci_retinanet_val7.tar         -> sweep/retinanet/736/<seq>.npz

Both archives are checked against their published SHA-256 before use. The
thresholds in Figures 3-4 are computed with the frozen online_calibration.nested_otsu
(imported from the pinned repository file, not re-implemented).

    ACMOT_CACHE_DIR=/path/to/cache python scripts/make_figures.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence as E  # noqa: E402

R = E.build()
OUT = E.PAPER / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.compression": 9, "svg.hashsalt": "acmot", "path.simplify": True,
})
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NEG, POS, MID = "#e34948", "#2a78d6", "#f0efec"

ASSETS = {
    "val_native": ("acmot_detcache_val_native.tar",
                   "69a76315459c14fb358b564816e02371dc7ee21649cfe6791464936a2b69516e", "v7-dev-assets-1"),
    "retinanet": ("sci_retinanet_val7.tar",
                  "4fbbc65efa9b6eae087e44490620da92a9c98f86eee39e0aacc4182f79953c06", "sci-v7f-unseen-1"),
}
DETS = [("yolov8", "YOLOv8n"), ("rtdetr", "RT-DETR-L"), ("fasterrcnn", "Faster R-CNN"), ("retinanet", "RetinaNet")]


def save(fig, name):
    fig.savefig(OUT / name, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    plt.close(fig)


def frozen_otsu():
    """Load online_calibration.py exactly as committed at the pinned commit."""
    src = E.show("online_calibration.py")
    tmp = Path(tempfile.mkdtemp()) / "online_calibration_pinned.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("online_calibration_pinned", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.nested_otsu


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def logit(s):
    s = np.clip(s, 1e-9, 1 - 1e-9)
    return np.log(s / (1 - s))


def load_streams(cache):
    """{det: {seq: (frames, scores per frame list)}} at 736 px, all cached candidates."""
    for key, (name, digest, _) in ASSETS.items():
        tar = cache / name
        if not tar.exists() or sha256(tar) != digest:
            raise SystemExit(f"missing or modified release asset: {tar}")
    out = {}
    for det, _ in DETS:
        if det == "retinanet":
            files = sorted((cache / "rn/sweep/retinanet/736").glob("*.npz"))
        else:
            files = sorted((cache / f"outputs/det_cache_val_native/{det}").glob("*.npz"))
        seqs = {}
        for f in files:
            z = np.load(f)
            a = z["det"] if "det" in z.files else z["det_736"]
            n = int(z["frames"])
            fr = a[:, 0].astype(int)
            seqs[f.stem] = (n, [a[fr == t, 5] for t in range(int(fr.min()), int(fr.min()) + n)])
        out[det] = seqs
    return out


# ----------------------------------------------------------------------------- Fig. 3 / 4 (caches)
def fig_scores(streams, nested_otsu):
    stats = {}
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.85), sharey=False)
    bins = np.linspace(-5, 5, 61)
    for ax, (det, label) in zip(axes, DETS):
        allsc, t1s, t2s, nframes = [], [], [], 0
        for seq, (n, per) in streams[det].items():
            nframes += n
            L = [logit(s) for s in per]
            for t in range(len(L)):
                allsc.append(per[t])
                if t >= 1:
                    H = np.concatenate(L[max(0, t - 10):t])
                    if H.size >= 3:
                        r = nested_otsu(H)
                        if r is not None:
                            t1s.append(r[0])
                            t2s.append(r[1])
        sc = np.concatenate(allsc)
        lg = logit(sc)
        t1, t2 = float(np.median(t1s)), float(np.median(t2s))
        stats[det] = dict(candidates=int(sc.size), frames=nframes, per_frame=sc.size / nframes,
                          share_ge_025=float((sc >= 0.25).mean()), median_score=float(np.median(sc)),
                          t1_median_score=float(1 / (1 + np.exp(-t1))), t2_median_score=float(1 / (1 + np.exp(-t2))),
                          windows=len(t1s))
        w = np.full(lg.shape, 1.0 / lg.size)
        ax.hist(np.clip(lg, bins[0], bins[-1]), bins=bins, weights=w, color=BLUE, edgecolor="#fcfcfb", linewidth=0.4)
        ax.axvline(logit(np.array(0.25)), color=INK2, linestyle=(0, (3, 2)), linewidth=1.0)
        ax.axvline(t1, color=ORANGE, linewidth=1.2)
        ax.axvline(t2, color=ORANGE, linewidth=1.2, linestyle=(0, (1, 1)))
        ax.set_title(f"{label}\n{sc.size / nframes:.0f} cand./frame, {100 * (sc >= 0.25).mean():.0f}% ≥ 0.25",
                     fontsize=6.8, color=INK, linespacing=1.3)
        ax.set_xlabel("score logit", fontsize=7)
        ax.set_xlim(bins[0], bins[-1])
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel("share of candidates", fontsize=7)
    handles = [Line2D([], [], color=INK2, linestyle=(0, (3, 2)), label="shared raw threshold 0.25"),
               Line2D([], [], color=ORANGE, label="median $t_1$"),
               Line2D([], [], color=ORANGE, linestyle=(0, (1, 1)), label="median $t_2$")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig3_score_distributions.pdf")
    return stats


def fig_otsu_window(streams, nested_otsu):
    det, seq = "rtdetr", sorted(streams["rtdetr"])[0]
    n, per = streams[det][seq]
    t_end = n // 2                       # decision frame (0-based index); window = 10 previous frames
    L = np.concatenate([logit(s) for s in per[t_end - 10:t_end]])
    t1, t2, eta = nested_otsu(L)
    fig, ax = plt.subplots(figsize=(6.0, 1.9))
    bins = np.linspace(np.floor(L.min()), np.ceil(L.max()), 49)
    parts = [(L < t1, MUTED, "reject (below $t_1$)"), ((L >= t1) & (L < t2), ORANGE, "extension ($t_1 \\leq z < t_2$)"),
             (L >= t2, BLUE, "primary ($z \\geq t_2$)")]
    for m, col, lab in parts:
        ax.hist(L[m], bins=bins, color=col, edgecolor="#fcfcfb", linewidth=0.4, label=f"{lab}: {int(m.sum())}")
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for t, name in [(t1, "$t_1$"), (t2, "$t_2$")]:
        ax.axvline(t, color=INK, linewidth=1.0)
        ax.text(t, 1.02, name, va="bottom", ha="center", fontsize=8, color=INK, transform=tr)
    h025 = float(logit(np.array(0.25)))
    ax.axvline(h025, color=INK2, linestyle=(0, (3, 2)), linewidth=1.0)
    ax.text(h025 + 0.06, 0.93, "host 0.25", va="center", ha="left", fontsize=6.8, color=INK2, transform=tr)
    ax.set_xlabel("candidate score logit $z$ (pooled over the 10 frames before the decision frame)", fontsize=7)
    ax.set_ylabel("candidates", fontsize=7)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.tick_params(labelsize=6.5)
    save(fig, "fig4_nested_otsu.pdf")
    return dict(detector=det, sequence=seq, decision_frame_index=int(t_end), pooled=int(L.size),
                t1=float(t1), t2=float(t2), eta=float(eta),
                n_reject=int((L < t1).sum()), n_ext=int(((L >= t1) & (L < t2)).sum()), n_primary=int((L >= t2).sum()),
                t1_score=float(1 / (1 + np.exp(-t1))), t2_score=float(1 / (1 + np.exp(-t2))))


# ----------------------------------------------------------------------------- result rows shared with the tables
def central_rows():
    """(group, label, key-prefix, status) of every native-vs-V7f cell with a HOTA interval."""
    return [
        ("VisDrone val-7 (internal)", "YOLOv8n · ByteTrack", "vd.yolov8.int.dHOTA", "D"),
        ("VisDrone val-7 (internal)", "RT-DETR-L · ByteTrack", "vd.rtdetr.int.dHOTA", "D"),
        ("VisDrone val-7 (internal)", "YOLOv8n · BoT-SORT", "bot.yolov8.int.dHOTA", "D"),
        ("VisDrone val-7 (internal)", "RT-DETR-L · BoT-SORT", "bot.rtdetr.int.dHOTA", "D"),
        ("VisDrone val-7 (internal)", "YOLOv8n · OC-SORT", "oc.yolov8.int.dHOTA", "D"),
        ("VisDrone val-7 (internal)", "RT-DETR-L · OC-SORT", "oc.rtdetr.int.dHOTA", "D"),
        ("VisDrone val-7 (internal)", "RetinaNet · ByteTrack", "rn.by.736.int.dHOTA", "U"),
        ("VisDrone val-7 (internal)", "RetinaNet · BoT-SORT", "bot.retinanet.int.dHOTA", "U"),
        ("KITTI tracking training", "YOLOv8n · ByteTrack", "kitd.yolov8.by.dHOTA", "D"),
        ("KITTI tracking training", "YOLOv8n · BoT-SORT", "kitd.yolov8.bot.dHOTA", "D"),
        ("KITTI tracking training", "YOLOv8n · OC-SORT", "kitd.yolov8.oc.dHOTA", "D"),
        ("KITTI tracking training", "RT-DETR-L · ByteTrack", "kitd.rtdetr.by.dHOTA", "D"),
        ("KITTI tracking training", "RT-DETR-L · BoT-SORT", "kitd.rtdetr.bot.dHOTA", "D"),
        ("KITTI tracking training", "RT-DETR-L · OC-SORT", "kitd.rtdetr.oc.dHOTA", "D"),
        ("MOT17 val-half", "YOLOX-X · SparseTrack", "mot.st.dHOTA", "D"),
        ("MOT17 val-half", "YOLOX-X · ByteTrack (official)", "mot.byoff.dHOTA", "D"),
        ("MOT17 val-half", "YOLOX-X · OC-SORT", "mot.oc.dHOTA", "D"),
        ("MOT17 val-half", "YOLOX-X · PD-SORT", "ext.pd.dHOTA", "P"),
        ("MOT17 val-half", "YOLOX-X · Hybrid-SORT", "ext.hy.dHOTA", "P"),
        ("MOT17 val-half", "YOLOX-X · C-TWiX", "ctx.mot.dHOTA", "X"),
        ("KITTIMOTS val", "PermaTrack · C-TWiX (car)", "ctx.kcar.dHOTA", "X"),
        ("KITTIMOTS val", "PermaTrack · C-TWiX (pedestrian)", "ctx.kped.dHOTA", "X"),
        ("DanceTrack val", "released · C-TWiX", "ctx.dance.dHOTA", "X"),
        ("DanceTrack val", "released · TrackTrack", "tt.dance.dHOTA", "X"),
    ]


STATUS = {"D": "development", "U": "unseen detector, development sequences", "P": "predeclared external",
          "X": "post-freeze external"}


def fig_forest():
    rows = central_rows()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    y = 0
    yt, yl = [], []
    group = None
    for g, lab, k, st in rows:
        if g != group:
            y -= 0.6 if group else 0
            ax.text(-17.6, y, g, fontsize=7, color=INK, fontweight="bold", va="center")
            group = g
            y -= 1
        v = R[k]
        lo, hi = v.ci
        sig = lo > 0 or hi < 0
        col = {"D": BLUE, "U": ORANGE, "P": AQUA, "X": AQUA}[st]
        marker = {"D": "o", "U": "s", "P": "D", "X": "^"}[st]
        ax.plot([lo, hi], [y, y], color=col, linewidth=1.4, solid_capstyle="round")
        ax.plot([v.value], [y], marker=marker, markersize=5, color=col,
                markerfacecolor=col if sig else "#fcfcfb", markeredgewidth=1.2, linestyle="none")
        yt.append(y)
        yl.append(lab)
        y -= 1
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=6.8)
    ax.set_xlabel("ΔHOTA, host + V7f − host alone (points, 95% paired bootstrap interval)", fontsize=7)
    ax.set_xlim(-6, 20)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    h = [Line2D([], [], color=BLUE, marker="o", label=STATUS["D"]),
         Line2D([], [], color=ORANGE, marker="s", label=STATUS["U"]),
         Line2D([], [], color=AQUA, marker="D", label=STATUS["P"]),
         Line2D([], [], color=AQUA, marker="^", label=STATUS["X"]),
         Line2D([], [], color=INK2, marker="o", markerfacecolor="#fcfcfb", linestyle="none",
                label="hollow: interval contains 0")]
    ax.legend(handles=h, fontsize=6.5, frameon=False, loc="lower right")
    save(fig, "fig5_forest.pdf")


def fig_matrix():
    streams = ["YOLOv8n\nVisDrone", "RT-DETR-L\nVisDrone", "RetinaNet\nVisDrone", "YOLOv8n\nKITTI", "RT-DETR-L\nKITTI",
               "YOLOX-X\nMOT17", "PermaTrack\nKITTIMOTS car", "released\nDanceTrack"]
    trackers = ["ByteTrack", "BoT-SORT", "OC-SORT", "SparseTrack", "BoostTrack", "PD-SORT", "Hybrid-SORT", "C-TWiX",
                "TrackTrack"]
    cell = {
        (0, 0): ("vd.yolov8.int.dHOTA", "D"), (1, 0): ("vd.rtdetr.int.dHOTA", "D"), (2, 0): ("rn.by.736.int.dHOTA", "U"),
        (0, 1): ("bot.yolov8.int.dHOTA", "D"), (1, 1): ("bot.rtdetr.int.dHOTA", "D"), (2, 1): ("bot.retinanet.int.dHOTA", "U"),
        (0, 2): ("oc.yolov8.int.dHOTA", "D"), (1, 2): ("oc.rtdetr.int.dHOTA", "D"),
        (3, 0): ("kitd.yolov8.by.dHOTA", "D"), (3, 1): ("kitd.yolov8.bot.dHOTA", "D"), (3, 2): ("kitd.yolov8.oc.dHOTA", "D"),
        (4, 0): ("kitd.rtdetr.by.dHOTA", "D"), (4, 1): ("kitd.rtdetr.bot.dHOTA", "D"), (4, 2): ("kitd.rtdetr.oc.dHOTA", "D"),
        (5, 0): ("mot.byoff.dHOTA", "D"), (5, 2): ("mot.oc.dHOTA", "D"), (5, 3): ("mot.st.dHOTA", "D"),
        (5, 4): ("mot.bt.identical", "D"), (5, 5): ("ext.pd.dHOTA", "P"), (5, 6): ("ext.hy.dHOTA", "P"),
        (5, 7): ("ctx.mot.dHOTA", "X"), (6, 7): ("ctx.kcar.dHOTA", "X"), (7, 7): ("ctx.dance.dHOTA", "X"),
        (7, 8): ("tt.dance.dHOTA", "X"),
    }
    lim = 5.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [NEG, MID, POS])
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for i in range(len(streams)):
        for j in range(len(trackers)):
            if (i, j) not in cell:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#fcfcfb", edgecolor=GRID, linewidth=0.6))
                ax.text(j, i, "–", ha="center", va="center", fontsize=7, color=MUTED)
                continue
            k, st = cell[(i, j)]
            same = k.endswith("identical") or (R[k].value == 0 and tuple(R[k].ci) == (0, 0))
            v = 0.0 if same else R[k].value
            sig = False if same else (R[k].ci[0] > 0 or R[k].ci[1] < 0)
            col = cmap((np.clip(v, -lim, lim) + lim) / (2 * lim))
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=col, edgecolor="#fcfcfb", linewidth=1.5))
            txt = "0 (id.)" if same else (f"{v:+.2f}" if abs(v) < 10 else f"{v:+.1f}")
            ink = "#ffffff" if abs(v) > 3.2 else INK
            ax.text(j, i - 0.08, txt + ("*" if sig else ""), ha="center", va="center", fontsize=6.6, color=ink)
            ax.text(j, i + 0.30, st, ha="center", va="center", fontsize=5.6, color=ink)
    ax.set_xlim(-0.5, len(trackers) - 0.5)
    ax.set_ylim(len(streams) - 0.5, -0.5)
    ax.set_xticks(range(len(trackers)))
    ax.set_xticklabels(trackers, fontsize=6.6, rotation=30, ha="right")
    ax.set_yticks(range(len(streams)))
    ax.set_yticklabels(streams, fontsize=6.4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(-lim, lim))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("ΔHOTA (clipped at ±5)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    save(fig, "fig6_transfer_matrix.pdf")


def fig_runtime():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.0), gridspec_kw=dict(width_ratios=[1.5, 1]))
    rows = [("Xeon 4 vCPU, KITTI\nhost alone", R["rt.base.det"].value, R["rt.base.ctrl"].value, R["rt.base.trk"].value),
            ("Xeon 4 vCPU, KITTI\n+ V7f", R["rt.v7f.det"].value, R["rt.v7f.ctrl"].value, R["rt.v7f.trk"].value),
            ("EPYC 2 thr., VisDrone\nhost alone", R["g1rt.yolov8.base.det"].value, R["g1rt.yolov8.base.score"].value,
             R["g1rt.yolov8.base.trk"].value),
            ("EPYC 2 thr., VisDrone\nGeneral AC-MOT", R["g1rt.yolov8.g1.det"].value, R["g1rt.yolov8.g1.score"].value,
             R["g1rt.yolov8.g1.trk"].value)]
    for i, (lab, d, c, t) in enumerate(rows):
        y = len(rows) - 1 - i
        a1.barh(y, d, color=BLUE, edgecolor="#fcfcfb", linewidth=1.0, height=0.62)
        a1.barh(y, c, left=d, color=ORANGE, edgecolor="#fcfcfb", linewidth=1.0, height=0.62)
        a1.barh(y, t, left=d + c, color=AQUA, edgecolor="#fcfcfb", linewidth=1.0, height=0.62)
        a1.text(d + c + t + 1.5, y, f"{d + c + t:.1f} ms", va="center", fontsize=6.5, color=INK2)
    a1.set_yticks(range(len(rows)))
    a1.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.3)
    a1.set_xlabel("mean per-frame time, YOLOv8n (ms; decode excluded)", fontsize=6.8)
    a1.set_xlim(0, 112)
    a1.tick_params(labelsize=6.3)
    a1.legend(["detector", "control layer", "tracker"], fontsize=6.3, frameon=False, loc="lower center", ncol=3,
              bbox_to_anchor=(0.45, 1.0))
    pct = [("YOLOv8n\n(Xeon)", R["rt.pipe_pct"].value), ("YOLOv8n\n(EPYC)", R["g1rt.yolov8.pct"].value),
           ("RT-DETR-L\n(EPYC)", R["g1rt.rtdetr.pct"].value), ("RetinaNet\n(EPYC)", R["g1rt.retinanet.pct"].value)]
    a2.bar(range(len(pct)), [p for _, p in pct], color=BLUE, width=0.6, edgecolor="#fcfcfb")
    for i, (_, p) in enumerate(pct):
        a2.text(i, p + 0.15, f"+{p:.1f}%", ha="center", fontsize=6.5, color=INK2)
    a2.set_xticks(range(len(pct)))
    a2.set_xticklabels([p[0] for p in pct], fontsize=6.2)
    a2.set_ylabel("total-time overhead (%)", fontsize=6.8)
    a2.tick_params(labelsize=6.3)
    a2.set_ylim(0, 8.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig7_runtime.pdf")


def fig_boundary():
    audit = json.loads(E.show(E.FIN + "recent/ctwix/audit_KITTIMOT.json"))
    seq = [r for r in audit if r["seq"] == "0014"]
    fr = np.array([r["frame"] for r in seq])
    rb = np.array([r.get("rho_bar", np.nan) for r in seq], dtype=float)
    noisy = np.array([r["regime"] == "noisy" for r in seq])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.1), gridspec_kw=dict(width_ratios=[1.2, 1]))
    for f in fr[noisy]:
        a1.axvspan(f - 0.5, f + 0.5, color=NEG, alpha=0.18, linewidth=0)
    a1.plot(fr, rb, color=BLUE, linewidth=1.4)
    a1.axhline(0.5, color=INK2, linestyle=(0, (3, 2)), linewidth=0.9)
    a1.text(fr.max(), 0.5, " ½", va="center", fontsize=7, color=INK2)
    a1.set_xlabel("frame (KITTIMOTS 0014, C-TWiX host)", fontsize=6.8)
    a1.set_ylabel(r"cumulative median $\bar\rho_t$", fontsize=6.8)
    a1.tick_params(labelsize=6.3)
    a1.text(0.40, 0.80, f"noisy frames: {int(noisy.sum())} (shaded)", transform=a1.transAxes, fontsize=6.5, color=INK2)
    px = [512, 576, 640, 704, 736, 768, 832, 896, 960]
    comp = [float(x) for x in R["sw.compute"].value.split(",")]
    for k, lab, col, ls, dy in [("yolov8", "YOLOv8n + V7f", BLUE, "-", -0.7), ("yolov8n", "YOLOv8n alone", BLUE, (0, (3, 2)), -0.3),
                                ("rtdetr", "RT-DETR-L + V7f", ORANGE, "-", 0.0), ("rtdetrn", "RT-DETR-L alone", ORANGE, (0, (3, 2)), 0.7)]:
        ys = [float(x) for x in R[f"sw.{k}"].value.split(",")]
        a2.plot(comp, ys, color=col, linestyle=ls, linewidth=1.6 if ls == "-" else 1.2, marker="o", markersize=3)
        a2.text(comp[-1] + 0.04, ys[-1] + dy, lab, fontsize=6, color=INK2, va="center")
    a2.axvline(1.0, color=MUTED, linewidth=0.8)
    a2.text(1.0, 26.2, " 736 px", fontsize=6, color=MUTED)
    a2.set_xlabel("relative compute $(r/736)^2$", fontsize=6.8)
    a2.set_ylabel("HOTA (internal)", fontsize=6.8)
    a2.set_xlim(0.4, 2.45)
    a2.tick_params(labelsize=6.3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig8_boundary.pdf")
    sw = [r for r in seq if r.get("rho_bar") is not None]
    switches = int(sum(1 for a, b in zip(seq, seq[1:]) if a["regime"] != b["regime"] and a["regime"] != "cold"))
    return dict(seq="0014", frames=len(seq), noisy=int(noisy.sum()), rho_bar_min=float(np.nanmin(rb)),
                rho_bar_max_after_first_50=float(np.nanmax(rb[50:])) if len(rb) > 50 else None,
                rho_bar_last=float(rb[-1]), regime_switches=switches, n_with_bands=len(sw))


if __name__ == "__main__":
    data = {}
    cache = os.environ.get("ACMOT_CACHE_DIR")
    if cache:
        nested = frozen_otsu()
        st = load_streams(Path(cache))
        data["fig3"] = fig_scores(st, nested)
        data["fig4"] = fig_otsu_window(st, nested)
        data["assets"] = {k: dict(file=v[0], sha256=v[1], release=v[2]) for k, v in ASSETS.items()}
    else:
        prev = OUT / "fig_data.json"
        if prev.exists():
            data = json.loads(prev.read_text())
        print("ACMOT_CACHE_DIR not set: Figures 3-4 kept from the previous run")
    fig_forest()
    fig_matrix()
    fig_runtime()
    data["fig8"] = fig_boundary()
    data["pin"] = E.PIN
    (OUT / "fig_data.json").write_text(json.dumps(data, indent=1))
    print("figures written:", sorted(p.name for p in OUT.glob("*.pdf")))
