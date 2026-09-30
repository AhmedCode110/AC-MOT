"""
Vector figures of Paper 2 (frozen V7f). File names carry the figure number of the
manuscript (Figs. 2 and 4 are TikZ drawings inside manuscript.tex); the functions keep
their development names (fig4 -> fig03_partition, fig7 -> fig08_calibration, etc.). Every plotted number comes from
scripts/data.py (committed result files); Fig. 4 additionally replays the
frozen layer, without a tracker, on one development detection cache as an
illustration of the nested partition. Run from any directory:

  python JAIS_PAPERS/PAPER2_UNIVERSAL_ACMOT_JAIS/scripts/make_figures.py [--cache <det_cache_val_native>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                     "pdf.fonttype": 42})
C = dict(dev="#4c72b0", pre="#dd8452", post="#55a868", neg="#c44e52", grey="#8c8c8c", v6="#c44e52",
         v7="#4c72b0", native="#8c8c8c")
TF = {"none": lambda s: s, "temp2": lambda s: 1 / (1 + np.exp(-np.log(s / (1 - s)) / 2.0)),
      "temp05": lambda s: 1 / (1 + np.exp(-np.log(s / (1 - s)) / 0.5)),
      "scale05": lambda s: 0.5 * s, "pow3": lambda s: s ** 3}
TF_LAB = {"none": "identity", "temp2": "temperature 2", "temp05": "temperature 0.5",
          "scale05": r"$0.5\,s$", "pow3": r"$s^3$"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig. 1
def fig1():
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw=dict(width_ratios=[1, 1.35]))
    s = np.linspace(1e-3, 1 - 1e-3, 400)
    for k, ls in zip(TF, ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]):
        ax[0].plot(s, TF[k](s), ls=ls, lw=1.2, label=TF_LAB[k])
    ax[0].axhline(0.6, color="k", lw=0.7)
    ax[0].axhline(0.1, color="k", lw=0.7, ls=":")
    ax[0].text(0.52, 0.53, "first-association\nthreshold 0.6", fontsize=6.0, bbox=dict(fc="white", ec="none", pad=0.5))
    ax[0].text(0.62, 0.12, "low stage 0.1", fontsize=6.3)
    ax[0].set_xlabel("Original detector score $s$")
    ax[0].set_ylabel("Score seen by the tracker")
    ax[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=6.0, handlelength=2.2)
    ax[0].set_title("(a) Score recalibrations used in the stress test", fontsize=8)
    hosts = [("ByteTrack\n(official)", "mot", "BY_official_st_NATIVE", "BY_official_st_NATIVE_t_{}"),
             ("ByteTrack\n(ultralytics)", "mot", "BY_ultra_st_NATIVE", "BY_ultra_st_NATIVE_t_{}"),
             ("OC-SORT", "mot", "OC_st_NATIVE", "OC_st_NATIVE_t_{}"),
             ("BoostTrack\n(online)", "boost", "BT7C_NATIVE_pf", "BT7C_NATIVE_{}_pf")]
    conds = ["none", "temp2", "temp05", "pow3", "scale05"]
    w = 0.16
    for j, cnd in enumerate(conds):
        for i, (lab, g, k0, kt) in enumerate(hosts):
            key = k0 if cnd == "none" else kt.format(cnd)
            src = D.MOT if g == "mot" else D.BOOST
            x = i + (j - 2) * w
            if key in src:
                v = src[key]["HOTA"]
                ax[1].bar(x, v, w, color=plt.cm.viridis(j / 4.6), label=TF_LAB[cnd] if i == 0 else None)
                if v < 1:
                    ax[1].text(x, 1.5, "0", ha="center", fontsize=6)
            else:
                ax[1].text(x, 2, "n.r.", ha="center", fontsize=5.5, rotation=90)
    ax[1].set_xticks(range(len(hosts)), [h[0] for h in hosts])
    ax[1].set_ylabel("HOTA of the host alone")
    ax[1].set_ylim(0, 75)
    ax[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=6.3)
    ax[1].set_title("(b) Fixed operating points under recalibration", fontsize=8, pad=22)
    save(fig, "fig01_fragility")


# ------------------------------------------------------------------ Fig. 4
def fig4(cache):
    sys.path.insert(0, str(D.ROOT))
    from acmot_v7 import HostContract, V7Layer, logit, spec_from_dict
    from online_calibration import nested_otsu
    from tools.run_policy_validation import CachedDetector
    import json
    spec = spec_from_dict(json.load(open(D.ROOT / "configs/universal_acmot_policy_v7.json"))["spec"])
    host = HostContract(assoc=0.25, birth=0.25, low=0.1, match=0.8)
    seq = "uav0000268_05773_v"         # same scene, two detectors (VisDrone val, development data)
    streams = {}
    t = 60                                   # decision frame shown in panels (a, b), 1-based
    for det, lab in (("yolov8", "YOLOv8n"), ("rtdetr", "RT-DETR-L")):
        cd = CachedDetector(str(Path(cache) / det / f"{seq}.npz"))
        layer, pooled, logs = V7Layer(spec, host), None, []
        for i in range(1, min(cd.frames, 300) + 1):
            cd.frame = i
            raw = cd.detect(None, 0.0, None, 736)
            b = np.array([[d.x1, d.y1, d.x2, d.y2] for d in raw]).reshape(-1, 4)
            s = np.array([d.confidence for d in raw])
            if i == t:                       # exactly the pooled logits H_t the layer uses at frame t
                pooled = np.concatenate(layer.window)
            dec = layer.step(b, s, None, classes=np.array([d.class_id for d in raw]))
            layer.observe(np.zeros((0, 4)))            # layer-only replay (no tracker)
            logs.append(dec.log)
        streams[lab] = (pooled, logs)
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.1), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for a, (lab, (H, logs)) in zip(ax[:2], streams.items()):
        g = logs[t - 1]
        t1, t2, rho = g["t1"], g["t2"], g["rho"]
        assert np.allclose(nested_otsu(H)[:2], (t1, t2))
        a.hist(H, bins=60, color=C["grey"], alpha=0.8)
        for v, n in ((t1, "$t_1$"), (t2, "$t_2$")):
            a.axvline(v, color="k", lw=0.9)
            a.text(v, a.get_ylim()[1] * 0.93, " " + n, fontsize=7)
        a.set_title(f"({'ab'[list(streams).index(lab)]}) {lab}, frame {t}: " + r"$\rho_t$" + f" = {rho:.2f}",
                    fontsize=7.5)
        a.set_xlabel("Pooled candidate logit")
        a.set_ylabel("Count")
    for lab, (frames, logs), col in zip(streams, streams.values(), (C["dev"], C["neg"])):
        f = [i + 1 for i, g in enumerate(logs) if "rho" in g]
        ax[2].plot(f, [logs[i - 1]["rho"] for i in f], ".", ms=1.5, color=col, alpha=0.5)
        ax[2].plot(f, [logs[i - 1]["rho_bar"] for i in f], "-", lw=1.2, color=col,
                   label=f"{lab}: " + r"$\bar\rho_t$")
    ax[2].axhline(0.5, color="k", lw=0.7, ls="--")
    ax[2].text(5, 0.52, "clean above, noisy below", fontsize=6.3)
    ax[2].set_ylim(0, 1.02)
    ax[2].set_xlabel("Frame $t$")
    ax[2].set_ylabel(r"$\rho_t$ (dots), $\bar\rho_t$ (lines)")
    ax[2].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.30), fontsize=6.3)
    ax[2].set_title("(c) Regime statistic over the stream", fontsize=7.5)
    save(fig, "fig03_partition")
    return {lab: dict(clean=sum(g.get("regime") == "clean" for g in logs),
                      noisy=sum(g.get("regime") == "noisy" for g in logs), frames=len(logs))
            for lab, (_, logs) in streams.items()}


# ------------------------------------------------------------------ Fig. 5
def fig5():
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.1), gridspec_kw=dict(width_ratios=[1.2, 1]))
    hosts = list(D.V6_TRANSFER)
    sp = D.SPARSE if "bootstrap" in D.SPARSE else None
    b = D.MAIN["sparsetrack"]["bootstrap_V7f_minus_reproduction"]["HOTA"]
    v7 = {"SparseTrack": (b["diff"], b["ci_lo"], b["ci_hi"]), "BoostTrack online": (0, 0, 0),
          "BoostTrack + GBI": (0, 0, 0)}
    y = np.arange(len(hosts))
    for off, lab, col, get in ((0.15, "frozen V6-TF (prior design)", C["v6"], lambda h: D.V6_TRANSFER[h]["HOTA"]),
                               (-0.15, "frozen V7f", C["v7"], lambda h: v7[h])):
        for i, h in enumerate(hosts):
            d, lo, hi = get(h)
            ax[0].errorbar(d, i + off, xerr=[[d - lo], [hi - d]], fmt="o", ms=4, color=col, capsize=2,
                           label=lab if i == 0 else None)
            if h != "SparseTrack" and lab.endswith("V7f"):
                ax[0].text(d + 0.15, i + off, "identical output", fontsize=6, va="center")
    ax[0].axvline(0, color="k", lw=0.6)
    ax[0].set_yticks(y, hosts)
    ax[0].set_xlabel(r"$\Delta$HOTA vs. reproduced host [95% CI]")
    ax[0].legend(frameon=False, fontsize=6.3, loc="center left", bbox_to_anchor=(0.0, 0.33))
    ax[0].set_title("(a) Same hosts, MOT17 val-half", fontsize=8)
    ax[0].set_xlim(-8.5, 1.8)
    f = D.V6_FATES
    left = 0
    cols = ["#9ecae1", "#fdae6b", "#e6550d", "#a63603"]
    for (k, v), c in zip(f.items(), cols):
        ax[1].barh(0, v, left=left, color=c, label=f"{k} ({v:,})")
        left += v
    ax[1].set_yticks([])
    ax[1].set_xlabel("True detections of SparseTrack's stream")
    ax[1].legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax[1].set_title("(b) Where V6-TF lost true detections", fontsize=8)
    save(fig, "fig05_v6_to_v7f")


# ------------------------------------------------------------------ Fig. 6
def fig6():
    rows = []
    b = D.MAIN["sparsetrack"]["bootstrap_V7f_minus_reproduction"]
    rows.append(("MOT17 · SparseTrack", (b["HOTA"]["diff"], b["HOTA"]["ci_lo"], b["HOTA"]["ci_hi"]),
                 (b["MOTA"]["diff"], b["MOTA"]["ci_lo"], b["MOTA"]["ci_hi"])))
    for k, v in D.STAT_MOT17.items():
        lab = "MOT17 · " + k
        if v.get("identical"):
            rows.append((lab, None, None))
        else:
            rows.append((lab, v["HOTA"], v["MOTA"]))
    rows.append(("MOT17 · BoostTrack + GBI, floor 0.1", None, None))
    for (h, det), v in D.STAT_KITTI.items():
        rows.append((f"KITTI · {h} · {det}", v["HOTA"], v["MOTA"]))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    y = np.arange(len(rows))[::-1]
    for a, j, name in ((ax[0], 1, "HOTA"), (ax[1], 2, "MOTA")):
        for yi, r in zip(y, rows):
            if r[j] is None:
                a.plot(0, yi, "D", mfc="white", mec=C["grey"], ms=4)
                continue
            d, lo, hi = r[j]
            col = C["dev"] if lo > 0 else C["neg"] if hi < 0 else C["grey"]
            a.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o", ms=3.5, color=col, capsize=2)
        a.axvline(0, color="k", lw=0.6)
        a.set_xlabel(rf"$\Delta${name} (V7f $-$ host) [95% CI]")
    ax[0].set_yticks(y, [r[0] for r in rows], fontsize=6.5)
    ax[0].axhspan(-0.5, 5.5, color="0.94", zorder=-1)
    ax[1].axhspan(-0.5, 5.5, color="0.94", zorder=-1)
    ax[1].text(0.98, 0.02, "open diamond: identical output\nblue: CI > 0, red: CI < 0, grey: CI includes 0",
               transform=ax[1].transAxes, ha="right", fontsize=6)
    save(fig, "fig06_development")


# ------------------------------------------------------------------ Fig. 8
def fig8():
    rows = []
    pd = D.EXT["PD-SORT (IEEE TCE 2025)"]
    per = [v["with_v7f"]["HOTA"] - v["reproduction"]["HOTA"] for v in pd["per_seq"].values()]
    rows.append(("PD-SORT · MOT17", "pre", D.PD_BOOT["HOTA"], per))
    rows.append(("Hybrid-SORT · MOT17", "pre", None, [0.0] * 7))
    for run, cls, lab in (("MOT17", "pedestrian", "C-TWiX · MOT17"), ("KITTIMOT", "car", "C-TWiX · KITTI car"),
                          ("KITTIMOT", "pedestrian", "C-TWiX · KITTI ped."),
                          ("DanceTrack", "pedestrian", "C-TWiX · DanceTrack")):
        r = D.recent("ctwix", run, cls)
        h = r["delta"]["HOTA"]
        rows.append((lab, "post", (h["diff"], h["ci_lo"], h["ci_hi"]), list(r["per_seq_dHOTA"].values())))
    r = D.recent("tracktrack", "DanceTrack_post", "pedestrian")
    h = r["delta"]["HOTA"]
    rows.append(("TrackTrack · DanceTrack", "post", (h["diff"], h["ci_lo"], h["ci_hi"]),
                 list(r["per_seq_dHOTA"].values())))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True, gridspec_kw=dict(width_ratios=[1, 1.25]))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, cat, ci, per) in zip(y, rows):
        col = C["pre"] if cat == "pre" else C["post"]
        if ci is None:
            ax[0].plot(0, yi, "D", mfc="white", mec=col, ms=4)
        else:
            d, lo, hi = ci
            mk = "o" if lo <= 0 <= hi else ("^" if lo > 0 else "v")
            ax[0].errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt=mk, ms=4, color=col, capsize=2)
        rng = np.random.default_rng(0)
        ax[1].scatter(per, yi + rng.uniform(-0.18, 0.18, len(per)), s=6, color=col, alpha=0.7, lw=0)
    for a in ax:
        a.axvline(0, color="k", lw=0.6)
    ax[0].set_yticks(y, [r[0] for r in rows], fontsize=6.8)
    ax[0].set_xlabel(r"Pooled $\Delta$HOTA [95% CI]")
    ax[0].set_title("(a) Pooled effect", fontsize=8)
    ax[1].set_xlabel(r"Per-sequence $\Delta$HOTA")
    ax[1].set_title("(b) Sequences", fontsize=8)
    ax[0].text(0.02, 0.98, "orange: predeclared at freeze\ngreen: selected after freeze",
               transform=ax[0].transAxes, fontsize=6, va="top")
    save(fig, "fig07_external")


# ------------------------------------------------------------------ Fig. 7
def fig7():
    if D.CALIB is None:
        print("fig7 skipped: calibration bootstrap not available")
        return
    rows = D.CALIB["conditions"]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        a, b = r["native"]["HOTA"], r["v7f"]["HOTA"]
        v = r["verdict"]
        col = C["dev"] if v == "significant recovery" else C["neg"] if v == "significant degradation" else C["grey"]
        ax.plot([a, b], [yi, yi], "-", color=col, lw=1.2)
        ax.plot(a, yi, "o", mfc="white", mec=C["native"], ms=4)
        ax.plot(b, yi, "o", color=col, ms=4)
        h = r["bootstrap"]["HOTA"]
        txt = ("identical" if v == "identical output" else
               f"{h['diff']:+.2f} [{h['ci_lo']:+.2f}, {h['ci_hi']:+.2f}]")
        ax.text(76, yi, txt, fontsize=6.3, va="center")
    ax.set_yticks(y, [f"{r['host']} · {TF_LAB[r['transform']]}" for r in rows], fontsize=6.5)
    ax.set_xlim(-2, 90)
    ax.set_xticks(range(0, 80, 10))
    ax.set_xlabel("HOTA: host alone (open) and host + V7f (filled); right: $\\Delta$HOTA [95% CI]")
    save(fig, "fig08_calibration")


# ------------------------------------------------------------------ Fig. 9
def fig9():
    import json
    car = D.recent("ctwix", "KITTIMOT", "car")
    ped = D.recent("ctwix", "KITTIMOT", "pedestrian")
    seqs = list(car["per_seq_dHOTA"])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw=dict(width_ratios=[1, 1.4]))
    x = np.arange(len(seqs))
    ax[0].bar(x - 0.2, [car["per_seq_dHOTA"][s] for s in seqs], 0.4, color=C["neg"], label="car")
    ax[0].bar(x + 0.2, [ped["per_seq_dHOTA"][s] for s in seqs], 0.4, color=C["grey"], label="pedestrian")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xticks(x, seqs, rotation=60, fontsize=6.5)
    ax[0].set_ylabel(r"$\Delta$HOTA (V7f $-$ C-TWiX)")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    ax[0].set_title("(a) KITTI val sequences", fontsize=8)
    au = [r for r in json.load(open(D.FIN / "recent/ctwix/audit_KITTIMOT.json")) if r["seq"] == "0014"]
    f = np.array([r["frame"] for r in au])
    rho = np.array([r.get("rho") if r.get("rho") is not None else np.nan for r in au], float)
    rb = np.array([r.get("rho_bar") if r.get("rho_bar") is not None else np.nan for r in au], float)
    noisy = np.array([r["regime"] == "noisy" for r in au])
    for i in np.where(noisy)[0]:
        ax[1].axvspan(f[i] - 0.5, f[i] + 0.5, color=C["neg"], alpha=0.15, lw=0)
    ax[1].plot(f, rho, ".", ms=2.5, color=C["grey"], label=r"$\rho_t$")
    ax[1].plot(f, rb, "-", lw=1.2, color="k", label=r"$\bar\rho_t$ (cumulative median)")
    ax[1].axhline(0.5, color="k", lw=0.6, ls="--")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("(b) Frame of sequence 0014 (shaded: noisy-regime frames)")
    ax[1].set_ylabel("Regime statistic")
    a2 = ax[1].twinx()
    a2.spines["right"].set_visible(True)
    a2.plot(f, [r["n_in"] for r in au], "-", lw=0.7, color=C["dev"], label="candidates in")
    a2.plot(f, [r["n_pass"] for r in au], "-", lw=0.7, color=C["pre"], label="candidates passed")
    a2.set_ylabel("Candidates per frame", fontsize=7)
    a2.set_ylim(0, 12)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=6, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig10_ctwix_kitti")


# ------------------------------------------------------------------ Fig. 10
def fig10():
    arms = D.RT["arms"]
    st = [("read", "frame decode"), ("det", "detector"), ("ctrl", "V7f controller"), ("trk", "tracker")]
    cols = ["#c7c7c7", "#4c72b0", "#dd8452", "#55a868"]
    fig, ax = plt.subplots(figsize=(6.6, 1.3))
    for yi, arm, lab in ((1, "baseline", "host alone"), (0, "v7", "host + V7f")):
        left = 0
        for (k, n), c in zip(st, cols):
            v = arms[arm][k]["mean_ms"]
            if arm == "baseline" and k == "ctrl":
                continue
            ax.barh(yi, v, left=left, color=c, label=n if arm == "v7" else None, height=0.6)
            left += v
        ax.text(left + 0.5, yi, f"{arms[arm]['e2e']['mean_ms']:.2f} ms/frame (P95 {arms[arm]['e2e']['p95_ms']:.2f})",
                va="center", fontsize=6.5)
    ax.set_yticks([0, 1], ["host + V7f", "host alone"])
    ax.set_xlim(0, 85)
    ax.set_xlabel("Mean time per frame (ms), 4-vCPU Xeon, YOLOv8n 736 px + ByteTrack, KITTI 0001/0009/0019")
    ax.legend(frameon=False, ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, 1.45))
    save(fig, "fig09_runtime")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Users/ahmedgouda/Desktop/Universal-ACMOT/outputs/det_cache_val_native")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    todo = a.only or ["1", "4", "5", "6", "7", "8", "9", "10"]
    for k in todo:
        r = fig4(a.cache) if k == "4" else globals()[f"fig{k}"]()
        if r:
            print(r)
