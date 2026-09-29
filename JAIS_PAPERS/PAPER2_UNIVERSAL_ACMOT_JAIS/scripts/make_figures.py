"""
Figures for Paper 2 (frozen V7f), built from scripts/evidence.py only.
  python JAIS_PAPERS/PAPER2_UNIVERSAL_ACMOT_JAIS/scripts/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence as E  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
C = dict(host="#7f7f7f", v6="#d62728", v7="#1f77b4", pos="#1f77b4", neg="#d62728", zero="#7f7f7f")

# ---------------------------------------------------------------- Fig. 2: V6 -> V7 on strong hosts
fig, ax = plt.subplots(figsize=(4.6, 2.1))
x = np.arange(len(E.V6))
w = 0.26
ax.bar(x - w, [r[1] for r in E.V6], w, color=C["host"], label="host alone")
ax.bar(x, [r[2] for r in E.V6], w, color=C["v6"], label="host + prior design (V6)")
ax.bar(x + w, [r[4] for r in E.V6], w, color=C["v7"], label="host + frozen V7f")
for i, r in enumerate(E.V6):
    ax.text(i, r[2] - 1.6, f"{r[3][0]:+.2f}", ha="center", color="white", fontsize=6.5)
ax.set_xticks(x, [r[0] for r in E.V6])
ax.set_ylim(55, 74)
ax.set_ylabel("HOTA (%)")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=6.5)
fig.savefig(OUT / "fig2_v6_to_v7.pdf")
plt.close(fig)


# ---------------------------------------------------------------- Fig. 3: forest plot of dHOTA
def ci_row(label, d, group):
    return (label, d, group)


rows = []
for h, v, _, dh, _, _ in E.MOT17_DEV:
    rows.append(ci_row(f"{h} {v}".strip(), dh or (0, 0, 0), "Development: MOT17"))
for h, det, n, _, dh, _, _ in E.KITTI:
    rows.append(ci_row(f"{h}, {det}", dh, "Development: KITTI (other detectors)"))
for n, _, _, dh, _, _ in E.EXT_PRE:
    rows.append(ci_row(n, dh or (0, 0, 0), "Predeclared external: MOT17"))
lab = {"MOT17": "MOT17", "KITTIMOT": "KITTI", "DanceTrack": "DanceTrack", "DanceTrack_post": "DanceTrack"}
for key, s in E.RECENT["systems"].items():
    for ds, r in s["runs"].items():
        if r["reproduction"].startswith("FAILED") or ds.endswith("_post") and key != "tracktrack":
            continue
        if key == "tracktrack" and not ds.endswith("_post"):
            continue
        for c, v in r["classes"].items():
            d = v["delta"]["HOTA"]
            name = f"{s['name']}, {lab[ds]}" + (f" {c}" if len(r["classes"]) > 1 else "")
            rows.append(ci_row(name, (d["diff"], d["ci_lo"], d["ci_hi"]), "Post-freeze external"))
groups = []
for r in rows:
    if r[2] not in groups:
        groups.append(r[2])
def forest(figsize, name, fs_lab, fs_grp):
    fig, ax = plt.subplots(figsize=figsize)
    y = 0
    yt, yl = [], []
    for g in groups:
        ax.text(-10.8, y, g, fontweight="bold", fontsize=fs_grp, va="center")
        y -= 1
        for lab_, (d, lo, hi), gg in rows:
            if gg != g:
                continue
            col = C["pos"] if lo > 0 else C["neg"] if hi < 0 else C["zero"]
            ax.plot([lo, hi], [y, y], color=col, lw=1.4)
            ax.plot(d, y, "o", color=col, ms=3.5)
            yt.append(y); yl.append(lab_)
            y -= 1
        y -= 0.4
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(yt, yl, fontsize=fs_lab)
    ax.set_xlim(-11, 11)
    ax.set_xlabel("$\\Delta$HOTA, host + layer minus host (points, 95% interval)", fontsize=fs_lab + 0.5)
    ax.tick_params(axis="x", labelsize=fs_lab)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / name)
    plt.close(fig)


forest((6.4, 5.2), "fig3_forest.pdf", 6.5, 7)
forest((3.45, 5.3), "fig3_forest_col.pdf", 5.8, 6)

# ---------------------------------------------------------------- Fig. 4: per-sequence heatmap
names = list(E.PER_SEQ)
M = np.array([E.PER_SEQ[n] for n in names], float)
fig, ax = plt.subplots(figsize=(4.4, 2.3))
lim = 3.0
im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(len(E.SEQS)), [f"MOT17-{s}" for s in E.SEQS], rotation=30, fontsize=6.5)
ax.set_yticks(range(len(names)), names, fontsize=6.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:+.2f}" if M[i, j] else "0", ha="center", va="center", fontsize=5.8)
plt.colorbar(im, ax=ax, label="$\\Delta$HOTA (points)", pad=0.02)
fig.savefig(OUT / "fig4_per_sequence.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 5: calibration shift
fig, axs = plt.subplots(1, 4, figsize=(6.9, 2.0), sharey=True)
for a, host in zip(axs, E.STRESS):
    d = E.STRESS[host]
    ks = [k for k in E.STRESS_KEYS if k in d]
    xx = np.arange(len(ks))
    a.bar(xx - 0.19, [d[k][0] for k in ks], 0.38, color=C["host"], label="host alone")
    a.bar(xx + 0.19, [d[k][1] for k in ks], 0.38, color=C["v7"], label="host + V7f")
    a.set_xticks(xx, ks, fontsize=6, rotation=30)
    for i, k in enumerate(ks):
        if d[k][0] == 0:
            a.text(i - 0.19, 1.5, "0", ha="center", fontsize=6)
    a.set_title(host, fontsize=7)
    a.set_ylim(0, 72)
axs[0].set_ylabel("HOTA (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
fig.savefig(OUT / "fig5_calibration_shift.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 6: regimes and the KITTI car failure
fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw=dict(width_ratios=[1, 1.25]))
labs = list(E.REGIMES)
tot = [sum(E.REGIMES[k].values()) for k in labs]
bottom = np.zeros(len(labs))
for reg, col in (("clean", "#9ecae1"), ("cold", "#bdbdbd"), ("noisy", "#fb6a4a")):
    v = np.array([100 * E.REGIMES[k].get(reg, 0) / t for k, t in zip(labs, tot)])
    ax[0].barh(range(len(labs)), v, left=bottom, color=col, label=reg)
    bottom += v
ax[0].set_yticks(range(len(labs)), labs, fontsize=6.5)
ax[0].set_xlabel("Share of frames (%)")
ax[0].legend(frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
ax[0].set_title("(a) Regime of the post-freeze streams", fontsize=7.5, pad=16)
kc = E.RECENT["systems"]["ctwix"]["runs"]["KITTIMOT"]["classes"]["car"]["per_seq_dHOTA"]
sr = E.kitti_car_seq_regimes()
seqs = sorted(kc)
noisy = [100 * sr[s].get("noisy", 0) / sum(sr[s].values()) for s in seqs]
ax[1].bar(range(len(seqs)), [kc[s] for s in seqs], color=[C["neg"] if kc[s] < 0 else C["pos"] for s in seqs])
ax[1].set_xticks(range(len(seqs)), seqs, fontsize=6.5)
ax[1].set_ylabel("$\\Delta$HOTA, car (points)")
ax2 = ax[1].twinx()
ax2.plot(range(len(seqs)), noisy, "k^", ms=4)
ax2.set_ylabel("Noisy-regime frames (%)")
ax2.spines["right"].set_visible(True)
ax[1].set_title("(b) C-TWiX, KITTI car: per-sequence effect", fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "fig6_regimes_failure.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 7: runtime
fig, ax = plt.subplots(figsize=(4.6, 1.5))
stages = ["Frame decode", "Detector (YOLOv8n, 736 px)", "AC-MOT controller", "Tracker (ByteTrack)"]
cols = ["#c7c7c7", "#8c8c8c", "#1f77b4", "#525252"]
for yi, arm in ((1, 1), (0, 2)):
    left = 0
    for st, col in zip(stages, cols):
        r = next(x for x in E.RUNTIME if x[0] == st)
        v = r[arm][0] if r[arm] else 0
        ax.barh(yi, v, left=left, color=col, label=st if yi == 1 else None)
        left += v
    tot = next(x for x in E.RUNTIME if x[0] == "End to end")[arm][0]
    ax.text(left + 0.8, yi, f"{tot:.2f} ms end to end", va="center", fontsize=6.5)
ax.set_yticks([1, 0], ["host alone", "host + V7f"])
ax.set_xlabel("Mean time per frame (ms), 4-vCPU Xeon, no GPU")
ax.set_xlim(0, 72)
ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.55))
fig.savefig(OUT / "fig7_runtime.pdf")
plt.close(fig)
print("figures written to", OUT)
