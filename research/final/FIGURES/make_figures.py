"""Publication figures from the V6 result files (reads outputs/v6/ only).

  PYTHONPATH=. .venv/bin/python research/final/FIGURES/make_figures.py

fig_thresholds       adaptive behaviour: per-frame primary threshold t2
                     (as a raw score sigmoid(t2)) per sequence and detector
fig_catastrophic     catastrophic cells (MOTA < 0) per system and split
fig_seq_scatter      per-sequence MOTA, V6-TF vs V4 (all post-freeze splits)
fig_operating_points precision vs recall per system / detector / split
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.v6.dev import load, split_sequences, summary  # noqa: E402

OUT = Path(__file__).resolve().parent
SYS = ["static_default", "shared_static", "V4", "E41", "V6TF"]
NAME = {"static_default": "tracker default", "shared_static": "shared static 0.5",
        "V4": "V4 (tuned)", "E41": "E41", "V6TF": "V6-TF (ours)"}
DETS = ["yolov8", "rtdetr"]


def fig_thresholds():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, split in zip(axes, ("val7", "conf16")):
        pos, data, lab = 0, [], []
        for d in DETS:
            for s in split_sequences(split):
                a = load(split, "V6TF", d, s)["audit"]
                t2 = np.array([x.get("tf_otsu_t2", np.nan) for x in a], float)
                t2 = t2[np.isfinite(t2)]
                data.append(1 / (1 + np.exp(-t2)))
                lab.append(d)
        colors = ["tab:blue" if l == "yolov8" else "tab:orange" for l in lab]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xticks([])
        ax.set_title(f"{split}: per-sequence primary threshold")
        ax.set_xlabel("sequences (blue YOLOv8n | orange RT-DETR-L)")
    axes[0].set_ylabel("self-calibrated primary threshold (raw score)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_thresholds.pdf")
    fig.savefig(OUT / "fig_thresholds.png", dpi=160)


def fig_catastrophic():
    splits = [("val7", "val-7 (dev)"), ("dev40", "dev-40 (dev)"), ("conf16", "confirm-16"),
              ("testdev", "test-dev (post-hoc)"), ("uavdt", "UAVDT")]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    w = 0.16
    for k, sy in enumerate(SYS):
        vals = []
        for sp, _ in splits:
            name = "X5" if (sy == "V6TF" and sp == "dev40") else sy
            try:
                r = summary(sp, name)
                vals.append(sum(len(r[d]["cat"]) for d in DETS))
            except FileNotFoundError:
                vals.append(np.nan)
        ax.bar(np.arange(len(splits)) + (k - 2) * w, vals, w, label=NAME[sy])
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([s[1] for s in splits])
    ax.set_ylabel("catastrophic cells (MOTA<0)")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_catastrophic.pdf")
    fig.savefig(OUT / "fig_catastrophic.png", dpi=160)


def fig_seq_scatter():
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for sp, mk in (("conf16", "o"), ("testdev", "s"), ("uavdt", "^")):
        for d, c in zip(DETS, ("tab:blue", "tab:orange")):
            try:
                a, b = summary(sp, "V4")[d]["per"], summary(sp, "V6TF")[d]["per"]
            except FileNotFoundError:
                continue
            x = [max(a[s]["MOTA"], -60) for s in a]
            y = [max(b[s]["MOTA"], -60) for s in a]
            ax.scatter(x, y, marker=mk, c=c, s=18, alpha=0.7, label=f"{sp} {d}")
    ax.plot([-60, 70], [-60, 70], "k--", lw=0.8)
    ax.set_xlabel("V4 per-sequence MOTA (clipped at −60)")
    ax.set_ylabel("V6-TF per-sequence MOTA")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_seq_scatter.pdf")
    fig.savefig(OUT / "fig_seq_scatter.png", dpi=160)


def fig_operating_points():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, sp in zip(axes, ("conf16", "testdev", "uavdt")):
        for sy, mk in zip(SYS, "vDsxo"):
            try:
                r = summary(sp, sy)
            except FileNotFoundError:
                continue
            for d, c in zip(DETS, ("tab:blue", "tab:orange")):
                ax.scatter(r[d]["Recall"], r[d]["Precision"], marker=mk, c=c, s=40,
                           label=f"{NAME[sy]} {d}")
        ax.set_title(sp)
        ax.set_xlabel("recall (%)")
    axes[0].set_ylabel("precision (%)")
    axes[-1].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_operating_points.pdf")
    fig.savefig(OUT / "fig_operating_points.png", dpi=160)


if __name__ == "__main__":
    for f in (fig_thresholds, fig_catastrophic, fig_seq_scatter, fig_operating_points):
        f()
        print("wrote", f.__name__)
