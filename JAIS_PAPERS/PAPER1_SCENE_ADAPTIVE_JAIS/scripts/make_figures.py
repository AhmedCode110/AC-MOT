"""
Figures for Paper 1 (scene-adaptive operating-point control), built only from
the evidence snapshot research/paper_split/evidence/legacy/ (see
result_provenance.md). Run from the repository root:

  python JAIS_PAPERS/PAPER1_SCENE_ADAPTIVE_JAIS/scripts/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EV = ROOT / "research/paper_split/evidence/legacy"
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
C = dict(default="#7f7f7f", heur="#1f77b4", q="#d62728", b="#2ca02c", static="#9467bd")


def rows(name):
    return list(csv.DictReader(open(EV / name)))


cfg = json.load(open(EV / "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"))
Q = cfg["optimized_parameters"]
# Trial 22 (balanced profile): freeze record section 12
B = dict(weight_crowd=.16464526567145857, weight_tiny=.17462652795045444, weight_edge=.5076112530333374,
         weight_night=.12069564693725379, weight_blur=.03242130640749567, conf_easy=.4, conf_hard=.4,
         nms_easy=.6, nms_hard=.35, threshold_mid=.2927135841069045, threshold_high=.6661671600900015)
CUES = ["crowd", "tiny", "edge", "night", "blur"]
LAB = {"crowd": "Crowd", "tiny": "Tiny-object\nratio", "edge": "Edge\ndensity", "night": "Low light", "blur": "Blur"}


def norm_w(p):
    w = np.array([p[f"weight_{k}"] for k in CUES], float)
    return w / w.sum()


# ---------------------------------------------------------------- Fig. 2
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw=dict(width_ratios=[1.1, 1]))
x = np.arange(len(CUES))
ax[0].bar(x - 0.19, norm_w(Q), 0.38, color=C["q"], label="Quality profile (Q)")
ax[0].bar(x + 0.19, norm_w(B), 0.38, color=C["b"], label="Balanced profile (B)")
ax[0].set_xticks(x, [LAB[k] for k in CUES])
ax[0].set_ylabel("Normalized weight $w_k$")
ax[0].legend(frameon=False, loc="upper left")
ax[0].set_title("(a) Frozen SCI weights", fontsize=8)
imp = json.load(open(EV / "EMPIRICAL_PARAMETER_IMPORTANCE_MOTA.json"))
names = {"threshold_a": "SCI threshold a", "threshold_b": "SCI threshold b", "nms_easy": "NMS (easy end)",
         "raw_blur": "raw weight: blur", "raw_night": "raw weight: low light", "conf_easy": "conf. (easy end)",
         "raw_crowd": "raw weight: crowd", "nms_hard": "NMS (hard end)", "raw_tiny": "raw weight: tiny",
         "raw_edge": "raw weight: edge", "conf_hard": "conf. (hard end)"}
items = sorted(imp.items(), key=lambda kv: kv[1])
ax[1].barh([names[k] for k, _ in items], [v for _, v in items], color="#555555")
ax[1].set_xlabel("Importance for validation MOTA")
ax[1].set_title("(b) Search-space importance (Q search)", fontsize=8)
fig.savefig(OUT / "fig2_sci_composition.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3
s = np.linspace(0, 1, 401)


def prof(p, res=(512, 928, 960)):
    conf = p["conf_easy"] + s * (p["conf_hard"] - p["conf_easy"])
    nms = p["nms_easy"] + s * (p["nms_hard"] - p["nms_easy"])
    size = np.where(s >= p["threshold_high"], res[2], np.where(s >= p["threshold_mid"], res[1], res[0]))
    return conf, nms, size


def heur():  # core_v17.PresentationController, scene nudges omitted (scene label 'clear')
    conf = np.clip(0.245 - 0.050 * s, 0.19, 0.28)
    nms = np.clip(0.490 - 0.050 * s, 0.40, 0.52)
    size = np.where(s > 0.60, 832, np.where(s > 0.35, 736, 640))
    return conf, nms, size


fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.1))
for (c, n, z), col, lab in [(prof(Q), C["q"], "Q (Trial 24)"), (prof(B), C["b"], "B (Trial 22)"),
                            (heur(), C["heur"], "Hand-designed")]:
    ax[0].plot(s, c, color=col, label=lab)
    ax[1].plot(s, n, color=col)
    ax[2].step(s, z, color=col, where="post")
ax[0].set_ylabel("Confidence threshold")
ax[1].set_ylabel("NMS IoU threshold")
ax[2].set_ylabel("Input resolution (px)")
for a in ax:
    a.set_xlabel("Smoothed SCI $\\bar{S}_t$")
ax[0].legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(OUT / "fig3_adaptation_map.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 4
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
r = rows("OPERATING_RESOLUTION_SWEEP.csv")
fps = [float(x["FPS"]) for x in r]; hota = [100 * float(x["HOTA"]) for x in r]; res = [int(x["resolution"]) for x in r]
sc = ax[0].scatter(fps, hota, c=res, cmap="viridis", s=14, label="resolution sweep (conf 0.25)")
for f_, h_, r_ in zip(fps, hota, res):
    if r_ in (512, 640, 800, 960):
        ax[0].annotate(str(r_), (f_, h_), textcoords="offset points", xytext=(3, -8), fontsize=6)
cr = rows("OPERATING_CONFIDENCE_SWEEP.csv")
ax[0].plot([float(x["FPS"]) for x in cr], [100 * float(x["HOTA"]) for x in cr], "s-", ms=3, lw=0.8,
           color="#ff7f0e", label="confidence sweep at 960 px")
for x_ in cr:
    if x_["conf"] in ("0.05", "0.25", "0.5"):
        ax[0].annotate(f"c={x_['conf']}", (float(x_["FPS"]), 100 * float(x_["HOTA"])), textcoords="offset points",
                       xytext=(3, 3), fontsize=6)
ax[0].set_xlabel("Processing FPS (Tesla T4)")
ax[0].set_ylabel("HOTA (%)")
ax[0].set_title("(a) Validation operating points", fontsize=8)
ax[0].legend(frameon=False, fontsize=6, loc="lower left")
plt.colorbar(sc, ax=ax[0], label="Input resolution (px)", pad=0.01)
test = {x["system"]: x for x in rows("FINAL_TEST_COMPARISON_3WORKER.csv")}
m = json.load(open(EV / "MATCHED_STATIC_A0_TESTDEV.json"))
b2 = json.load(open(EV / "V2_TRIAL22_TESTDEV_RESULT.json"))
pts = [("Static default", float(test["Baseline_Default"]["FPS"]), 100 * float(test["Baseline_Default"]["HOTA"]), C["default"], "o"),
       ("Hand-designed", float(test["Old_ACMOT_Frozen"]["FPS"]), 100 * float(test["Old_ACMOT_Frozen"]["HOTA"]), C["heur"], "o"),
       ("Q", float(test["New_ACMOT_Frozen"]["FPS"]), 100 * float(test["New_ACMOT_Frozen"]["HOTA"]), C["q"], "o"),
       ("B", b2["FPS"], 100 * b2["HOTA"], C["b"], "o"),
       ("Matched static*", m["A0_system"]["processing_fps"], 100 * m["observed_A0"]["HOTA"], C["static"], "D")]
u = {x["system"]: x for x in json.load(open(EV / "UAVDT_FINAL_COMPARISON.json"))}
upts = [("Static default", u["Baseline_Frozen"], C["default"]), ("Q", u["V1_Trial24_Frozen"], C["q"]),
        ("B", u["V2_Trial22_Frozen"], C["b"])]
for lab, f_, h_, col, mk in pts:
    ax[1].scatter(f_, h_, color=col, marker=mk, s=28, zorder=3)
    ax[1].annotate(lab, (f_, h_), textcoords="offset points", xytext=(4, 2), fontsize=6.5)
for lab, d, col in upts:
    ax[1].scatter(d["FPS"], 100 * d["HOTA"], color=col, marker="^", s=28, zorder=3)
    ax[1].annotate(lab, (d["FPS"], 100 * d["HOTA"]), textcoords="offset points", xytext=(4, -8), fontsize=6.5)
ax[1].axvline(25, color="k", lw=0.6, ls=":")
ax[1].text(25.5, 23.8, "25 FPS", fontsize=6)
ax[1].set_xlabel("Processing FPS (Tesla T4)")
ax[1].set_ylabel("HOTA (%)")
ax[1].set_title("(b) Held-out: VisDrone test-dev (o), UAVDT ($\\triangle$)", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig4_accuracy_runtime.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 5
fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.4), gridspec_kw=dict(width_ratios=[1.1, 1, 1.2]))
o = rows("OLD_ACMOT_COMPONENT_ABLATION.csv")
lab = {"OLD-A0": "H0", "OLD-A1": "H1", "OLD-A2": "H2", "OLD-A2R": "H2R", "OLD-A3": "H3"}
ax[0].bar(range(len(o)), [100 * float(x["HOTA"]) for x in o], color=C["heur"])
ax[0].set_xticks(range(len(o)), [lab[x["stage"]] for x in o], fontsize=7)
ax[0].set_ylim(28, 34)
ax[0].set_ylabel("Validation HOTA (%)")
ax[0].set_title("(a) Hand-designed controller", fontsize=8)
for i, x in enumerate(o):
    ax[0].text(i, 100 * float(x["HOTA"]) + 0.08, f"{float(x['FPS']):.0f} FPS", ha="center", fontsize=5.5)
n = rows("NEW_ACMOT_COMPONENT_ABLATION.csv")
labn = {"A0": "C0", "A1": "C1", "A2": "C2", "A3": "C3", "HIST": "H3"}
ax[1].bar(range(len(n)), [100 * float(x["HOTA"]) for x in n],
          color=[C["static"], C["q"], C["q"], C["q"], C["heur"]])
ax[1].set_xticks(range(len(n)), [labn[x["stage"]] for x in n], fontsize=7)
ax[1].set_ylim(32, 37)
ax[1].set_title("(b) Calibrated controller", fontsize=8)
t = rows("TEMPORAL_ABLATION_FULL.csv")
W = sorted({int(x["smoothing_window"]) for x in t}); S = sorted({int(x["analysis_stride"]) for x in t})
M = np.zeros((len(W), len(S)))
for x in t:
    M[W.index(int(x["smoothing_window"])), S.index(int(x["analysis_stride"]))] = 100 * float(x["HOTA"])
im = ax[2].imshow(M, cmap="Blues", aspect="auto")
ax[2].set_xticks(range(len(S)), S); ax[2].set_yticks(range(len(W)), W)
ax[2].set_xlabel("Analysis stride $K$ (frames)"); ax[2].set_ylabel("Smoothing window $W$")
for i in range(len(W)):
    for j in range(len(S)):
        ax[2].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
ax[2].add_patch(plt.Rectangle((S.index(10) - 0.5, W.index(7) - 0.5), 1, 1, fill=False, ec="r", lw=1.2))
ax[2].set_title("(c) Temporal settings, HOTA (%)", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig5_ablation.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 6
ps = rows("V1_PER_SEQUENCE_METRICS.csv")
d = {}
for x in ps:
    d.setdefault(x["system"], {})[x["sequence"]] = 100 * float(x["HOTA"])
seqs = sorted(d["Baseline_Default"])
up = rows("UAVDT_PER_SEQUENCE.csv")
e = {}
for x in up:
    e.setdefault(x["system"], {})[x["sequence"]] = 100 * float(x["HOTA"])
useq = sorted(e["Baseline_Frozen"])
fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.5))
a = np.array([d["New_ACMOT_Frozen"][s_] - d["Baseline_Default"][s_] for s_ in seqs])
b = np.array([d["New_ACMOT_Frozen"][s_] - d["Old_ACMOT_Frozen"][s_] for s_ in seqs])
order = np.argsort(a)
ax[0].scatter(range(len(seqs)), a[order], color=C["q"], s=12, label="Q $-$ static default")
ax[0].scatter(range(len(seqs)), b[order], color=C["heur"], s=12, marker="s", label="Q $-$ hand-designed")
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_xlabel("VisDrone test-dev sequence (sorted)")
ax[0].set_ylabel("$\\Delta$HOTA (points)")
ax[0].legend(frameon=False, fontsize=6)
ax[0].set_title("(a) VisDrone test-dev, 17 sequences", fontsize=8)
qa = np.array([e["V1_Trial24_Frozen"][s_] - e["Baseline_Frozen"][s_] for s_ in useq])
ba = np.array([e["V2_Trial22_Frozen"][s_] - e["Baseline_Frozen"][s_] for s_ in useq])
order = np.argsort(qa)
ax[1].scatter(range(len(useq)), qa[order], color=C["q"], s=12, label="Q $-$ static default")
ax[1].scatter(range(len(useq)), ba[order], color=C["b"], s=12, marker="s", label="B $-$ static default")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].set_xlabel("UAVDT test sequence (sorted by Q gain)")
ax[1].legend(frameon=False, fontsize=6)
ax[1].set_title("(b) UAVDT, 20 sequences, zero tuning", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig6_per_sequence.pdf")
plt.close(fig)
print("figures written to", OUT)

# ---------------------------------------------------------------- Fig. 7 (scene examples; needs figures/scene_frames/)
SF = OUT / "scene_frames"
if (SF / "raw_cues.json").exists():
    import matplotlib.image as mpimg
    cues = json.load(open(SF / "raw_cues.json"))
    summ = cfg["cue_calibration_summary"]
    band = lambda v, s: ("below Q1" if v < s["q25"] else "Q1–median" if v < s["median"]
                         else "median–Q3" if v < s["q75"] else "above Q3")
    seqs = sorted(cues)
    n = len(seqs); cols = 4; rows_ = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows_, cols, figsize=(6.9, 1.55 * rows_ + 0.2))
    for a in np.ravel(ax):
        a.axis("off")
    for a, sq in zip(np.ravel(ax), seqs):
        c = cues[sq]
        a.imshow(mpimg.imread(SF / f"{sq}_0000001.jpg"))
        a.set_title(sq.replace("uav", "").replace("_v", ""), fontsize=6.5)
        a.text(0.0, -0.04, f"gray {c['brightness']:.0f} ({band(c['brightness'], summ['brightness'])})\n"
                           f"Laplacian var. {c['blur']:.0f} ({band(c['blur'], summ['blur'])})\n"
                           f"gradient {c['edges']:.1f} ({band(c['edges'], summ['edge'])})",
               transform=a.transAxes, va="top", fontsize=5.5)
    fig.tight_layout(h_pad=2.6)
    fig.savefig(OUT / "fig7_scene_examples.pdf")
    plt.close(fig)
    print("fig7 written")
