"""Publication tables + per-sequence figure for the external published-system
transfer (reads tools/v6/external/vendor/external_results.json and the
bootstrap JSONs; never re-runs anything)."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

V = Path(__file__).resolve().parent / "vendor"
OUT = Path(__file__).resolve().parents[3] / "research/final/TABLES"
FIG = Path(__file__).resolve().parents[3] / "research/final/FIGURES"
R = json.load(open(V / "external_results.json"))


def boot(name):
    t = open(V / name).read()
    return json.loads(t[t.index("{"):])


ROWS = [
    ("SparseTrack (IEEE TCSVT 2025) — paper, Table VI", dict(MOTA=76.8, HOTA=69.2, IDF1=81.4)),
    ("SparseTrack — our faithful execution (official code+ckpt)", R["ST_A_official"]["trackeval"]),
    ("SparseTrack + frozen V6-TF (same detections)", R["ST_plus_V6TF"]["trackeval"]),
    ("  diagnostic: + V6-TF without duplicate suppression", R["ST_DIAG_dedup_iou_0"]["trackeval"]),
    ("  diagnostic: + V6-TF without motion rule", R["ST_DIAG_assoc_motion_0"]["trackeval"]),
    ("BoostTrack (MVA 2024) — authors' re-reported online (issue #8)", dict(MOTA=75.561, HOTA=68.371, IDF1=81.354, IDS=118)),
    ("BoostTrack — our execution (online)", R["BT_replay_baseline"]["trackeval"]),
    ("BoostTrack + frozen V6-TF (online)", R["BT_plus_V6TF"]["trackeval"]),
    ("BoostTrack — authors' re-reported + GBI", dict(MOTA=80.549, HOTA=71.326, IDF1=83.839, IDS=106)),
    ("BoostTrack — our execution + GBI", R["BT_replay_baseline_post_gbi"]["trackeval"]),
    ("BoostTrack + frozen V6-TF + GBI", R["BT_plus_V6TF_post_gbi"]["trackeval"]),
]
COLS = ["MOTA", "HOTA", "IDF1", "IDS", "FP", "FN", "Precision", "Recall"]


def fmt(v):
    return "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "external_transfer_mot17val.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["System"] + COLS)
    for n, r in ROWS:
        w.writerow([n] + [r.get(c) for c in COLS])
md = ["# External transfer — MOT17 val-half, TrackEval MOTChallenge protocol "
      "(paper rows as published; Precision/Recall from TrackEval CLEAR)", "",
      "| System | MOTA ↑ | HOTA ↑ | IDF1 ↑ | IDS ↓ | FP ↓ | FN ↓ | Prec ↑ | Rec ↑ |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
md += [f"| {n} | " + " | ".join(fmt(r.get(c)) for c in COLS) + " |" for n, r in ROWS]
st, bo, bg = boot("boot_st.json"), boot("boot_bt_online.json"), boot("boot_bt_gbi.json")
md += ["", "## Δ frozen V6-TF (paired sequence bootstrap, 10,000 resamples, seed 42, 95% CI)", "",
       "| Host system | ΔMOTA | ΔHOTA | ΔIDF1 | ΔIDS | ΔFP | ΔFN |", "|---|---|---|---|---|---|---|"]
for n, b in (("SparseTrack", st), ("BoostTrack (online)", bo), ("BoostTrack + GBI", bg)):
    md.append(f"| {n} | " + " | ".join(f"{b[k]['diff']:+.2f} [{b[k]['ci_lo']:.2f}, {b[k]['ci_hi']:.2f}]"
                                          for k in ("MOTA", "HOTA", "IDF1", "IDS", "FP", "FN")) + " |")
(OUT / "external_transfer_mot17val.md").write_text("\n".join(md) + "\n")
tex = ["% External transfer, MOT17 val-half (TrackEval)", "\\begin{tabular}{lrrrrrr}", "\\toprule",
       "System & MOTA & HOTA & IDF1 & IDS & FP & FN \\\\", "\\midrule"]
tex += [f"{n.replace('&', '\\&')} & " + " & ".join(fmt(r.get(c)) for c in COLS[:6]) + " \\\\" for n, r in ROWS]
tex += ["\\bottomrule", "\\end{tabular}"]
(OUT / "external_transfer_mot17val.tex").write_text("\n".join(tex) + "\n")

# per-sequence figure
seqs = list(R["ST_A_official"]["per_seq"].keys())
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, k in zip(axes, ("HOTA", "MOTA", "IDF1")):
    x = np.arange(len(seqs))
    for off, (lab, key) in zip((-0.3, -0.1, 0.1, 0.3), (("SparseTrack", "ST_A_official"),
                                                        ("+V6-TF", "ST_plus_V6TF"),
                                                        ("BoostTrack", "BT_replay_baseline"),
                                                        ("+V6-TF", "BT_plus_V6TF"))):
        ax.bar(x + off, [R[key]["per_seq"][s][k] for s in seqs], 0.2, label=f"{lab}" if k == "HOTA" else None)
    ax.set_xticks(x)
    ax.set_xticklabels([s.split("-")[1] for s in seqs])
    ax.set_title(f"MOT17 val-half per-sequence {k}")
axes[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(FIG / "fig_external_per_sequence.pdf")
fig.savefig(FIG / "fig_external_per_sequence.png", dpi=160)
print("\n".join(md))
