"""
Markdown tables of the SCI + V7f cycle, generated from the result files
(no number is typed by hand):

  python tools/sci_v7/make_report.py > research/final/SCI_V7F_TABLES.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "research/final/sci_v7f"
DETN = {"yolov8": "YOLOv8n", "rtdetr": "RT-DETR-L", "retinanet": "RetinaNet"}


def load(p):
    p = F / p
    return json.loads(p.read_text()) if p.exists() else None


def ci(x, k="HOTA", nd=2):
    v = x[k]
    return f"{v['diff']:+.{nd}f} [{v['ci_lo']:+.{nd}f}, {v['ci_hi']:+.{nd}f}]"


def four_way():
    s = load("C1_5a8502f/summary.json")
    rows = [("A Native (736 px)", "NATIVE+MEDIUM"), ("B SCI only", "NATIVE+SCI"),
            ("C V7f only (736 px) = G1", "V7f+MEDIUM"), ("D SCI + V7f", "V7f+SCI"),
            ("static LOW + V7f (640 px)", "V7f+LOW"), ("static HIGH + V7f (832 px)", "V7f+HIGH"),
            ("scene-blind, same compute as D (PERM1)", "V7f+PERM1"),
            ("scene-blind, same compute as D (PERM2)", "V7f+PERM2"),
            ("scene-blind, same compute as D (PERM3)", "V7f+PERM3")]
    out = []
    for proto in ("internal", "official"):
        out.append(f"\n**{proto} protocol** (val-7, ByteTrack; candidate C1 = 5a8502f, cloud run 36998503435)\n")
        out.append("| arm | detector | HOTA | MOTA | IDF1 | IDS | FP | FN | P | R | compute | L/M/H | cat. |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, sy in rows:
            for d in ("yolov8", "rtdetr"):
                m = s[sy][d][proto]
                o = s[sy][d]["ops"]
                cat = len(s[sy][d]["catastrophic"]) if proto == "internal" else "–"
                out.append(f"| {name} | {DETN[d]} | {m['HOTA']:.2f} | {m['MOTA']:.2f} | {m['IDF1']:.2f} | {m['IDS']:.0f} | "
                           f"{m['FP']:.0f} | {m['FN']:.0f} | {m['Precision']:.1f} | {m['Recall']:.1f} | {o['rel_compute']:.3f} | "
                           + "/".join(f"{o['frac'][k]:.2f}" for k in ("LOW", "MEDIUM", "HIGH")) + f" | {cat} |")
    out.append("\nOperating statistics (V7f arms): clean / noisy regime share and intervention rate\n")
    out.append("| arm | detector | clean | noisy | intervention | level switches |")
    out.append("|---|---|---|---|---|---|")
    for name, sy in rows:
        if not sy.startswith("V7f"):
            continue
        for d in ("yolov8", "rtdetr"):
            o = s[sy][d]["ops"]
            out.append(f"| {name} | {DETN[d]} | {o['regime']['clean']:.3f} | {o['regime']['noisy']:.3f} | "
                       f"{o['intervention']:.3f} | {o['switches']} |")
    return "\n".join(out)


def boot_table(path, title):
    b = load(path)
    if not b:
        return ""
    out = [f"\n**{title}**\n", "| B − A | scope | ΔHOTA | ΔMOTA | ΔIDF1 | ΔIDS | HOTA seq +/− |", "|---|---|---|---|---|---|---|"]
    for r in b:
        scopes = list(r["per_det"].items()) + ([("pooled", r["pooled_cells"])] if "pooled_cells" in r else [])
        for sc, x in scopes:
            out.append(f"| {r['B']} − {r['A']} | {DETN.get(sc, sc)} | {ci(x)} | {ci(x, 'MOTA')} | {ci(x, 'IDF1')} | "
                       f"{ci(x, 'IDS', 0)} | {x['HOTA']['seq_wins']}/{x['HOTA']['seq_losses']} |")
    return "\n".join(out)


def runtime():
    out = []
    for d in ("yolov8", "rtdetr", "retinanet"):
        r = load(f"G1_runtime/bench_{d}.json")
        if not r:
            continue
        host = (F / f"G1_runtime/host_{d}.txt").read_text().strip()
        if not out:
            out += ["| detector | arm | detector ms | analyzer ms | scene ms | score layer ms | tracker ms | "
                    "total ms (mean / p95) | controller ms | FPS (excl. decode) | mean px |",
                    "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, v in r["arms"].items():
            out.append(f"| {DETN[d]} ({host}, {r['hardware']['threads']} threads) | {a} | {v['detector']['mean_ms']:.1f} | "
                       f"{v['analyzer']['mean_ms']:.2f} | {v['scene']['mean_ms']:.2f} | {v['score']['mean_ms']:.2f} | "
                       f"{v['tracker']['mean_ms']:.2f} | {v['total']['mean_ms']:.1f} / {v['total']['p95_ms']:.1f} | "
                       f"{v['controller_ms']['mean']:.2f} | {v['fps_excl_decode']:.1f} | {v['mean_resolution']:.0f} |")
    curves = []
    for d in ("yolov8", "rtdetr", "retinanet"):
        r = load(f"G1_runtime/latency_curve_{d}.json")
        if r:
            curves.append(f"| {DETN[d]} | " + " | ".join(f"{v['detector']['mean_ms']:.1f}" for v in r["arms"].values()) + " |")
    if curves:
        out += ["", "Detector latency (ms, mean over the timed frames; same host per detector)", "",
                "| detector | 512 | 576 | 640 | 704 | 736 | 768 | 832 | 896 | 960 |",
                "|---|---|---|---|---|---|---|---|---|---|"] + curves
    return "\n".join(out)


if __name__ == "__main__":
    print("# SCI + V7f — generated tables\n")
    print("Generated by `tools/sci_v7/make_report.py` from `research/final/sci_v7f/`.\n")
    print("## Four-way ablation, static compute points and budget-matched controls")
    print(four_way())
    print(boot_table("C1_5a8502f/bootstrap_internal.json", "Paired bootstrap, internal protocol (10,000 resamples, seed 42)"))
    print(boot_table("C1_5a8502f/bootstrap_official.json", "Paired bootstrap, official-compatible protocol"))
    print("\n## G1 with other hosts and an unseen detector")
    print(boot_table("G1_transfer_local/bootstrap_ocsort.json", "OC-SORT host, internal protocol"))
    print(boot_table("G1_transfer_local/bootstrap_ocsort_official.json", "OC-SORT host, official protocol"))
    print(boot_table("G1_transfer/bootstrap_botsort.json", "BoT-SORT host, internal protocol"))
    print(boot_table("G1_transfer/bootstrap_botsort_official.json", "BoT-SORT host, official protocol"))
    print(boot_table("G1_transfer/bootstrap_retinanet.json", "RetinaNet (unseen detector), ByteTrack, internal protocol"))
    print(boot_table("G1_transfer/bootstrap_retinanet_official.json", "RetinaNet (unseen detector), ByteTrack, official protocol"))
    print("\n## Runtime (CPU runner; no T4 was reachable)\n")
    print(runtime())
