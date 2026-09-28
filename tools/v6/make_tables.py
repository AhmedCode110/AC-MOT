"""Publication tables (Markdown / CSV / LaTeX) from the V6 result files.
Reads only outputs/v6/<split>/ artifacts; never re-runs anything.

  python tools/v6/make_tables.py            # writes research/final/TABLES/
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.v6.dev import load, out_dir, split_sequences  # noqa: E402

OUT = ROOT / "research/final/TABLES"
LABEL = {"V6TF": "V6-TF (ours, frozen)", "V4": "V4 (frozen, VisDrone-tuned)",
         "shared_static": "Shared static (raw 0.5)", "static_default": "Tracker default (raw 0.25)",
         "E41": "E41 (rejected V5-TF lock)"}
COLS = ["n_catastrophic", "MOTA", "HOTA", "IDF1", "IDS", "FP", "FN", "Precision", "Recall"]


def pooled(split, systems, dets, proto):
    from tools.seqstats import combine
    from tools.v6.eval_official import combine_official
    rows = []
    for sy in systems:
        for d in dets:
            seqs = split_sequences(split)
            if proto == "internal":
                st = [load(split, sy, d, s) for s in seqs]
                comb = combine
            else:
                p = [out_dir(split) / sy / d / f"{s}.official.pkl" for s in seqs]
                if not all(x.exists() for x in p):
                    continue
                st = [pickle.load(open(x, "rb")) for x in p]
                comb = combine_official
            m = comb(st)
            m["n_catastrophic"] = sum(comb([x])["MOTA"] < 0 for x in st)
            rows.append(dict(split=split, protocol=proto, detector=d,
                             system=LABEL.get(sy.split("@")[0], sy) + (" + BoT-SORT" if "botsort" in sy else ""),
                             **{c: m[c] for c in COLS}))
    return rows


def write(name, rows, title):
    if not rows:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    md = [f"# {title}", "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n")
    num = [k for k in keys if k in COLS]
    tex = ["% " + title, "\\begin{tabular}{ll" + "r" * len(num) + "}", "\\toprule",
           "Detector & System & " + " & ".join(k.replace("n_catastrophic", "Cat.") for k in num) + " \\\\",
           "\\midrule"]
    tex += [f"{r['detector']} & {r['system']} & " + " & ".join(
        (f"{r[k]:.1f}" if isinstance(r[k], float) else str(r[k])) for k in num) + " \\\\"
        for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n")


def boot_table(split):
    f = out_dir(split) / "bootstrap.json"
    if not f.exists():
        return []
    return [dict(split=split, **{k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in b.items()}) for b in json.load(open(f))]


SPECS = [
    ("val7_development", "val7", ["V4", "shared_static", "static_default", "E41", "V6TF"], ["yolov8", "rtdetr"],
     "val-7 (DEVELOPMENT sandbox; V4 in-sample)"),
    ("dev40_robustness", "dev40", ["V4", "shared_static", "static_default", "E41", "X5"], ["yolov8", "rtdetr"],
     "development-40 robustness check (not iterated; X5 = V6-TF)"),
    ("conf16_confirmation", "conf16", ["V4", "shared_static", "static_default", "E41", "V6TF"], ["yolov8", "rtdetr"],
     "confirmation-16 (ONE-WAY, post-freeze)"),
    ("conf16_botsort", "conf16", ["static_default@trk:botsort", "shared_static@trk:botsort", "V4@trk:botsort",
                                  "V6TF@trk:botsort"], ["yolov8", "rtdetr"],
     "tracker transfer: BoT-SORT on confirmation-16 (post-freeze)"),
    ("testdev_posthoc", "testdev", ["V4", "shared_static", "static_default", "V6TF"], ["yolov8", "rtdetr"],
     "VisDrone test-dev (POST-HOC; V4 held-out E31)"),
    ("uavdt_transfer", "uavdt", ["V4", "shared_static", "static_default", "V6TF"], ["yolov8", "rtdetr"],
     "UAVDT test (dataset transfer, post-freeze)"),
    ("frcnn_val7", "val7", ["V4", "shared_static", "static_default", "V6TF"], ["fasterrcnn"],
     "Faster R-CNN (unseen detector) on val-7 (post-freeze)"),
    ("frcnn_testdev", "testdev", ["V4", "shared_static", "static_default", "V6TF"], ["fasterrcnn"],
     "Faster R-CNN (unseen detector) on test-dev (post-hoc)"),
    ("frcnn_uavdt", "uavdt", ["V4", "shared_static", "static_default", "V6TF"], ["fasterrcnn"],
     "Faster R-CNN on UAVDT (post-freeze)"),
]


def main():
    for name, split, systems, dets, title in SPECS:
        for proto in ("internal", "official"):
            if proto == "official" and split == "uavdt":
                continue
            try:
                rows = pooled(split, systems, dets, proto)
            except FileNotFoundError:
                continue
            write(f"{name}_{proto}", rows, f"{title} — {proto} protocol")
    for split in ("conf16", "testdev", "uavdt"):
        write(f"bootstrap_{split}", boot_table(split),
              f"paired sequence bootstrap {split} (10,000 resamples, seed 42)")
    print("tables:", sorted(p.name for p in OUT.glob("*.md")))


if __name__ == "__main__":
    main()
