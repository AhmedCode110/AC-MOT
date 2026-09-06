"""Rewrite v17 result labels into strict/acceptable realtime tiers."""
import argparse, csv, json
from pathlib import Path


def tier(fps, strict=25.0, acceptable=20.0):
    if fps >= strict:
        return "STRICT_REALTIME"
    if fps >= acceptable:
        return "ACCEPTABLE_REALTIME"
    return "BELOW_REALTIME"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("result", type=Path)
    p.add_argument("--strict", type=float, default=25.0)
    p.add_argument("--acceptable", type=float, default=20.0)
    a=p.parse_args()
    src=a.result/"paper_comparison_v17.csv"
    rows=list(csv.DictReader(src.open()))
    for r in rows:
        fps=float(r["FPS"])
        r["realtime_tier"]=tier(fps,a.strict,a.acceptable)
        r["acceptable_realtime"]="YES" if fps>=a.acceptable else "NO"
        r["strict_25fps"]="YES" if fps>=a.strict else "NO"
    candidates=[r for r in rows if float(r["FPS"])>=a.acceptable]
    key=lambda r:(float(r["HOTA"]),float(r["IDF1"]),float(r["MOTA"]))
    best_quality=max(rows,key=key)
    best_acceptable=max(candidates,key=key) if candidates else None
    out=a.result/"paper_comparison_v17_FINAL.csv"
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    selection={
        "strict_realtime_fps":a.strict,
        "acceptable_realtime_fps":a.acceptable,
        "selection_rule":"Highest HOTA among systems at or above acceptable realtime; IDF1 then MOTA break ties.",
        "best_quality_overall":best_quality,
        "best_acceptable_realtime":best_acceptable,
        "research_note":"25 FPS remains the strict camera-rate target. 20-24.99 FPS is reported separately as acceptable/near-real-time, not silently relabeled as 25 FPS realtime."
    }
    (a.result/"BEST_RESULT_FINAL.json").write_text(json.dumps(selection,indent=2)+"\n")
    lines=["# AC-MOT v17 Final Comparison","",f"Strict realtime: >= {a.strict:.0f} FPS; acceptable realtime: >= {a.acceptable:.0f} FPS.","","| Method | HOTA | MOTA | IDF1 | IDS | FPS | Tier |","|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['system']} | {float(r['HOTA']):.4f} | {float(r['MOTA']):.4f} | {float(r['IDF1']):.4f} | {r['IDS']} | {float(r['FPS']):.2f} | {r['realtime_tier']} |")
    lines += ["",f"Best quality overall: **{best_quality['system']}**",f"Best acceptable realtime: **{best_acceptable['system']}**" if best_acceptable else "Best acceptable realtime: **NONE**"]
    (a.result/"paper_comparison_v17_FINAL.md").write_text("\n".join(lines)+"\n")
    print("\n=== V17 FINAL REALTIME RANKING ===")
    for r in rows:
        print(f"{r['system']}: HOTA={float(r['HOTA']):.4f} FPS={float(r['FPS']):.2f} -> {r['realtime_tier']}")
    print("Best quality overall:",best_quality["system"])
    print("Best acceptable realtime:",best_acceptable["system"] if best_acceptable else "NONE")

if __name__=="__main__":main()
