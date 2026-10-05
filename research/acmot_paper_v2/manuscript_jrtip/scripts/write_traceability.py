"""Write the JRTIP numerical traceability registry from frozen artifacts only.

This script performs read-only extraction and arithmetic formatting. It never
launches a detector, tracker, search, bootstrap, or transfer evaluation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PAPER = ROOT / "research" / "acmot_paper_v2"
OUT = Path(__file__).resolve().parents[1] / "RESULT_TRACEABILITY.md"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100.0 * x:.3f}"


def agg(path: Path, key: str):
    return load(path)[key]["aggregate"]


def main() -> None:
    legacy_test = load(PAPER.parent / "paper_split" / "evidence" / "legacy" / "FINAL_TEST_RESULTS_3WORKER.json")
    legacy_uavdt = load(PAPER.parent / "paper_split" / "evidence" / "legacy" / "UAVDT_FINAL_COMPARISON.json")
    ablation_path = PAPER.parent / "paper_split" / "evidence" / "legacy" / "OLD_ACMOT_COMPONENT_ABLATION.csv"
    baseline_path = PAPER / "BASELINE_SYSTEMS_RESULT.json"
    modern_path = PAPER / "SECOND_VAL_READ_V3_RESULT.json"
    bootstrap_path = PAPER / "BOOTSTRAP_V3_RESULT.json"
    runtime_path = PAPER / "V3_RUNTIME_BENCHMARK_RESULT.json"
    baseline_runtime_path = PAPER / "RUNTIME_BENCHMARK_RESULT.json"
    transfer_path = PAPER / "UAVDT_TRANSFER_RESULT.json"
    static_path = PAPER / "FINAL_ACMOT_SYSTEMS_RESULT.json"
    freeze_path = PAPER / "controller_search" / "FREEZE_MANIFEST_V3_ADAPTIVE.json"
    shuffled_path = PAPER / "controller_search" / "SHUFFLED_CONTROL_RESULT.json"

    baseline = load(baseline_path)
    modern = load(modern_path)
    bootstrap = load(bootstrap_path)
    runtime = load(runtime_path)
    baseline_runtime = load(baseline_runtime_path)
    transfer = load(transfer_path)
    static = load(static_path)
    freeze = load(freeze_path)
    shuffled = load(shuffled_path)

    lines = [
        "# JRTIP result traceability",
        "",
        "This registry is generated from frozen JSON/CSV/manifest files. It records direct source keys and identifies simple display arithmetic; it does not run scientific experiments.",
        "",
        "## Freeze identity",
        "",
        f"- Scientific tag: `acmot-v3-exp-2026-10-04-freeze`.",
        f"- Modern freeze manifest: `{freeze_path.relative_to(ROOT)}`; frozen controller commit recorded in manifest: `{freeze['frozen_at_git_commit']}`.",
        f"- Detector SHA256: `{freeze['detector']['sha256']}` from `FREEZE_MANIFEST_V3_ADAPTIVE.json -> detector.sha256`.",
        "- Display convention: normalized legacy metrics are multiplied by 100 for percentage display; modern aggregates are already percentage-style values in their JSON records; rounded tables use three decimals unless stated otherwise.",
        "",
        "## Table and figure registry",
        "",
        "| Manuscript item | Source artifact and exact key/row | Value treatment |",
        "|---|---|---|",
        "| Table 1, historical component ablation | `research/paper_split/evidence/legacy/OLD_ACMOT_COMPONENT_ABLATION.csv`; rows `OLD-A0`, `OLD-A1`, `OLD-A2`, `OLD-A2R`, `OLD-A3` | Direct CSV values; normalized MOTA/HOTA/IDF1 shown as percentages |",
        "| Table 2, historical VisDrone | `research/paper_split/evidence/legacy/FINAL_TEST_RESULTS_3WORKER.json -> results` | Direct JSON values; normalized percentages multiplied by 100 |",
        "| Table 3, historical UAVDT | `research/paper_split/evidence/legacy/UAVDT_FINAL_COMPARISON.json`; list rows `Baseline_Frozen`, `V1_Trial24_Frozen`, `V2_Trial22_Frozen` | Direct JSON values; normalized percentages multiplied by 100 |",
        "| Table 4, v3 action map | `research/acmot_paper_v2/controller_search/FREEZE_MANIFEST_V3_ADAPTIVE.json -> controller.action_mapping` | Direct frozen tuples |",
        "| Table 5, modern VisDrone | `BASELINE_SYSTEMS_RESULT.json -> YOLO11m+ByteTrack.aggregate`, `YOLO11m+OATrack.aggregate`; `SECOND_VAL_READ_V3_RESULT.json -> AC-MOT-v3+YOLO11m+ByteTrack.aggregate`, `AC-MOT-v3+YOLO11m+OATrack.aggregate` | Direct aggregate values, rounded for display |",
        "| Table 6, modern paired bootstrap | `BOOTSTRAP_V3_RESULT.json -> bytetrack_v3_vs_system1.<metric>` and `oatrack_v3_vs_system2.<metric>` | Direct `diff`, `ci_lo`, `ci_hi` fields |",
        "| Table 7, T4 runtime | `RUNTIME_BENCHMARK_RESULT.json -> results[]` and `V3_RUNTIME_BENCHMARK_RESULT.json -> results[]` | Direct measured fields, rounded for display |",
        "| Table 8, modern UAVDT | `UAVDT_TRANSFER_RESULT.json -> <system>.aggregate` | Direct aggregate values; deltas are derived only where stated |",
        "| Table 9, attribution controls | `FINAL_ACMOT_SYSTEMS_RESULT.json -> AC-MOT+YOLO11m+<host>.aggregate`; `SHUFFLED_CONTROL_RESULT.json -> causal_cv_mean`, `shuffled_cv_mean` | Static values direct; timing comparison direct; no new control run |",
        "| Figures 1--7 | `manuscript_jrtip/figures/*.tex` generated by `scripts/make_figures.py` | Vector TikZ diagrams using the same frozen values listed above |",
        "",
        "## Extracted numerical anchors",
        "",
        "### Historical component rows",
        "",
        "| Row | MOTA | HOTA | IDF1 | IDS | FPS | Source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    with ablation_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lines.append(
                f"| {row['stage']} | {pct(float(row['MOTA']))} | {pct(float(row['HOTA']))} | {pct(float(row['IDF1']))} | {row['IDS']} | {float(row['FPS']):.3f} | `OLD_ACMOT_COMPONENT_ABLATION.csv -> {row['stage']}` |"
            )

    lines += ["", "### Modern aggregate keys", "", "| Key | HOTA | MOTA | IDF1 | IDS | FP | FN | Source |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    modern_rows = [
        ("baseline ByteTrack", baseline, "YOLO11m+ByteTrack"),
        ("v3 ByteTrack", modern, "AC-MOT-v3+YOLO11m+ByteTrack"),
        ("baseline OATrack", baseline, "YOLO11m+OATrack"),
        ("v3 OATrack", modern, "AC-MOT-v3+YOLO11m+OATrack"),
    ]
    for label, source, key in modern_rows:
        a = source[key]["aggregate"]
        source_name = "BASELINE_SYSTEMS_RESULT.json" if source is baseline else "SECOND_VAL_READ_V3_RESULT.json"
        lines.append(f"| {label} | {a['HOTA']:.6f} | {a['MOTA']:.6f} | {a['IDF1']:.6f} | {a['IDS']} | {a['FP']} | {a['FN']} | `{source_name} -> {key}.aggregate` |")

    lines += ["", "### Bootstrap and runtime keys", "", "- Bootstrap: `BOOTSTRAP_V3_RESULT.json -> bytetrack_v3_vs_system1.HOTA`, `.MOTA`, `.IDF1`, `.IDS`; corresponding OATrack values are under `oatrack_v3_vs_system2`. Global settings are `n=5000`, `seed=42`."]
    for host in ("bytetrack_v3_vs_system1", "oatrack_v3_vs_system2"):
        for metric in ("HOTA", "MOTA", "IDF1", "IDS", "FP", "FN"):
            item = bootstrap[host][metric]
            lines.append(f"- `{host}.{metric}`: diff `{item['diff']}`, CI `[{item['ci_lo']}, {item['ci_hi']}]`.")
    lines.append("- V3 runtime: `V3_RUNTIME_BENCHMARK_RESULT.json -> results[]`; baseline runtime: `RUNTIME_BENCHMARK_RESULT.json -> results[]`; GPU field is `Tesla T4` and each record has `n_frames=200`.")
    lines += ["", "### Transfer and attribution keys", ""]
    for key in ("baseline_r1536_n70+bytetrack", "v3_adaptive+bytetrack", "baseline_r1536_n70+oatrack", "v3_adaptive+oatrack"):
        a = transfer[key]["aggregate"]
        lines.append(f"- `{key}.aggregate`: HOTA `{a['HOTA']}`, MOTA `{a['MOTA']}`, IDF1 `{a['IDF1']}`, IDS `{a['IDS']}`, FP `{a['FP']}`, FN `{a['FN']}`.")
    for key in ("AC-MOT+YOLO11m+ByteTrack", "AC-MOT+YOLO11m+OATrack"):
        a = static[key]["aggregate"]
        lines.append(f"- Matched static `{key}.aggregate`: HOTA `{a['HOTA']}`, MOTA `{a['MOTA']}`, IDF1 `{a['IDF1']}`, IDS `{a['IDS']}`.")
    lines.append(f"- Shuffled timing: `SHUFFLED_CONTROL_RESULT.json -> causal_cv_mean={shuffled['causal_cv_mean']}`, `shuffled_cv_mean={shuffled['shuffled_cv_mean']}`, `seed={shuffled['seed']}`, `causal_beats_shuffled={shuffled['causal_beats_shuffled']}`.")
    lines += ["", "## Audit note", "", "`BASELINE_SYSTEMS_RESULT.json -> meta.protocol` currently says `official (tools/v6/eval_official.py)`, while `PAPER_READY_RESULTS.md`, the freeze narrative, and the manuscript describe the modern comparison as the project class-agnostic protocol. This source-metadata inconsistency is preserved as an author review issue; the manuscript does not claim official leaderboard comparability.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
