"""Regenerate the table section of research/final/FINAL_RESULTS.md from
research/final/TABLES/*.md (run tools/v6/make_tables.py first). The
narrative above the marker line is kept verbatim."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "research/final/FINAL_RESULTS.md"
MARK = "# Generated tables"
ORDER = ["external_transfer_mot17val", "conf16_confirmation_internal", "conf16_confirmation_official", "conf16_botsort_internal",
         "conf16_botsort_official", "frcnn_val7_internal", "frcnn_val7_official",
         "frcnn_testdev_internal", "frcnn_testdev_official", "frcnn_uavdt_internal",
         "uavdt_transfer_internal", "testdev_posthoc_internal", "testdev_posthoc_official",
         "val7_development_internal", "val7_development_official", "dev40_robustness_internal"]
head = F.read_text().split(MARK)[0]
parts = [head + MARK + " (tools/v6/make_tables.py; CSV and LaTeX in TABLES/)\n"]
for t in ORDER:
    p = ROOT / "research/final/TABLES" / f"{t}.md"
    if p.exists():
        parts.append("\n" + p.read_text().replace("# ", "## ", 1))
F.write_text("".join(parts))
print("FINAL_RESULTS.md tables:", sum((ROOT / "research/final/TABLES" / f"{t}.md").exists() for t in ORDER))
