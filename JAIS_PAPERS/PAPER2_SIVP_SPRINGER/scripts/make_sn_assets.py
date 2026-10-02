"""
Springer (two-column) copies of the Paper 2 tables and figures. The numbers come from
../PAPER2_UNIVERSAL_ACMOT_JAIS/scripts (make_tables.py, make_figures.py, evidence.py);
this script only changes the float environment and copies the figure files.
  python JAIS_PAPERS/PAPER2_SIVP_SPRINGER/scripts/make_sn_assets.py
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SRC = HERE.parent / "PAPER2_UNIVERSAL_ACMOT_JAIS"
(HERE / "tables").mkdir(exist_ok=True)
(HERE / "figures").mkdir(exist_ok=True)
for n in ["tab1_contracts.tex", "tab3_mot17.tex", "tab4_kitti.tex", "tab5_external.tex"]:
    s = (SRC / "tables" / n).read_text()
    s = s.replace("\\begin{table}[t]", "\\begin{table*}[t]").replace("\\end{table}", "\\end{table*}")
    s = s.replace("\\small", "\\footnotesize").replace("\\hline\n\\end{tabular}", "\\botrule\n\\end{tabular}")
    (HERE / "tables" / n).write_text(s)
for n in ["fig3_forest_col.pdf", "fig5_calibration_shift.pdf", "fig6_regimes_failure.pdf"]:
    shutil.copy(SRC / "figures" / n, HERE / "figures" / n)
print("written to", HERE)
