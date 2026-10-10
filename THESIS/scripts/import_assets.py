"""
Copy the figures and tables of the two journal manuscripts into the thesis.
Labels are prefixed with the chapter (tab:ch4-..., tab:ch5-...) so that the
two sets can coexist; table bodies and numbers are copied unchanged.

  python THESIS/scripts/import_assets.py
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / "THESIS"
SRC = {
    "ch4": ROOT / "JAIS_PAPERS" / "PAPER1_SCENE_ADAPTIVE_JAIS",
    "ch5": ROOT / "JAIS_PAPERS" / "PAPER2_UNIVERSAL_ACMOT_JAIS",
}


def main():
    for ch, src in SRC.items():
        for f in sorted((src / "figures").glob("*.pdf")):
            if f.stem.endswith("_col"):
                continue
            shutil.copy(f, THESIS / "figures" / f"{ch}_{f.name}")
        for f in sorted((src / "tables").glob("*.tex")):
            t = f.read_text()
            t = re.sub(r"\\(label|ref)\{(tab|sec|fig|eq):", lambda m: f"\\{m.group(1)}{{{m.group(2)}:{ch}-", t)
            t = t.replace("\\begin{table}[t]", "\\begin{table}[tbp]")
            t = t.replace("\\small\n", "\\small\\setlength{\\tabcolsep}{4pt}\n")
            t = re.sub(r"(?<!arraybackslash})p\{([0-9.]+cm)\}", r">{\\raggedright\\arraybackslash}p{\1}", t)
            t = t.replace("p{1.2cm}}", "p{1.4cm}}")
            if "resizebox" in t:
                # wide tables are set in landscape so that they keep a readable size
                t = t.replace("\\begin{table}[tbp]", "\\begin{sidewaystable}").replace("\\end{table}", "\\end{sidewaystable}")
                t = t.replace("\\resizebox{\\textwidth}{!}{%", "\\begin{adjustbox}{max width=\\textheight}")
                t = t.replace("\\end{tabular}}", "\\end{tabular}\n\\end{adjustbox}")
            else:
                t = re.sub(r"(\\begin\{tabular\}.*?\\end\{tabular\})",
                           lambda m: "\\begin{adjustbox}{max width=\\textwidth}\n" + m.group(1) + "\n\\end{adjustbox}", t, flags=re.S)
            (THESIS / "tables" / f"{ch}_{f.name}").write_text(t)
    print("assets copied")


if __name__ == "__main__":
    main()
