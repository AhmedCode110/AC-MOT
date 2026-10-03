"""
references.bib for the General AC-MOT manuscript: only the entries cited in
manuscript.tex, built from the Crossref/arXiv verification records
JAIS_PAPERS/refs/verified.json (through refs/make_bib.py) and
JAIS_PAPERS/refs/verified_paper2.json. The only manual entry is the software
reference of the Ultralytics library (no DOI; version 8.3.200 recorded in the
run environments). Fails if a cited key has no verified record.

  python scripts/make_bib.py
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REFS = PAPER.parent / "refs"
spec = importlib.util.spec_from_file_location("mb", REFS / "make_bib.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

OVERRIDES2 = {
    # Crossref gives the series for Lecture Notes chapters, not the proceedings title.
    "zhou2020centertrack": dict(booktitle="Computer Vision -- ECCV 2020"),
}
MANUAL = {
    "jocher2023ultralytics": (
        "@misc{jocher2023ultralytics,\n  title = {{Ultralytics YOLO}},\n"
        "  author = {Glenn Jocher and Jing Qiu and Ayush Chaurasia},\n  year = {2023},\n"
        "  howpublished = {Software, \\url{https://github.com/ultralytics/ultralytics}, version 8.3.200}\n}\n"),
}


def entries():
    out = mb.all_entries()
    for r in json.load(open(REFS / "verified_paper2.json")):
        k = r["key"]
        if k in out:
            continue
        if r["status"] == "DOI_RESOLVED":
            out[k] = mb.entry(k, r["crossref"], OVERRIDES2.get(k))
        elif r["status"] == "ARXIV_RESOLVED":
            out[k] = mb.arxiv_entry(k, r["arxiv"])
    out.update(MANUAL)
    return out


if __name__ == "__main__":
    src = (PAPER / "manuscript.tex").read_text()
    cited = []
    for m in re.finditer(r"\\cite[pt]?\*?\{([^}]*)\}", src):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in cited:
                cited.append(k)
    ents = entries()
    missing = [k for k in cited if k not in ents]
    if missing:
        sys.exit(f"cited but not verified: {missing}")
    text = "".join(ents[k] + "\n" for k in cited)
    # sn-mathphys-num prints "???" for a publisher without an address (as for the SIVP manuscript)
    text = re.sub(r",\n  publisher = \{[^}]*\}", "", text)
    (PAPER / "references.bib").write_text(text)
    print(f"{len(cited)} entries -> references.bib")
