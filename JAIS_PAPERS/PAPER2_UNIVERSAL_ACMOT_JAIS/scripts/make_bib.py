"""
references.bib for Paper 2: only the entries cited in manuscript.tex, built
from the verification records JAIS_PAPERS/refs/verified.json (shared with
Paper 1, via refs/make_bib.py) and JAIS_PAPERS/refs/verified_paper2.json
(verify_refs_paper2.py). The only manual entry is the software reference of
the Ultralytics library (no DOI; version pinned in research/final/env).
Fails if a cited key has no verified record.

  python JAIS_PAPERS/PAPER2_UNIVERSAL_ACMOT_JAIS/scripts/make_bib.py
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
    # The author's companion manuscript (Paper 1), submitted separately; disclosed in the cover letter.
    "gouda2026sceneadaptive": (
        "@misc{gouda2026sceneadaptive,\n  title = {{Scene-Adaptive Detector Operating-Point Control for Unmanned Aerial "
        "Vehicle Multi-Object Tracking}},\n  author = {Ahmed Gouda},\n  year = {2026},\n"
        "  howpublished = {Manuscript submitted to the Journal of Aerospace Information Systems}\n}\n"),
    "jocher2023ultralytics": (
        "@misc{jocher2023ultralytics,\n  title = {{Ultralytics YOLO}},\n"
        "  author = {Glenn Jocher and Jing Qiu and Ayush Chaurasia},\n  year = {2023},\n"
        "  howpublished = {Software, \\url{https://github.com/ultralytics/ultralytics}, version 8.3.200 used}\n}\n"),
}


def entries():
    out = mb.all_entries()
    for r in json.load(open(REFS / "verified_paper2.json")):
        k = r["key"]
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
    (PAPER / "references.bib").write_text("".join(ents[k] + "\n" for k in cited))
    print(f"{len(cited)} entries -> references.bib")
