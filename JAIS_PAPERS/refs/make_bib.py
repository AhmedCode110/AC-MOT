"""
Build BibTeX entries from the Crossref/arXiv verification record verified.json
(written by verify_refs.py through the verify_refs workflow). Every entry keeps
the DOI (or arXiv identifier) that was resolved; only fields missing from the
Crossref record (proceedings title of Lecture Notes chapters, print year of a
volume, a truncated title) are completed in OVERRIDES, each with its reason.

  python JAIS_PAPERS/refs/make_bib.py <manuscript.tex> <references.bib>

writes only the entries cited in <manuscript.tex>.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = {x["key"]: x for x in json.load(open(HERE / "verified.json"))}

# Journal of Aerospace Information Systems papers found by the verification
# search (records in verified.json, key -> DOI of the chosen candidate).
JAIS = {
    "desai2017uavfeature": ("jaisquery_uav_tracking", "10.2514/1.i010503"),
    "lu2024uavdetection": ("jaisquery_uav_tracking", "10.2514/1.i011378"),
    "miller2015multitarget": ("jaisquery_uav_tracking", "10.2514/1.i010345"),
    "choi2022docking": ("jaisquery_onboard_vision", "10.2514/1.i011053"),
    "parsons2019refueling": ("jaisquery_onboard_vision", "10.2514/1.i010658"),
}
# Entries resolved by title search (status SEARCHED) whose best candidate is the paper itself.
SEARCHED_OK = ["liu2025sparsetrack", "wang2025pdsort", "yang2024hybridsort", "cao2025topictrack",
               "shim2025tracktrack", "zhao2024rtdetr", "gao2025motip"]
ARXIV_OK = ["aharon2022botsort", "ge2021yolox", "milan2016mot16"]

OVERRIDES = {
    # Lecture Notes in Computer Science chapters: Crossref gives the series, not the proceedings title.
    "zhang2022bytetrack": dict(booktitle="Computer Vision -- ECCV 2022", title="ByteTrack: Multi-object Tracking by Associating Every Detection Box"),
    "du2018uavdt": dict(booktitle="Computer Vision -- ECCV 2018"),
    "ristani2016idf1": dict(booktitle="Computer Vision -- ECCV 2016 Workshops"),
    "zeng2022motr": dict(booktitle="Computer Vision -- ECCV 2022"),
    "carion2020detr": dict(booktitle="Computer Vision -- ECCV 2020"),
    # Volume 129 of the International Journal of Computer Vision is the 2021 volume (Crossref year = online date).
    "luiten2021hota": dict(year="2021"),
    "dendorfer2021motchallenge": dict(year="2021"),
    # Crossref title truncated to the product name.
    "akiba2019optuna": dict(title="Optuna: A Next-generation Hyperparameter Optimization Framework",
                            booktitle="Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining"),
    # Crossref has no year for this proceedings record; the proceedings are ICPR-2000.
    "pechpacheco2000blur": dict(year="2000"),
    # Crossref title truncated to the system name (the DOI record's title starts with the query title's first word).
    "jiang2018chameleon": dict(title="Chameleon: Scalable Adaptation of Video Analytics"),
}
# DOI records whose Crossref title differs from the cited title and that are NOT the cited paper.
REJECTED = {"xu2020approxdet": "DOI 10.1145/3384419.3430771 resolves to an unrelated paper (RFID vibration sensing)"}


def esc(s):
    import unicodedata
    if s:
        s = s.replace("s\u0306", "\\v{s}").replace("\u0161", "\\v{s}")
        s = unicodedata.normalize("NFC", s)
        s = "".join({"\u00e9": "\\'{e}", "\u00e8": "\\`{e}", "\u00fc": '\\"{u}', "\u00f6": '\\"{o}', "\u00e4": '\\"{a}',
                     "\u00e1": "\\'{a}", "\u00ed": "\\'{i}", "\u00f3": "\\'{o}", "\u00fa": "\\'{u}", "\u00f1": "\\~{n}",
                     "\u00e7": "\\c{c}"}.get(ch, ch) for ch in s)
    s = s.replace("\u00a0", " ").replace("&amp;", "\\&").replace("&", "\\&") if s else s
    s = s.replace("\\\\&", "\\&")
    return s


def authors(lst):
    return " and ".join(esc(a) for a in lst)


def entry(key, c, extra=None):
    extra = extra or {}
    t = c["type"]
    f = {"title": "{" + esc(extra.get("title", c["title"])) + "}", "author": authors(c["authors"]),
         "year": str(extra.get("year", c.get("year") or "")), "doi": c["doi"]}
    if c.get("pages"):
        f["pages"] = c["pages"].replace("-", "--")
    if t == "journal-article":
        kind = "article"
        f["journal"] = esc(c["container"])
        if c.get("volume"):
            f["volume"] = c["volume"]
        if c.get("issue"):
            f["number"] = c["issue"]
    elif t in ("proceedings-article", "book-chapter"):
        kind = "inproceedings"
        f["booktitle"] = esc(extra.get("booktitle", c["container"]))
        if t == "book-chapter":
            f["series"] = "Lecture Notes in Computer Science"
        f["publisher"] = esc(c.get("publisher") or "")
    elif t == "book":
        kind = "book"
        f["publisher"] = esc(c.get("publisher") or "")
    else:
        raise ValueError((key, t))
    f = {k: v for k, v in f.items() if v}
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items())
    return f"@{kind}{{{key},\n{body}\n}}\n"


def arxiv_entry(key, a):
    y = a["published"][:4]
    return (f"@misc{{{key},\n  title = {{{{{esc(a['title'])}}}}},\n  author = {{{authors(a['authors'])}}},\n"
            f"  year = {{{y}}},\n  eprint = {{{a['id']}}},\n  archivePrefix = {{arXiv}},\n"
            f"  howpublished = {{arXiv preprint arXiv:{a['id']}}}\n}}\n")


def all_entries():
    out = {}
    import difflib
    for k, r in REC.items():
        if r["status"] == "DOI_RESOLVED" and k not in REJECTED:
            sim = difflib.SequenceMatcher(None, r["query_title"].lower(), r["crossref"]["title"].lower()).ratio()
            if sim < 0.9 and "title" not in OVERRIDES.get(k, {}):
                continue          # title mismatch: never emitted without an explicit, justified override
            out[k] = entry(k, r["crossref"], OVERRIDES.get(k))
    for k in SEARCHED_OK:
        b = REC[k]["best"]
        assert b["title"].lower().startswith(REC[k]["query_title"].lower()[:30]), k
        out[k] = entry(k, b, OVERRIDES.get(k))
    for k in ARXIV_OK:
        out[k] = arxiv_entry(k, REC[k]["arxiv"])
    for k, (src, doi) in JAIS.items():
        c = next(x for x in REC[src]["candidates"] if x["doi"] == doi)
        out[k] = entry(k, c)
    return out


if __name__ == "__main__":
    tex, bib = Path(sys.argv[1]), Path(sys.argv[2])
    springer = "--springer" in sys.argv   # sn-mathphys-num prints "???" for a publisher without address
    src = tex.read_text()
    cited = []
    for m in re.finditer(r"\\cite[pt]?\*?\{([^}]*)\}", src):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in cited:
                cited.append(k)
    ents = all_entries()
    missing = [k for k in cited if k not in ents]
    if missing:
        sys.exit(f"cited but not verified: {missing}")
    text = "".join(ents[k] + "\n" for k in cited)
    if springer:
        text = re.sub(r",\n  publisher = \{[^}]*\}", "", text)
    bib.write_text(text)
    print(f"{len(cited)} entries -> {bib}")
